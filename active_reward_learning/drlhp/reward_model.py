import io
from zipfile import ZipFile

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.running_mean_std import RunningMeanStd

import active_reward_learning
from active_reward_learning.drlhp.buffer import SampleBuffer


class TrajectorySequence:
    def __init__(self, observations, actions, rewards):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards


class RewardModelNN(nn.Module):
    def __init__(self, in_size, hidden_size=64, embedding_size=64, l2_reg=0):
        super(RewardModelNN, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, embedding_size, bias=False)
        self.fc3 = nn.Linear(embedding_size, 1, bias=False)
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg

    def forward_without_last_layer(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        return x

    def forward(self, x):
        x = self.forward_without_last_layer(x)
        x = self.fc3(x)
        return x

    def eval(self, test_data):
        assert test_data.observation_type == "value"
        test_x, test_y = test_data.get_all()
        y_pred = self.forward(torch.Tensor(test_x)).detach().numpy()[:, :, 0]
        y_pred = y_pred.sum(axis=-1)
        pcor = np.corrcoef(np.reshape(y_pred, -1), np.reshape(test_y, -1))
        pdis = np.mean(np.sqrt((1 - pcor) / 2))

        n = y_pred.shape[0]
        m = n // 2
        y_pred1 = np.array(y_pred)[:m]
        y_pred2 = np.array(y_pred)[m : 2 * m]
        test_y1 = np.array(test_y)[:m]
        test_y2 = np.array(test_y)[m : 2 * m]

        acc = np.mean(
            np.logical_or(
                np.logical_and(y_pred1 > y_pred2, test_y1 > test_y2),
                (test_y1 <= test_y2),
            )
        )

        P_1gt2 = np.exp(y_pred1) / (np.exp(y_pred1) + np.exp(test_y2))
        P_1gt2 = np.clip(P_1gt2, 0.01, 0.99)
        validation_loss = -np.mean(
            np.array(test_y1 >= test_y2, dtype=np.float) * np.log(P_1gt2)
            + np.array(test_y2 > test_y1, dtype=np.float) * np.log(1 - P_1gt2)
        )

        return acc, pdis, validation_loss

    def train(
        self,
        data,
        batch_size=50,
        n_epochs=10,
        comparisons=True,
        validation_data=None,
        train_only_last_layer=False,
    ):
        if validation_data is not None:
            assert validation_data.observation_type == "value"
        else:
            acc, pdis = None, None
        avg_var = None

        if train_only_last_layer:
            self.fc1.weight.requires_grad = False
            self.fc2.weight.requires_grad = False
            self.fc3.weight.requires_grad = True
        else:
            self.fc1.weight.requires_grad = True
            self.fc2.weight.requires_grad = True
            self.fc3.weight.requires_grad = True

        optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=self.l2_reg)

        if comparisons:
            for epoch in range(n_epochs):
                print("Epoch:", epoch, flush=True)
                x_batches, y_batches = data.get_all_batches(batch_size)

                for b, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
                    seq1, seq2, labels = [], [], []
                    for i in range(len(x_batch)):
                        s1, s2 = x_batch[i]
                        seq1.append(s1)
                        seq2.append(s2)
                        labels.append(y_batch[i])
                    seq1 = torch.Tensor(seq1)
                    seq2 = torch.Tensor(seq2)
                    labels = torch.Tensor(labels)

                    optimizer.zero_grad()

                    pred1 = torch.sum(self.forward(seq1), axis=(1, 2))
                    pred2 = torch.sum(self.forward(seq2), axis=(1, 2))

                    pred1 = torch.clip(pred1, -10, 10)
                    pred2 = torch.clip(pred2, -10, 10)

                    P_1gt2 = torch.exp(pred1) / (torch.exp(pred1) + torch.exp(pred2))
                    P_1gt2 = torch.clip(P_1gt2, 0.01, 0.99)

                    loss = -torch.mean(
                        labels[:, 0] * torch.log(P_1gt2)
                        + labels[:, 1] * torch.log(1 - P_1gt2)
                    )

                    if torch.isnan(loss):
                        print("Warning: nan loss", flush=True)

                    loss.backward()
                    optimizer.step()
                    print(
                        f"batch: {b}, loss: {loss}",
                        flush=True,
                    )

                if validation_data is not None:
                    acc, pdis, validation_loss = self.eval(validation_data)
                    print(f"acc: {acc}, pdis: {pdis}", flush=True)
        else:
            for epoch in range(n_epochs):
                print("Epoch:", epoch, flush=True)
                x_batches, y_batches = data.get_all_batches(batch_size)
                for b, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
                    x_batch = torch.Tensor(x_batch)
                    y_batch = torch.Tensor(y_batch)
                    y_pred = self.forward(x_batch)
                    y_pred = y_pred.sum(-1)
                    optimizer.zero_grad()
                    loss = torch.mean((y_pred - torch.reshape(y_batch, (-1, 1))) ** 2)
                    loss.backward()
                    optimizer.step()
                    print(
                        f"batch: {b}, loss: {loss}",
                        flush=True,
                    )

                if validation_data is not None:
                    acc, pdis, validation_loss = self.eval(validation_data)
                    print(f"acc: {acc}, pdis: {pdis}", flush=True)
        return float(loss.detach().numpy()), validation_loss, acc, pdis, avg_var

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "in_size": self.in_size,
                "hidden_size": self.hidden_size,
                "embedding_size": self.embedding_size,
                "l2_reg": self.l2_reg,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        model_dict = torch.load(path)
        model = cls(
            model_dict["in_size"],
            hidden_size=model_dict["hidden_size"],
            embedding_size=model_dict["embedding_size"],
            l2_reg=model_dict["l2_reg"],
        )
        model.load_state_dict(model_dict["model_state_dict"])
        return model


class RewardModelEnsemble:
    """Might be deprecated"""

    def __init__(self, n_models, in_size, hidden_size=64):
        self.n_models = n_models
        self.models = []
        self.bootstrap_samples = []
        for _ in range(n_models):
            model = RewardModelNN(in_size, hidden_size=hidden_size)
            self.models.append(model)
        self.bootstrapped_until_idx = 0

    def train(
        self, data, batch_size=50, n_epochs=10, comparisons=True, validation_data=None
    ):
        n = len(data)
        loss_list, validation_loss_list, acc_list, pdis_list = [], [], [], []
        for i in range(self.n_models):
            if i >= len(self.bootstrap_samples):
                self.bootstrap_samples.append(SampleBuffer(data.observation_type))
            sample = self.bootstrap_samples[i]
            # online bootstrap by sampling from poisson distribution
            if len(data) > self.bootstrapped_until_idx:
                j = self.bootstrapped_until_idx
                items, labels = data.get_all()
                while j < len(data):
                    n_j = np.random.poisson(lam=1)
                    for k in range(n_j):
                        sample.add_single(items[j], labels[j])
                    j += 1
            print("len(sample)", len(sample))
            loss, validation_loss, acc, pdis = self.models[i].train(
                sample,
                batch_size=batch_size,
                n_epochs=n_epochs,
                comparisons=comparisons,
                validation_data=validation_data,
            )
            loss_list.append(loss)
            validation_loss_list.append(validation_loss)
            acc_list.append(acc)
            pdis_list.append(pdis)
        self.bootstrapped_until_idx = len(data)
        loss = np.mean(loss_list)
        validation_loss = np.mean(validation_loss_list)
        acc = np.mean(acc_list) if not None in acc_list else None
        pdis = np.mean(pdis_list) if not None in pdis_list else None
        avg_var = None
        return loss, validation_loss, acc, pdis, avg_var

    def predict_all(self, x):
        y = []
        for model in self.models:
            y_ = model.forward(torch.Tensor(x)).detach().numpy()
            y.append(y_)
        return y

    def ensemble_predict(self, x):
        y = self.predict_all(x)
        return np.mean(y, axis=0), np.std(y, axis=0)

    def save(self, path):
        assert path.endswith(".zip")
        with ZipFile(path, "w") as zf:
            for i, model in enumerate(self.models):
                with zf.open("%02i.pt" % i, "w") as f:
                    model.save(f)

    @classmethod
    def load(cls, path):
        assert path.endswith(".zip")
        models = []
        with ZipFile(path, "r") as zf:
            for filename in zf.namelist():
                with zf.open(filename, "r") as f:
                    model = RewardModelNN.load(io.BytesIO(f.read()))
                    models.append(model)
        n_models = len(models)
        in_size = models[0].in_size
        hidden_size = models[0].hidden_size
        ensemble = cls(n_models, in_size, hidden_size)
        ensemble.models = models
        return ensemble


class RewardModelAdaptiveBayesianLinear:
    def __init__(self, in_size, hidden_size=64, embedding_size=64):
        self.l2_reg = 0.5
        self.basis_fct_model = RewardModelNN(
            in_size,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            l2_reg=self.l2_reg,
        )
        self.mu = self.basis_fct_model.fc3.weight.data.detach().numpy()[0, :]
        self.H = self.l2_reg * np.identity(self.mu.shape[0])
        self.cov = (1 / self.l2_reg) * np.identity(self.mu.shape[0])

    def get_features(self, x):
        return (
            self.basis_fct_model.forward_without_last_layer(torch.Tensor(x))
            .detach()
            .numpy()
        )

    def forward(self, x):
        return self.basis_fct_model.forward(x)

    def train(
        self,
        data,
        batch_size=50,
        n_epochs=10,
        comparisons=True,
        validation_data=None,
        train_basis_functions=True,
    ):
        assert comparisons

        loss, validation_loss, acc, pdis, avg_var = self.basis_fct_model.train(
            data,
            batch_size=batch_size,
            n_epochs=n_epochs,
            comparisons=comparisons,
            validation_data=validation_data,
            train_only_last_layer=(not train_basis_functions),
        )
        print(loss, acc, pdis)

        # Laplace approximation
        # use last layer as MAP weights
        self.mu = self.basis_fct_model.fc3.weight.data.detach().numpy()[0, :]

        # compute Hessian
        X, y = data.get_all()
        X = np.array(X)
        X = self.get_features(X)
        y = np.array(y)

        w = self.mu
        y_pred = np.dot(X, w).sum(-1)
        diff = y_pred[:, 0] - y_pred[:, 1]
        P_1gt2 = 1 / (1 + np.exp(-diff))
        P_2gt1 = 1 - P_1gt2

        deltaX = X.sum(2)
        deltaX = deltaX[:, 0, :] - deltaX[:, 1, :]

        self.H = sum(
            [
                np.outer(deltaX[i], deltaX[i]) * P_1gt2[i] * P_2gt1[i]
                for i in range(deltaX.shape[0])
            ],
            self.l2_reg * np.identity(w.shape[0]),
        )

        self.cov = np.linalg.inv(self.H)

        print("nn", loss, acc, pdis)
        if validation_data is not None:
            acc, pdis, avg_var, validation_loss = self.eval(validation_data)
        print("bayes", loss, acc, pdis)

        return loss, validation_loss, acc, pdis, avg_var

    def get_conditional_cov(self, x):
        w = self.mu
        x = self.get_features(x)
        y_pred = np.dot(x, w).sum(-1)
        diff = y_pred[0] - y_pred[1]
        P_1gt2 = 1 / (1 + np.exp(-diff))
        P_2gt1 = 1 - P_1gt2
        deltaX = x.sum(1)
        deltaX = deltaX[0, :] - deltaX[1, :]
        H = self.H + np.outer(deltaX, deltaX) * P_1gt2 * P_2gt1
        return np.linalg.inv(H)

    def eval(self, test_data):
        assert test_data.observation_type == "value"
        test_x, test_y = test_data.get_all()

        test_x = self.get_features(test_x).sum(-2)
        y_pred = np.dot(test_x, self.mu)
        y_var = np.matmul(test_x, np.matmul(self.cov, test_x.T))
        avg_var = y_var.mean()

        pcor = np.corrcoef(np.reshape(y_pred, -1), np.reshape(test_y, -1))
        pdis = np.mean(np.sqrt((1 - pcor) / 2))

        n = y_pred.shape[0]
        m = n // 2
        y_pred1 = np.array(y_pred[:m])
        y_pred2 = np.array(y_pred[m : 2 * m])
        test_y1 = np.array(test_y[:m])
        test_y2 = np.array(test_y[m : 2 * m])

        acc = np.mean(
            np.logical_or(
                np.logical_and(y_pred1 > y_pred2, test_y1 > test_y2),
                (test_y1 <= test_y2),
            )
        )

        P_1gt2 = np.exp(y_pred1) / (np.exp(y_pred1) + np.exp(test_y2))
        P_1gt2 = np.clip(P_1gt2, 0.01, 0.99)
        validation_loss = -np.mean(
            np.array(test_y1 >= test_y2, dtype=np.float) * np.log(P_1gt2)
            + np.array(test_y2 > test_y1, dtype=np.float) * np.log(1 - P_1gt2)
        )

        return acc, pdis, avg_var, validation_loss

    def save(self, path):
        assert path.endswith(".zip")
        with ZipFile(path, "w") as zf:
            with zf.open("nn_model.pt", "w") as f:
                self.basis_fct_model.save(f)
            with zf.open("bayes_model.pt", "w") as f:
                torch.save(
                    {
                        "mu": self.mu,
                        "cov": self.cov,
                    },
                    f,
                )

    @classmethod
    def load(cls, path):
        assert path.endswith(".zip")
        with ZipFile(path, "r") as zf:
            with zf.open("nn_model.pt", "r") as f:
                basis_fct_model = RewardModelNN.load(io.BytesIO(f.read()))
            with zf.open("bayes_model.pt", "r") as f:
                bayes_model_dict = torch.load(io.BytesIO(f.read()))
        in_size = basis_fct_model.in_size
        hidden_size = basis_fct_model.hidden_size
        model = cls(in_size, hidden_size)
        model.basis_fct_model = basis_fct_model
        model.mu = bayes_model_dict["mu"]
        model.cov = bayes_model_dict["cov"]
        return model


class RewardModelEnvWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        self.reward_model = reward_model
        self.ensemble = isinstance(reward_model, RewardModelEnsemble)
        if self.ensemble:
            self.running_mean_std = [
                RunningMeanStd() for _ in range(len(self.reward_model.models))
            ]
        else:
            self.running_mean_std = RunningMeanStd()
        self.epsilon = 1e-8
        self.true_return = 0
        super().__init__(env)

    def reset(self):
        self.true_return = 0
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info["true_reward"] = rew
        self.true_return += rew
        info["true_return"] = self.true_return
        x = np.concatenate([obs, action])
        if self.ensemble:
            rewards = self.reward_model.predict_all(x)
            for i in range(len(rewards)):
                self.running_mean_std[i].update(np.array([rewards[i]]))
                rewards[i] = (rewards[i] - self.running_mean_std[i].mean) / np.sqrt(
                    self.running_mean_std[i].var + self.epsilon
                )
            rew = np.mean(rewards)
        else:
            rew = self.reward_model.forward(torch.Tensor(x)).detach().numpy()[0]
            self.running_mean_std.update(np.array([rew]))
            rew = (rew - self.running_mean_std.mean) / np.sqrt(
                self.running_mean_std.var + self.epsilon
            )
        return obs, rew, done, info


class RewardModelSampleEnvWrapper(gym.Wrapper):
    def __init__(self, env, reward_model, theta):
        self.reward_model = reward_model
        assert isinstance(reward_model, RewardModelAdaptiveBayesianLinear)
        self.running_mean_std = RunningMeanStd()
        self.epsilon = 1e-8
        self.theta = theta
        self.true_return = 0
        super().__init__(env)

    def reset(self):
        self.true_return = 0
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info["true_reward"] = rew
        self.true_return += rew
        info["true_return"] = self.true_return
        x = np.concatenate([obs, action])
        f = self.reward_model.get_features(x)
        rew = np.dot(f, self.theta)
        self.running_mean_std.update(np.array([rew]))
        rew = (rew - self.running_mean_std.mean) / np.sqrt(
            self.running_mean_std.var + self.epsilon
        )
        return obs, rew, done, info


class TrueRewardCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(TrueRewardCallback, self).__init__(verbose)

    def _on_training_start(self):
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self):
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self):
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self):
        """
        This event is triggered before updating the policy.
        """
        log_interval = self.locals["log_interval"]
        iteration = self.n_calls
        if log_interval is not None and iteration % log_interval == 0:
            safe_mean = self.globals["safe_mean"]
            self.logger.record(
                "rollout/ep_avg_true_ret_mean",
                safe_mean(
                    [ep_info["true_return"] for ep_info in self.model.ep_info_buffer]
                ),
            )

    def _on_training_end(self):
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def get_label(yi, yj):
    return [
        int(yi > yj) + 0.5 * int(yi == yj),
        int(yj > yi) + 0.5 * int(yi == yj),
    ]


def main():
    T = 40
    N = 100
    comparisons = True
    n_ensemble = 3
    n_epochs = 100

    dummy_data = False
    data_x, data_y = [], []

    if dummy_data:
        # dummy data
        n_obs, n_act = 2, 2
        X = np.random.random((N, T, n_obs + n_act))
        w = np.array([1, 0, 0, 0])
        for i in range(X.shape[0]):
            x = X[i]
            y = np.matmul(x, w).sum()
            data_x.append(x)
            data_y.append(y)
    else:
        from stable_baselines3 import SAC

        policy = SAC.load("sac_pendulum_60k.zip")
        eps = 1  # 0.3

        # sample data
        env = gym.make("InvertedPendulum-Short-v2")
        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.shape[0]
        done = True
        seq_obs, seq_rew = [], []
        i = 1
        for _ in range(N * T):
            if done:
                t = 0
                seq_obs, seq_rew = [], []
                last_obs = env.reset()

            if np.random.random() < eps:
                a = env.action_space.sample()
            else:
                a, _states = policy.predict(last_obs, deterministic=False)

            obs, rew, done, info = env.step(a)

            t += 1
            x = np.concatenate((last_obs, a), 0)
            seq_obs.append(x)
            seq_rew.append(rew)
            last_obs = obs
            if t == T:
                print(i, end=" ", flush=True)
                i += 1
                t = 0
                data_x.append(seq_obs)
                data_y.append(np.sum(seq_rew))
                seq_obs, seq_rew = [], []

    if comparisons:
        data = SampleBuffer("comparison")
        L = len(data_x)
        for i in range(L // 2):
            for j in range(L // 2):
                xi = data_x[i]
                yi = data_y[i]
                xj = data_x[j]
                yj = data_y[j]
                y = get_label(yi, yj)
                data.add_single((xi, xj), y)
    else:
        data = SampleBuffer("value")
        data.add(data_x, data_y)
    print("\ny.mean", np.mean(data.get_all()[1]))

    validation_data = SampleBuffer("value")
    validation_data.add(data_x, data_y)

    reward_model = RewardModelAdaptiveBayesianLinear(n_obs + n_act)

    reward_model.train(
        data,
        batch_size=50,
        n_epochs=n_epochs,
        comparisons=comparisons,
        validation_data=validation_data,
    )

    reward_model.save("pendulum_reward_model.zip")
    reward_model2 = RewardModelAdaptiveBayesianLinear.load("pendulum_reward_model.zip")

    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
