import datetime
import os
import pickle

import gym
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

import active_reward_learning
from active_reward_learning.drlhp.buffer import SampleBuffer
from active_reward_learning.drlhp.reward_model import (
    RewardModelAdaptiveBayesianLinear,
    RewardModelEnsemble,
    RewardModelEnvWrapper,
    RewardModelSampleEnvWrapper,
    get_label,
)
from active_reward_learning.drlhp.util import render_mujoco_from_obs
from active_reward_learning.envs.time_feature_wrapper import TimeFeatureWrapper
from active_reward_learning.util.video import save_video


def harmonic_number(n):
    return sum([1 / i for i in range(1, int(n) + 1)])


def record_video(policy, video_folder, label):
    env = policy.env

    # stable_baselines specific way to get one of the inner envs
    video_length = env.unwrapped.envs[0].unwrapped.spec.max_episode_steps

    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=label,
    )

    obs = env.reset()
    for _ in range(video_length + 1):
        action, _states = policy.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
    # Save the video
    env.close()


def make_video_of_sequence(env, seq, filename):
    ims = []
    for obs in seq:
        rgb = render_mujoco_from_obs(env, obs)
        ims.append(rgb)
    save_video(ims, filename)


def make_video_of_comparison(env_id, seq1, seq2, filename_base):
    env = gym.make(env_id)
    video1 = make_video_of_sequence(env, seq1, f"{filename_base}_seq1.mp4")
    video2 = make_video_of_sequence(env, seq2, f"{filename_base}_seq2.mp4")


class DRLHPCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(
        self,
        _run,
        reward_model,
        n_samples=400,
        n_rollouts=200,
        seq_length=40,
        total_timesteps=100000,
        schedule_update_every=10000,
        update_policies_every_n=1,
        acquisition_function="random",
        comparisons=True,
        video_folder=None,
        policy_folder=None,
        policy_ensemble=None,
        policy_ensemble_timesteps=10000,
        n_epochs_model=100,
        fixed_basis_functions=False,
        use_time_feature_wrapper_for_policy=False,
        exploration_model=None,
        exploration_env=None,
        exploration_sigma=0,
        exploration_eps=0,
        rollout_candidate_policies_for_exploration=False,
        candidate_query_file=None,
        record_videos=False,
        save_models=True,
        verbose=0,
    ):
        super().__init__(verbose)
        self._run = _run
        self.n_samples = n_samples
        self.n_rollouts = n_rollouts
        self.seq_length = seq_length
        self.schedule_update_every = schedule_update_every
        self.update_policies_every_n = update_policies_every_n
        self.acquisition_function = acquisition_function
        self.video_folder = video_folder
        self.policy_folder = policy_folder
        self.policy_ensemble = policy_ensemble
        self.policy_ensemble_timesteps = policy_ensemble_timesteps
        self.fixed_basis_functions = fixed_basis_functions
        self.use_time_feature_wrapper_for_policy = use_time_feature_wrapper_for_policy
        self.record_videos = record_videos
        self.save_models = save_models
        self.rollout_candidate_policies_for_exploration = (
            rollout_candidate_policies_for_exploration
        )
        if acquisition_function == "idrl" and policy_ensemble is None:
            raise Exception("IDRL requires policy ensemble")
        if self.rollout_candidate_policies_for_exploration and policy_ensemble is None:
            raise Exception("Exploration requires policy ensemble")

        self.reward_model = reward_model
        isinstance(self.reward_model, RewardModelAdaptiveBayesianLinear)

        self.model_batch_size = 20
        self.n_epochs_init = n_epochs_model
        self.n_epochs = n_epochs_model
        self.schedule_perc = 0.25
        assert self.model_batch_size < self.n_samples * self.schedule_perc

        self.total_timesteps = total_timesteps

        n = self.total_timesteps // self.schedule_update_every
        N = np.ceil((1 - self.schedule_perc) * self.n_samples)
        Hn = harmonic_number(n)
        self.schedule_c = N / Hn

        self.last_update_at = 0
        self.n_sampling_step = 0
        self.count_until_policy_update = 0
        self.comparisons = comparisons
        observation_type = "comparison" if self.comparisons else "value"
        self.data = SampleBuffer(observation_type)

        self.exploration_model = exploration_model
        self.exploration_env = exploration_env
        self.exploration_sigma = exploration_sigma
        self.exploration_eps = exploration_eps

        if candidate_query_file is not None:
            print(f"Loading candidate queries from {candidate_query_file}...")
            with open(candidate_query_file, "rb") as f:
                queries = pickle.load(f)
            data_x, data_y = [], []
            for query in queries:
                data_x.append(query["seq_obs"])
                data_y.append(np.sum(query["seq_rew"]))
            self.fixed_candidate_queries = True
            idx = np.arange(len(data_x))
            np.random.shuffle(idx)
            self.fixed_candidate_queries_x = []
            self.fixed_candidate_queries_y = []
            for i in idx:
                self.fixed_candidate_queries_x.append(data_x[i])
                self.fixed_candidate_queries_y.append(data_y[i])
        else:
            self.fixed_candidate_queries = False

    def _collect_sequences(self, n_rollouts, episode_length=None):
        if self.rollout_candidate_policies_for_exploration:
            expl_policies = [self.model] + self.policy_ensemble.models
            expl_envs = [model.env for model in expl_policies]
            n_rollouts_per_policy = [
                n_rollouts // len(expl_policies) for _ in range(len(expl_policies))
            ]
            n_rollouts_per_policy[0] += n_rollouts - sum(n_rollouts_per_policy)
            env_wrapped = True
            print("Using current policy and candidate policies for exploration...")
        elif self.exploration_model is not None and self.exploration_env is not None:
            expl_policies = [self.exploration_model]
            expl_envs = [self.exploration_env]
            n_rollouts_per_policy = [n_rollouts]
            env_wrapped = False
            print("Using specified exploration policy...")
        else:
            expl_policies = [self.model]
            expl_envs = [self.model.env]
            n_rollouts_per_policy = [n_rollouts]
            env_wrapped = True
            print("Using current policy for exploration...")

        data_x, data_y = [], []

        for model, env, n_rollouts in zip(
            expl_policies, expl_envs, n_rollouts_per_policy
        ):
            assert isinstance(env, VecNormalize)
            done = True
            seq_obs, seq_rew = [], []
            i = 1
            for _ in range(n_rollouts * self.seq_length):
                if done or (episode_length is not None and t >= episode_length):
                    t = 0
                    seq_obs, seq_rew = [], []
                    last_obs_norm = env.reset()
                    last_obs = env.get_original_obs()

                if (
                    self.exploration_eps <= 0
                    or np.random.random() > self.exploration_eps
                ):
                    a, _ = model.predict(last_obs_norm, deterministic=False)
                    if self.exploration_sigma > 0:
                        a += np.random.normal(
                            loc=0, scale=self.exploration_sigma, size=a.shape
                        )
                else:
                    a = np.array([env.action_space.sample()])

                obs_norm, model_rew, done, info = env.step(a)
                obs = env.get_original_obs()

                if env_wrapped:
                    # env is wrapped in reward model, but we want to label with true reward
                    rew = info[0]["true_reward"]
                else:
                    # env is not wrapped in reward model wrapper
                    rew = model_rew

                t += 1
                # select [0] because env is a VecEnv
                obs_to_log = last_obs[0]
                if self.use_time_feature_wrapper_for_policy:
                    obs_to_log = obs_to_log[:-1]
                seq_obs.append(np.concatenate((obs_to_log, a[0]), 0))
                seq_rew.append(rew)
                last_obs_norm = obs_norm
                last_obs = obs

                if t == self.seq_length:
                    print(i, end=" ", flush=True)
                    i += 1
                    data_x.append(seq_obs)
                    data_y.append(np.sum(seq_rew))
                    t = 0
                    seq_obs, seq_rew = [], []
            print()
        return data_x, data_y

    def _label_clips(self, x, y, n_samples, acquisition_function, plot_path=None):
        if self.fixed_candidate_queries:
            x_full, y_full = x, y
            x, y = [], []
            idx = np.random.choice(len(x_full), self.n_rollouts)
            for i in idx:
                x.append(x_full[i])
                y.append(y_full[i])
            print(len(x))

        if acquisition_function == "random":
            if self.comparisons:
                idx = np.random.choice(len(x), 2 * n_samples)
                idx1 = idx[:n_samples]
                idx2 = idx[n_samples:]
            else:
                idx = np.random.choice(len(x), n_samples)
        elif acquisition_function == "idrl":
            assert self.comparisons

            ensemble_policy_features = self.policy_ensemble.get_policy_features()
            n_policies = len(self.policy_ensemble.models)
            i_max, j_max = None, None
            value = -float("inf")
            for i in range(n_policies):
                for j in range(n_policies):
                    if i != j:
                        fi = ensemble_policy_features[i]
                        fj = ensemble_policy_features[j]
                        cov = self.reward_model.cov
                        var = np.dot(fi - fj, np.dot(cov, fi - fj))
                        if var > value:
                            value = var
                            i_max, j_max = i, j
            print(f"Selected policies {i_max} and {j_max}...")
            fi = ensemble_policy_features[i_max]
            fj = ensemble_policy_features[j_max]
            feat = fi - fj

            if self.comparisons:
                indices = []
                variances = []
                for i in range(len(x)):
                    for j in range(i + 1, len(x)):
                        indices.append((i, j))
                        x_ij = np.stack([x[i], x[j]])
                        cond_cov = self.reward_model.get_conditional_cov(x_ij)
                        var = np.dot(feat, np.dot(cond_cov, feat))
                        variances.append(var)

                i_sort = np.argsort(variances)
                i_sort = i_sort[:n_samples]
                idx1, idx2 = zip(*[indices[i] for i in i_sort])
                idx1 = np.array(idx1)
                idx2 = np.array(idx2)
            else:
                x_feat = self.reward_model.get_features(x)
                x_feat = x_feat.sum(1)
                variances = [
                    np.dot(x_feat[i], np.dot(self.reward_model.cov, x_feat[i]))
                    for i in range(x.shape[0])
                ]
                i_sort = np.argsort(variances)
                idx = np.array(i_sort[:n_samples])
        elif acquisition_function == "variance":
            x_feat = self.reward_model.get_features(x)
            x_feat = x_feat.sum(1)

            if self.comparisons:
                indices = []
                variances = []
                for i in range(len(x)):
                    for j in range(i + 1, len(x)):
                        indices.append((i, j))
                        feat = x_feat[i] - x_feat[j]
                        var = np.dot(feat, np.dot(self.reward_model.cov, feat))
                        variances.append(var)

                i_sort = np.argsort(variances)[::-1]
                i_sort = i_sort[:n_samples]
                idx1, idx2 = zip(*[indices[i] for i in i_sort])
                idx1 = np.array(idx1)
                idx2 = np.array(idx2)
            else:
                variances = [
                    np.dot(x_feat[i], np.dot(self.reward_model.cov, x_feat[i]))
                    for i in range(x.shape[0])
                ]
                i_sort = np.argsort(variances)[::-1]
                idx = np.array(i_sort[:n_samples])
        else:
            raise NotImplementedError(f"{acquisition_function} not implemented")

        if self.comparisons:
            for i, j in zip(idx1, idx2):
                sample_x1 = x[i]
                sample_x2 = x[j]
                sample_y1 = y[i]
                sample_y2 = y[j]
                self.data.add_single(
                    (sample_x1, sample_x2), get_label(sample_y1, sample_y2)
                )
            if plot_path is not None:
                env_id = self.model.env.unwrapped.envs[0].unwrapped.spec.id
                make_video_of_comparison(env_id, x[idx1[0]], x[idx2[0]], plot_path)
        else:
            for i in idx:
                sample_x = x[i]
                sample_y = y[i]
                self.data.add_single(sample_x, sample_y)

    def _on_training_start(self):
        """
        This method is called before the first rollout starts.
        """
        self.last_update_at = 0
        self.n_sampling_step = 0
        print("Collecting initial rollouts...")
        if self.fixed_candidate_queries:
            x, y = self.fixed_candidate_queries_x, self.fixed_candidate_queries_y
        else:
            x, y = self._collect_sequences(self.n_rollouts)
        n_samples_init = int(np.floor(self.n_samples * self.schedule_perc))
        print(f"Getting {n_samples_init} samples...")
        self._label_clips(x, y, n_samples_init, "random")
        self.validation_data = SampleBuffer("value")
        self.validation_data.add(x, y)
        print("Training initial model...")
        train_loss, validation_loss, acc, pdis, avg_var = self.reward_model.train(
            self.data,
            batch_size=self.model_batch_size,
            n_epochs=self.n_epochs_init,
            comparisons=self.comparisons,
            validation_data=self.validation_data,
            train_basis_functions=True,
        )
        self.logger.record("reward_model/training_loss", train_loss)
        self.logger.record("reward_model/validation_loss", validation_loss)
        self.logger.record("reward_model/acc", acc)
        self.logger.record("reward_model/pdis", pdis)
        self.logger.record("reward_model/avg_var", avg_var)
        self.logger.record("reward_model/cov_trace", np.trace(self.reward_model.cov))
        self.logger.record("reward_model/n_samples", len(self.data))
        self._run.info["tb_log"] = self.logger.get_dir()

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

        num_timesteps = self.model.num_timesteps
        if num_timesteps - self.last_update_at >= self.schedule_update_every:
            self.n_sampling_step += 1

            n_samples = int(np.ceil(self.schedule_c / self.n_sampling_step))
            n_samples = min(n_samples, self.n_samples - len(self.data) - n_samples)

            if n_samples > 0:
                print(f"Updating reward model after {num_timesteps} timesteps...")
                print(f"n_samples: {n_samples}")
                if self.fixed_candidate_queries:
                    x, y = (
                        self.fixed_candidate_queries_x,
                        self.fixed_candidate_queries_y,
                    )
                else:
                    print(f"Collecting {self.n_rollouts} rollouts...")
                    x, y = self._collect_sequences(self.n_rollouts)

                if self.video_folder is not None and self.record_videos:
                    filename_base = "comparison_{}_timesteps_{}_samples".format(
                        num_timesteps, len(self.data)
                    )
                    plot_path = os.path.join(self.video_folder, filename_base)
                else:
                    plot_path = None

                print(f"Labeling {n_samples} samples...")
                self._label_clips(
                    x, y, n_samples, self.acquisition_function, plot_path=plot_path
                )

                print("Updating model...")
                (
                    training_loss,
                    validation_loss,
                    acc,
                    pdis,
                    avg_var,
                ) = self.reward_model.train(
                    self.data,
                    batch_size=self.model_batch_size,
                    n_epochs=self.n_epochs,
                    comparisons=self.comparisons,
                    validation_data=self.validation_data,
                    train_basis_functions=(not self.fixed_basis_functions),
                )

                self.logger.record("reward_model/training_loss", training_loss)
                self.logger.record("reward_model/validation_loss", validation_loss)
                self.logger.record("reward_model/acc", acc)
                self.logger.record("reward_model/pdis", pdis)
                self.logger.record("reward_model/avg_var", avg_var)
                self.logger.record(
                    "reward_model/cov_trace", np.trace(self.reward_model.cov)
                )
                self.logger.record("reward_model/n_samples", len(self.data))

                if self.count_until_policy_update % self.update_policies_every_n == 0:
                    print("Updating policies...")

                    if self.record_videos or self.save_models:
                        print("Recording video of current policy...")
                        label = "current_policy_{}_timesteps_{}_samples".format(
                            num_timesteps, len(self.data)
                        )
                        if self.record_videos:
                            record_video(
                                self.model,
                                self.video_folder,
                                label,
                            )
                        if self.save_models:
                            self.model.save(
                                os.path.join(self.policy_folder, label + ".zip")
                            )
                            self.model.env.save(
                                os.path.join(self.policy_folder, label + "_venv.zip")
                            )
                        if self.policy_ensemble is not None:
                            print("Recording videos of ensemble policies...")
                            for i, policy in enumerate(self.policy_ensemble.models):
                                label = "ensemble_policies_{}_{}_timesteps_{}_samples".format(
                                    i, num_timesteps, len(self.data)
                                )
                                if self.record_videos:
                                    record_video(
                                        policy,
                                        self.video_folder,
                                        label,
                                    )
                                if self.save_models:
                                    self.model.save(
                                        os.path.join(self.policy_folder, label + ".zip")
                                    )
                                    self.model.env.save(
                                        os.path.join(
                                            self.policy_folder, label + "_venv.zip"
                                        )
                                    )

                    if self.policy_ensemble is not None:
                        self.policy_ensemble.learn(
                            total_timesteps=self.policy_ensemble_timesteps,
                            log_interval=None,
                        )
                        for model_i, model in enumerate(self.policy_ensemble.models):
                            if (
                                len(model.ep_info_buffer) > 0
                                and len(model.ep_info_buffer[0]) > 0
                            ):
                                self.logger.record(
                                    f"candidate_policies/policy_{model_i}_ep_true_ret_mean",
                                    safe_mean(
                                        [
                                            ep_info["true_return"]
                                            for ep_info in model.ep_info_buffer
                                        ]
                                    ),
                                )
                            else:
                                print(
                                    "Warning: Problem recording candidate policy returns"
                                )

                self.count_until_policy_update += 1
                self.last_update_at = num_timesteps

    def _on_training_end(self):
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class PosteriorSamplePolicyEnsemble:
    def __init__(
        self,
        reward_model,
        model_class,
        policy,
        env_id,
        n_samples,
        use_time_feature_wrapper=False,
        reinitialize_when_updating=False,
        **kwargs,
    ):
        assert isinstance(reward_model, RewardModelAdaptiveBayesianLinear)
        self.reward_model = reward_model
        self.use_time_feature_wrapper = use_time_feature_wrapper
        self.reinitialize_when_updating = reinitialize_when_updating
        self.models = []
        self.sample_envs = []
        for _ in range(n_samples):
            env = RewardModelSampleEnvWrapper(
                gym.make(env_id), self.reward_model, self.reward_model.mu
            )
            if use_time_feature_wrapper:
                env = TimeFeatureWrapper(env)
            self.sample_envs.append(env)
            env = Monitor(env, info_keywords=("true_return",))
            venv = DummyVecEnv([lambda: env])
            venv = VecNormalize(venv)
            self.model_class = model_class
            self.policy = policy
            self.model_kwargs = kwargs
            model = model_class(policy, venv, **kwargs)
            self.models.append(model)

    def learn(self, **kwargs):
        mu, cov = self.reward_model.mu, self.reward_model.cov
        for env in self.sample_envs:
            env.theta = np.random.multivariate_normal(mu, cov)
        for i in range(len(self.models)):
            if self.reinitialize_when_updating:
                print(f"Reinitializing model {i}...")
                venv = self.models[i].env
                self.models[i] = self.model_class(
                    self.policy, venv, **self.model_kwargs
                )
            model = self.models[i]
            model.learn(**kwargs)

    def get_policy_features(self):
        policy_features = []
        n_rollouts = 10
        for policy in self.models:
            env = policy.env
            assert isinstance(env, VecNormalize)
            features = []
            for _ in range(n_rollouts):
                obs_norm = env.reset()
                obs = env.get_original_obs()
                done = False
                while not done:
                    action, _states = policy.predict(obs_norm, deterministic=True)
                    # select [0] because env is a DummyVecEnv
                    if self.use_time_feature_wrapper:
                        x = np.concatenate([obs[0][:-1], action[0]])
                    else:
                        x = np.concatenate([obs[0], action[0]])
                    f = self.reward_model.get_features(x)
                    obs_norm, true_reward, done, _ = env.step(action)
                    obs = env.get_original_obs()
                    features.append(f)
            policy_features.append(np.sum(features, axis=0) / n_rollouts)
        policy_features = np.array(policy_features)
        return policy_features


def get_timestep_seed_label():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    seed = np.random.randint(100000, 999999)
    np.random.seed()  # reset seed to not get conflicts between runs with the same
    rand = np.random.randint(100000, 999999)
    np.random.seed(seed)
    label = f"{timestamp}_{rand}"
    return label


# changes the run _id and thereby the path that the FileStorageObserver
# writes the results
# cf. https://github.com/IDSIA/sacred/issues/174
class SetID(RunObserver):
    priority = 50  # very high priority to set id

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        custom_id = get_timestep_seed_label()
        return custom_id  # started_event returns the _run._id


ex = Experiment("drlhp")
ex.observers = [SetID(), FileStorageObserver("results/drlhp")]


@ex.named_config
def pendulum_debug():
    env_id = "InvertedPendulum-Penalty-Long-v2"
    n_samples = 200
    n_rollouts = 100
    seq_length = 40
    n_model_updates = 10
    n_policy_updates = 10
    total_timesteps = 1e4
    n_ensemble = 3
    comparisons = True
    n_epochs_model = 100
    fixed_basis_functions = False
    acquisition_function = "variance"
    rollout_candidate_policies_for_exploration = True
    reinitialize_candidate_policies = False


@ex.named_config
def pendulum_short():
    env_id = "InvertedPendulum-Penalty-Short-v2"


@ex.named_config
def double_pendulum_short():
    env_id = "InvertedDoublePendulum-Penalty-Short-v2"


@ex.named_config
def pendulum_long():
    env_id = "InvertedPendulum-Penalty-Long-v2"


@ex.named_config
def double_pendulum_long():
    env_id = "InvertedDoublePendulum-Penalty-Long-v2"


@ex.named_config
def cheetah_short_v3():
    env_id = "HalfCheetah-Short-v3"
    use_time_feature_wrapper_for_policy = True


@ex.named_config
def hopper_short_v3():
    env_id = "Hopper-Short-v3"
    use_time_feature_wrapper_for_policy = True


@ex.named_config
def walker_short_v3():
    env_id = "Walker2d-Short-v3"
    use_time_feature_wrapper_for_policy = True


@ex.named_config
def cheetah_long_v3():
    env_id = "HalfCheetah-Long-v3"
    use_time_feature_wrapper_for_policy = True


@ex.named_config
def swimmer_long_v3():
    env_id = "Swimmer-Long-v3"
    use_time_feature_wrapper_for_policy = True


@ex.named_config
def reacher():
    env_id = "Reacher-v2"
    use_time_feature_wrapper_for_policy = True


@ex.named_config
def cheetah_long_withpos_v3():
    env_id = "HalfCheetah-Long-WithPos-v3"
    use_time_feature_wrapper_for_policy = True


@ex.named_config
def swimmer_long_withpos_v3():
    env_id = "Swimmer-Long-WithPos-v3"
    use_time_feature_wrapper_for_policy = True


@ex.named_config
def hopper_long_v3():
    env_id = "Hopper-Long-v3"
    use_time_feature_wrapper_for_policy = True


@ex.named_config
def walker_long_v3():
    env_id = "Walker2d-Long-v3"
    use_time_feature_wrapper_for_policy = True


@ex.named_config
def ant_long_v3():
    env_id = "Ant-Long-v3"
    use_time_feature_wrapper_for_policy = True


@ex.named_config
def base():
    # fixed hyperparameters
    comparisons = True
    n_epochs_model = 10
    fixed_basis_functions = False
    use_time_feature_wrapper_for_policy = False
    acquisition_function = "random"
    total_timesteps = 1e7
    n_rollouts = 200
    reinitialize_candidate_policies = False

    # fixed exploration model
    exploration_model_path = None
    exploration_env_path = None
    exploration_sigma = 0
    exploration_eps = 0

    # hyperparameters taken from DRLHP
    n_samples = 1400
    seq_length = 40

    # tunable hyperparameters
    # for all methods
    n_model_updates = 1000
    # only for IDRL
    n_ensemble = 3
    n_policy_updates = 100


@ex.config
def cfg():
    env_id = None
    n_samples = None
    n_rollouts = None
    seq_length = None
    n_model_updates = None
    n_policy_updates = 1
    total_timesteps = None
    n_ensemble = None
    comparisons = True
    n_epochs_model = None
    acquisition_function = None
    fixed_basis_functions = False
    use_time_feature_wrapper_for_policy = False
    exploration_model_path = None
    exploration_env_path = None
    exploration_sigma = 0
    rollout_candidate_policies_for_exploration = False
    candidate_query_file = None
    reinitialize_candidate_policies = False
    record_videos = False
    save_models = True


@ex.automain
def main(
    _run,
    seed,
    env_id,
    n_samples,
    n_rollouts,
    seq_length,
    n_model_updates,
    n_policy_updates,
    total_timesteps,
    n_ensemble,
    acquisition_function,
    comparisons,
    n_epochs_model,
    fixed_basis_functions,
    use_time_feature_wrapper_for_policy,
    exploration_model_path,
    exploration_env_path,
    exploration_sigma,
    exploration_eps,
    rollout_candidate_policies_for_exploration,
    candidate_query_file,
    reinitialize_candidate_policies,
    record_videos,
    save_models,
):
    assert n_policy_updates <= n_model_updates
    schedule_update_every = total_timesteps // n_model_updates
    policy_ensemble_timesteps = total_timesteps // n_policy_updates
    update_policies_every_n = n_model_updates // n_policy_updates
    print("Update schedule:")
    print("schedule_update_every", schedule_update_every)

    env = gym.make(env_id)
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.shape[0]
    reward_model = RewardModelAdaptiveBayesianLinear(n_obs + n_act)
    if acquisition_function == "idrl" or rollout_candidate_policies_for_exploration:
        print("policy_ensemble_timesteps", policy_ensemble_timesteps)
        print("update_policies_every_n", update_policies_every_n)
        policy_ensemble = PosteriorSamplePolicyEnsemble(
            reward_model,
            SAC,
            "MlpPolicy",
            env_id,
            use_time_feature_wrapper=use_time_feature_wrapper_for_policy,
            n_samples=n_ensemble,
            reinitialize_when_updating=reinitialize_candidate_policies,
            verbose=1,
            # tensorboard_log="./tb_drlhp/candidates",
        )
    else:
        policy_ensemble = None

    if exploration_model_path is not None and exploration_env_path is not None:
        exploration_model = SAC.load(exploration_model_path)
        exploration_env = gym.make(env_id)
        if use_time_feature_wrapper_for_policy:
            exploration_env = TimeFeatureWrapper(exploration_env)
        exploration_env = DummyVecEnv([lambda: Monitor(exploration_env)])
        exploration_env = VecNormalize.load(exploration_env_path, exploration_env)
    else:
        exploration_model = None
        exploration_env = None

    timestepseed = get_timestep_seed_label()
    video_folder = (
        f"videos/{env_id}_{acquisition_function}_{comparisons}_{timestepseed}"
    )
    policy_folder = (
        f"policies/{env_id}_{acquisition_function}_{comparisons}_{timestepseed}"
    )
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(policy_folder, exist_ok=True)
    callback = DRLHPCallback(
        _run,
        reward_model,
        n_samples=n_samples,
        n_rollouts=n_rollouts,
        seq_length=seq_length,
        total_timesteps=total_timesteps,
        schedule_update_every=schedule_update_every,
        update_policies_every_n=update_policies_every_n,
        acquisition_function=acquisition_function,
        comparisons=comparisons,
        video_folder=video_folder,
        policy_folder=policy_folder,
        policy_ensemble=policy_ensemble,
        policy_ensemble_timesteps=policy_ensemble_timesteps,
        n_epochs_model=n_epochs_model,
        fixed_basis_functions=fixed_basis_functions,
        use_time_feature_wrapper_for_policy=use_time_feature_wrapper_for_policy,
        exploration_model=exploration_model,
        exploration_env=exploration_env,
        exploration_sigma=exploration_sigma,
        exploration_eps=exploration_eps,
        rollout_candidate_policies_for_exploration=rollout_candidate_policies_for_exploration,
        candidate_query_file=candidate_query_file,
        record_videos=record_videos,
        save_models=save_models,
    )

    env = RewardModelEnvWrapper(env, reward_model)
    if use_time_feature_wrapper_for_policy:
        env = TimeFeatureWrapper(env)
    env = Monitor(env, info_keywords=("true_return",))

    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize(venv)

    model = SAC("MlpPolicy", venv, verbose=1, tensorboard_log="./tb_drlhp/")
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        callback=callback,
        tb_log_name=f"tb_log_{env_id}_{acquisition_function}_{comparisons}_{timestepseed}",
    )
    model.save(f"drhlp_sac_{env_id}.zip")
    venv.save(f"drhlp_venv_{env_id}.zip")
    reward_model.save(f"drhlp_model_{env_id}.zip")
