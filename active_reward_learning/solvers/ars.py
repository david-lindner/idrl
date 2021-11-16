"""
Implementation of Augmented Random Search with linear policies as described in [1].

The code is based on [2].

[1] Mania, Horia, Aurelia Guy, and Benjamin Recht. "Simple random search of static
    linear policies is competitive for reinforcement learning."
    Advances in Neural Information Processing Systems. 2018.
[2] https://github.com/alexis-jacq/numpy_ARS
"""

import multiprocessing
import time
from multiprocessing import Pool, Process, Queue
from typing import Callable, Dict, Optional

import gym
import numpy as np
from gym import wrappers

from active_reward_learning.common.policy import LinearPolicy

from .base import BaseSolver

# for being able to use pytorch models we need to set the start methods
# cf. https://stackoverflow.com/questions/48822463/how-to-use-pytorch-multiprocessing
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass


class Normalizer:
    def __init__(self, num_inputs, only_mean=False):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.fixed_mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.std = np.ones(num_inputs)
        self.fixed_std = np.ones(num_inputs)
        self.only_mean = only_mean

    def add(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        if not self.only_mean:
            self.mean_diff += (x - last_mean) * (x - self.mean)
            var = self.mean_diff / self.n
            var[var < 1e-8] = np.inf
            self.std = np.sqrt(var)

    def add_set(self, other_normalizer):
        other_var = np.square(other_normalizer.std)
        other_mean_diff = other_var * other_normalizer.n
        delta = other_normalizer.mean - self.mean
        total_n = self.n + other_normalizer.n
        self.mean += delta * other_normalizer.n / total_n
        self.n = total_n
        if not self.only_mean:
            self.mean_diff += (
                other_mean_diff
                + np.square(delta) * self.n * other_normalizer.n / total_n
            )
            var = self.mean_diff / self.n
            var[var < 1e-8] = np.inf
            self.std = np.sqrt(var)

    def update(self):
        self.fixed_mean = self.mean
        self.fixed_std = self.std

    def normalize(self, inputs):
        return (inputs - self.fixed_mean) / self.fixed_std


# linear policy
class Policy:
    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))
        # self.theta = np.array([[-1, 0, 0, -0.1], [0, -1, 0.1, 0]], dtype=np.float64)

    def evaluate(self, input):
        return self.theta.dot(input)

    def perturbation(self, input, delta, noise):
        return (self.theta + noise * delta).dot(input)

    def sample_deltas(self, n_directions):
        # return np.random.randn(n_directions, *self.theta.shape)
        return np.array(
            [np.random.randn(*self.theta.shape) for _ in range(n_directions)]
        )

    def update(self, rollouts, sigma_r, step_size, b):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        self.theta += step_size * step / (sigma_r * b)


def get_rollout_parallel(env, normalize, horizon, ob_dim, noise, queue_in, queue_out):
    for data_in in iter(queue_in.get, "STOP"):
        obs_normalizer, rew_normalizer, policy, delta = data_in
        state = env.reset()
        done = False
        timestep = 0.0
        total_reward = 0
        if normalize:
            rollout_obs_normalizer = Normalizer(ob_dim)
            rollout_rew_normalizer = Normalizer(1, only_mean=True)
        else:
            rollout_obs_normalizer = None
            rollout_rew_normalizer = None
        while not done and timestep < horizon:
            if normalize:
                rollout_obs_normalizer.add(state)
                state = obs_normalizer.normalize(state)
            action = policy.perturbation(state, delta, noise)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            state, reward, done, _ = env.step(action)

            if normalize:
                rollout_rew_normalizer.add(reward)
                reward = rew_normalizer.normalize(reward)
            # reward = max(min(reward, 1), -1)
            total_reward += reward
            timestep += 1
        data_out = (total_reward, rollout_obs_normalizer, rollout_rew_normalizer)
        queue_out.put(data_out)
    return


class AugmentedRandomSearchSolver(BaseSolver):
    def __init__(
        self,
        env,
        horizon=1000,
        step_size=0.02,
        n_directions=8,
        b=4,
        noise=0.01,
        n_jobs=1,
        normalize=True,
        policy_out_file="ars_policy.npy",
    ):
        print(locals())
        self.horizon = horizon
        self.step_size = step_size
        self.n_directions = n_directions
        self.b = b
        assert self.b <= self.n_directions, "b must be <= n_directions"
        self.noise = noise
        self.n_jobs = n_jobs
        self.normalize = normalize

        super().__init__(env)
        self.set_env(env)
        self.policy = LinearPolicy(self.w, None, None, env)
        self.policy_evaluation = -float("inf")

        self.policy_out_file = policy_out_file

        if self.n_jobs > 1:
            self.queue_in = Queue()
            self.queue_out = Queue()
            self.processes = [
                Process(
                    target=get_rollout_parallel,
                    args=(
                        env,
                        self.normalize,
                        self.horizon,
                        self.ob_dim,
                        self.noise,
                        self.queue_in,
                        self.queue_out,
                    ),
                )
                for _ in range(self.n_jobs)
            ]
            for process in self.processes:
                process.start()

    def set_env(self, env):
        assert isinstance(env, gym.Env)
        assert isinstance(env.action_space, gym.spaces.Box)
        assert len(env.action_space.shape) == 1
        assert len(env.observation_space.shape) == 1
        self.env = env
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.w = np.zeros((self.ac_dim, self.ob_dim))
        self.new_w = np.zeros((self.ac_dim, self.ob_dim))
        self.mean = np.zeros(self.ob_dim)
        self.new_mean = np.zeros(self.ob_dim)
        self.std = np.ones(self.ob_dim)
        self.new_std = np.ones(self.ob_dim)

    def _get_rollout(self, obs_normalizer, rew_normalizer, policy, env, delta):
        state = env.reset()
        done = False
        timestep = 0.0
        total_reward = 0
        if self.normalize:
            rollout_obs_normalizer = Normalizer(self.ob_dim)
            rollout_rew_normalizer = Normalizer(1, only_mean=True)
        else:
            rollout_obs_normalizer = None
            rollout_rew_normalizer = None
        while not done and timestep < self.horizon:
            if self.normalize:
                rollout_obs_normalizer.add(state)
                state = obs_normalizer.normalize(state)
            action = policy.perturbation(state, delta, self.noise)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            state, reward, done, _ = env.step(action)

            if self.normalize:
                rollout_rew_normalizer.add(reward)
                reward = rew_normalizer.normalize(reward)
            # reward = max(min(reward, 1), -1)
            total_reward += reward
            timestep += 1
        return total_reward, rollout_obs_normalizer, rollout_rew_normalizer

    def _get_perturbations(self, obs_normalizer, rew_normalizer, policy, deltas):
        deltas_pos_neg = np.concatenate([deltas, -deltas])
        rewards = np.zeros(2 * self.n_directions)
        if self.n_jobs > 1:
            # with Pool(self.n_jobs) as p:
            #     f = functools.partial(
            #         self._get_rollout_parallel,
            #         obs_normalizer,
            #         rew_normalizer,
            #         policy,
            #         self.env,
            #     )
            #     results = p.map(f, deltas_pos_neg)
            #     rewards, rollout_obs_normalizers, rollout_rew_normalizers = zip(
            #         *results
            #     )
            for delta in deltas_pos_neg:
                data_in = obs_normalizer, rew_normalizer, policy, delta
                self.queue_in.put(data_in)
            rewards, rollout_obs_normalizers, rollout_rew_normalizers = [], [], []
            for delta in deltas_pos_neg:
                data_out = self.queue_out.get()
                reward, rollout_obs_normalizer, rollout_rew_normalizer = data_out
                rewards.append(reward)
                rollout_obs_normalizers.append(rollout_obs_normalizer)
                rollout_rew_normalizers.append(rollout_rew_normalizer)
        else:
            rollout_obs_normalizers = []
            rollout_rew_normalizers = []
            for k in range(2 * self.n_directions):
                (
                    rewards[k],
                    rollout_obs_normalizer,
                    rollout_rew_normalizer,
                ) = self._get_rollout(
                    obs_normalizer, rew_normalizer, policy, self.env, deltas_pos_neg[k]
                )
                rollout_obs_normalizers.append(rollout_obs_normalizer)
                rollout_rew_normalizers.append(rollout_rew_normalizer)

        if self.normalize:
            for rollout_obs_normalizer in rollout_obs_normalizers:
                obs_normalizer.add_set(rollout_obs_normalizer)
            for rollout_rew_normalizer in rollout_rew_normalizers:
                rew_normalizer.add_set(rollout_rew_normalizer)

        # print()
        # print("normalizer", normalizer.n, normalizer.mean, normalizer.std)
        # print()

        rewards_positive = np.array(rewards[: self.n_directions])
        rewards_negative = np.array(rewards[self.n_directions :])
        return rewards_positive, rewards_negative, np.std(rewards)

    def solve(
        self,
        n_episodes: int = 100,
        logging_callback: Optional[Callable[[Dict, Dict], bool]] = None,
        init_w=None,
    ) -> LinearPolicy:
        self.policy = LinearPolicy(self.w, None, None, self.env)
        self.policy_evaluation = -float("inf")
        self.w = np.zeros((self.ac_dim, self.ob_dim))
        self.new_w = np.zeros((self.ac_dim, self.ob_dim))

        if self.normalize:
            obs_normalizer = Normalizer(self.ob_dim)  # type: ignore
            rew_normalizer = Normalizer(1, only_mean=True)  # type: ignore
        else:
            obs_normalizer = None  # type: ignore
            rew_normalizer = None  # type: ignore

        policy = Policy(self.ob_dim, self.ac_dim)

        if init_w is not None:
            self.w = init_w
            self.new_w = init_w
            policy.theta = init_w

        for i in range(n_episodes):
            t = time.time()
            # init deltas and rewards
            deltas = policy.sample_deltas(self.n_directions)
            reward_positive, reward_negative, sigma_r = self._get_perturbations(
                obs_normalizer, rew_normalizer, policy, deltas
            )
            if self.normalize:
                print("rew_normalizer.fixed_mean", rew_normalizer.fixed_mean)
                print("rew_normalizer.fixed_std", rew_normalizer.fixed_std)
            if np.all(reward_positive - reward_negative == 0):
                print("Warning: ARS did not collect different rewards.")
            else:
                # sort rollouts wrt max(r_pos, r_neg) and take (self.b) best
                if self.b < self.n_directions:
                    scores = {
                        k: max(r_pos, r_neg)
                        for k, (r_pos, r_neg) in enumerate(
                            zip(reward_positive, reward_negative)
                        )
                    }
                    order = sorted(scores.keys(), key=lambda x: scores[x])[-self.b :]
                    rollouts = [
                        (reward_positive[k], reward_negative[k], deltas[k])
                        for k in order[::-1]
                    ]
                else:
                    rollouts = [
                        (reward_positive[k], reward_negative[k], deltas[k])
                        for k in range(self.n_directions)
                    ]

                # update policy:
                policy.update(rollouts, sigma_r, self.step_size, self.b)

                self.new_w = np.copy(policy.theta)

                if self.normalize:
                    self.new_mean = np.copy(obs_normalizer.fixed_mean)
                    self.new_std = np.copy(obs_normalizer.fixed_std)

                updated = self.update_policy()

                if self.normalize:
                    obs_normalizer.update()
                    rew_normalizer.update()

                delta_t = time.time() - t
                print("\t\tARS Iteration {} time {}s".format(i, delta_t))

                if logging_callback is not None:
                    logging_callback(locals(), globals())

                if updated and self.policy_out_file is not None:
                    self.policy.save(self.policy_out_file)

        return self.policy

    def update_policy(self):
        new_policy = LinearPolicy(self.new_w, self.new_mean, self.new_std, self.env)
        new_policy_evaluation = new_policy.evaluate(self.env, N=10)
        if new_policy_evaluation > self.policy_evaluation:
            print("updated")
            self.w = self.new_w
            self.mean = self.new_mean
            self.std = self.new_std
            self.policy = new_policy
            self.policy_evaluation = new_policy_evaluation
            return True
        return False
