import datetime
import os
import pickle
from abc import ABC, abstractmethod

import gym
import numpy as np


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, obs, deterministic=True):
        raise NotImplementedError()

    def evaluate(self, env, N=10, rollout=True):
        if not rollout:
            print("Warning: Rolling out policy despite rollout=False")
        res = 0
        for _ in range(N):
            obs = env.reset()
            done = False
            while not done:
                a = self.get_action(obs)
                obs, reward, done, _ = env.step(a)
                res += reward
        return res / N


class EpsGreedyPolicy(BasePolicy):
    def __init__(self, greedy_policy: BasePolicy, eps: float, action_space: gym.Space):
        self.greedy = greedy_policy
        self.eps = eps
        self.action_space = action_space

    def get_action(self, obs, deterministic=False):
        if deterministic or np.random.random() > self.eps:
            return self.greedy.get_action(obs, deterministic=True)
        else:
            return self.action_space.sample()


class TabularPolicy(BasePolicy):
    def __init__(self, policy: np.ndarray):
        self.matrix = np.copy(policy)

    def get_action(self, state, deterministic=True):
        if deterministic:
            return np.argmax(self.matrix[state, :])
        else:
            return np.random.choice(
                range(self.matrix.shape[1]), p=self.matrix[state, :]
            )

    def evaluate(self, env, N=1, rollout=False):
        assert env.observation_type == "state"
        if rollout:
            return super().evaluate(env, N)
        else:
            return env.evaluate_policy(self)

    def __eq__(self, other):
        return np.all(self.matrix == other.matrix)


class FixedPolicy(BasePolicy):
    def __init__(self, policy: np.ndarray):
        self.matrix = np.copy(policy)

    def get_action(self, state, deterministic=True):
        t = int(state[-1])
        return self.matrix[t]

    def __eq__(self, other):
        return np.all(self.matrix == other.matrix)


class LinearPolicy(BasePolicy):
    def __init__(self, w, obs_mean=None, obs_std=None, env=None):
        self.w = w
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        if env is not None:
            self.alow = env.action_space.low
            self.ahigh = env.action_space.high
        else:
            self.alow = -np.inf
            self.ahigh = np.inf

    def normalize(self, obs):
        if self.obs_mean is not None and self.obs_std is not None:
            return (obs - self.obs_mean) / self.obs_std
        else:
            return obs

    def get_action(self, obs, deterministic=True):
        obs = self.normalize(obs)
        a = np.dot(self.w, obs)
        a = np.clip(a, self.alow, self.ahigh)
        return a

    def save(self, path):
        policy_dict = {
            "w": list(self.w),
            "mean": list(self.obs_mean),
            "std": list(self.obs_std),
        }
        with open(path, "wb") as f:
            pickle.dump(policy_dict, f)

    @classmethod
    def load(cls, path, env=None):
        with open(path, "rb") as f:
            policy_dict = pickle.load(f)
        policy = cls(
            policy_dict["w"],
            obs_mean=policy_dict["mean"],
            obs_std=policy_dict["std"],
            env=env,
        )
        return policy


class StableBaselinesPolicy(BasePolicy):
    def __init__(self, model):
        # save and load the model as a workaround for creating a copy of the policy
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"tmp_model_{timestamp}.zip"
        model.save(filename)
        self.model = model.__class__.load(filename)
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

    def get_action(self, obs, deterministic=True):
        a, _ = self.model.predict(obs, deterministic=deterministic)
        return a


class CombinedPolicy(BasePolicy):
    def __init__(self, policies, p=None):
        self.policies = policies
        for policy in self.policies:
            assert issubclass(policy.__class__, BasePolicy)
        if p is None:
            n = len(self.policies)
            p = np.ones(n) / n
        self.p = p

    def get_action(self, obs, deterministic=True):
        policy_idx = np.random.choice(np.arange(len(self.policies)), p=self.p)
        policy = self.policies[policy_idx]
        return policy.get_action(obs, deterministic=deterministic)


class GaussianNoisePolicy(BasePolicy):
    def __init__(self, policy: BasePolicy, sigma: float):
        self.policy = policy
        self.sigma = sigma

    def get_action(self, obs, deterministic=False):
        action = self.policy.get_action(obs, deterministic=deterministic)
        action += np.random.normal(loc=0, scale=self.sigma, size=action.shape)
        return action
