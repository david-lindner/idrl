from abc import ABC, abstractmethod
from typing import Optional

import gym
import numpy as np

from active_reward_learning.common.policy import BasePolicy


class BaseSolver(ABC):
    def __init__(self, env: gym.Env):
        self.env = env
        self.solve_time: float = 0
        self.policy: Optional[BasePolicy] = None

    @abstractmethod
    def solve(
        self, n_episodes: int = 100, rewards: Optional[np.ndarray] = None
    ) -> BasePolicy:
        raise NotImplementedError()

    def update_policy(self) -> None:
        raise NotImplementedError()

    def evaluate(self, env=None, N=1):
        assert self.policy is not None
        if env is not None:
            return self.policy.evaluate(env, N=N)
        else:
            return self.policy.evaluate(self.env, N=N)

    def set_env(self, env):
        self.env = env
