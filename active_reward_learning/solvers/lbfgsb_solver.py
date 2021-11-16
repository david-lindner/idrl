import time
from typing import Callable, Optional

import numpy as np

from active_reward_learning.envs import (
    HighwayDriving,
    RewardModelMeanWrapper,
    RewardModelSampleWrapper,
)

from .base import BaseSolver


class LBFGSSolver(BaseSolver):
    def __init__(self, env, n_restarts=10):
        assert isinstance(env.unwrapped, HighwayDriving)
        super().__init__(env)

    def solve(
        self,
        n_episodes: int = 1,
        rewards: Optional[np.ndarray] = None,
        logging_callback: Optional[Callable] = None,
    ):
        t = time.time()
        if isinstance(self.env, RewardModelMeanWrapper):
            w = self.env.reward_model.gp_model.linear_predictive_mean
        elif isinstance(self.env, RewardModelSampleWrapper):
            w = self.env.theta
        else:
            w = None
        self.policy = self.env.get_optimal_policy(w=w, restarts=n_episodes)
        assert self.policy is not None
        self.solve_time += time.time() - t
        return self.policy
