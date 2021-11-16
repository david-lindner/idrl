import time
from typing import Callable, Optional

import numpy as np

from active_reward_learning.common import BasePolicy
from active_reward_learning.envs import (
    RewardModelMeanWrapper,
    RewardModelSampleWrapper,
    TabularMDP,
)

from .base import BaseSolver


class LPSolver(BaseSolver):
    def __init__(self, env):
        assert isinstance(env, TabularMDP) or isinstance(env.unwrapped, TabularMDP)
        super().__init__(env)

    def solve(
        self,
        n_episodes: int = 100,
        rewards: Optional[np.ndarray] = None,
        logging_callback: Optional[Callable] = None,
    ) -> BasePolicy:
        if isinstance(self.env, RewardModelMeanWrapper):
            all_states = self.env.unwrapped.get_all_states_repr()
            rewards, _ = self.env.reward_model.gp_model.predict_multiple(all_states)
        elif isinstance(self.env, RewardModelSampleWrapper):
            raise Exception("RewardModelSampleWrapper is not supported with LPSolver.")

        if rewards is not None:
            assert rewards.shape == (self.env.N_states,)

        t = time.time()
        self.policy = self.env.get_lp_solution(rewards)
        assert self.policy is not None
        self.solve_time += time.time() - t
        return self.policy

    def evaluate(self, env=None, rollout=False, N=1):
        assert self.policy is not None
        if env is not None:
            return self.policy.evaluate(env, N=N, rollout=rollout)
        else:
            return self.policy.evaluate(self.env, N=N, rollout=rollout)
