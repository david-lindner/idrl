import os
from typing import Callable, Optional

import numpy as np

from active_reward_learning.common.policy import LinearPolicy
from active_reward_learning.envs import (
    RewardModelMeanWrapper,
    RewardModelSampleWrapper,
    TabularMDP,
)

from .base import BaseSolver


class ArgmaxSolver(BaseSolver):
    def __init__(
        self, env, n_eval=10, candidate_policies=None, candidate_policy_folder=None
    ):
        if candidate_policies is not None:
            self.candidate_policies = candidate_policies
        elif candidate_policy_folder is not None:
            self.candidate_policies = []
            for cur_path, dirs, files in os.walk(candidate_policy_folder):
                for filename in files:
                    if filename.endswith(".npy"):
                        print(f"Loading {filename}")
                        path = os.path.join(candidate_policy_folder, filename)
                        self.candidate_policies.append(LinearPolicy.load(path, env))
        else:
            self.candidate_policies = env.get_candidate_policies()
        self.n_eval = n_eval
        super().__init__(env)

    def solve(
        self,
        n_episodes: int = 1,
        rewards: Optional[np.ndarray] = None,
        logging_callback: Optional[Callable] = None,
    ):
        if isinstance(self.env.unwrapped, TabularMDP):
            if isinstance(self.env, RewardModelMeanWrapper):
                all_states = self.env.unwrapped.get_all_states_repr()
                rewards, _ = self.env.reward_model.gp_model.predict_multiple(all_states)
            elif isinstance(self.env, RewardModelSampleWrapper):
                raise Exception(
                    "RewardModelSampleWrapper is not supported with ArgmaxSolver."
                )
            elif rewards is None:
                rewards = self.env.rewards

        if rewards is not None:
            assert rewards.shape == (self.env.N_states,)

        best_policy = None
        best_value = -float("inf")
        for policy in self.candidate_policies:
            if rewards is None:
                value = policy.evaluate(self.env, N=self.n_eval, rollout=True)
            else:
                W = self.env.get_return_trafo_for_policy(policy)
                value = W.dot(rewards)
            # print("value", value)
            if value > best_value:
                best_policy, best_value = policy, value
        self.policy = best_policy
        return self.policy

    def evaluate(self, env=None, rollout=False, N=1):
        assert self.policy is not None
        if env is not None:
            return self.policy.evaluate(env, N=N, rollout=rollout)
        else:
            return self.policy.evaluate(self.env, N=N, rollout=rollout)
