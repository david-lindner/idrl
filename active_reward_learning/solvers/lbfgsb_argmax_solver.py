import pickle
import time
from typing import Callable, Optional

import numpy as np

from active_reward_learning.common.policy import FixedPolicy
from active_reward_learning.envs import (
    HighwayDriving,
    RewardModelMeanWrapper,
    RewardModelSampleWrapper,
)

from .base import BaseSolver


def get_features_from_policy(env, policy):
    """Represent policies with average feature vector.

    This only makes sense for linear reward functions, but it is only used for the
    HighwayDriving environment.
    """
    assert isinstance(env.unwrapped, HighwayDriving)
    assert isinstance(policy, FixedPolicy)
    N = 10

    features = np.zeros(env.Ndim_repr)
    for i in range(N):
        obs = env.reset()
        done = False

        while not done:
            act = policy.get_action(obs)
            obs, reward, done, info = env.step(act)
            features += info["gp_repr"]

    features /= N
    return features


class LBFGSArgmaxSolver(BaseSolver):
    def __init__(
        self,
        env,
        solver_policies_file=None,
        candidate_policies=None,
        debug=False,
    ):
        assert isinstance(env.unwrapped, HighwayDriving)
        assert solver_policies_file is not None or candidate_policies is not None

        if solver_policies_file is not None:
            with open(solver_policies_file, "rb") as f:
                self.candidate_policies = pickle.load(f)
        elif candidate_policies is not None:
            self.candidate_policies = candidate_policies
        else:
            return NotImplementedError()

        self.features = []
        for policy in self.candidate_policies:
            features = get_features_from_policy(env, policy)
            self.features.append(features)

        assert len(self.candidate_policies) > 0
        assert len(self.candidate_policies) == len(self.features)

        self.debug = debug
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
            w = self.env.reward_w

        if n_episodes > 0:
            best_policy = self.env.get_optimal_policy(w=w, restarts=n_episodes)
            best_value = best_policy.evaluate(self.env, N=10, rollout=True)
        else:
            best_policy = None
            best_value = -float("inf")

        if self.debug:
            print("best_value", best_value)

        for policy, features in zip(self.candidate_policies, self.features):
            value = np.dot(features, w)
            if self.debug:
                print("value", value)
            if value > best_value:
                best_policy, best_value = policy, value
                if self.debug:
                    print("update, best_value", best_value)

        self.policy = best_policy
        assert self.policy is not None
        self.solve_time += time.time() - t
        return self.policy
