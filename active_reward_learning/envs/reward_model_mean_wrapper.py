from types import SimpleNamespace
from typing import Dict, Tuple, Union

import gym
import numpy as np


class RewardModelMeanWrapper(gym.RewardWrapper):
    def __init__(self, env: gym.Env, reward_model, debug=False, normalize=False):
        self.reward_model = reward_model
        self.debug = debug
        self.normalize = normalize

        # import outside leads to circular import
        from active_reward_learning.reward_models.kernels.linear import LinearKernel

        self.is_linear = isinstance(self.reward_model.gp_model.kernel, LinearKernel)
        super().__init__(env)

    def step(self, action: int) -> Tuple[Union[int, np.ndarray], float, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        orig_reward = reward
        info["true_reward"] = reward
        if self.debug:
            print()
            print("gp_repr", info["gp_repr"])
            print("reward true", reward)

        if self.is_linear:
            weight = self.reward_model.gp_model.linear_predictive_mean
            if self.normalize:
                weight /= np.linalg.norm(weight) + 1e-3
            # DL: This is necessary for performance reasons in the Mujoco environments
            reward = np.dot(info["gp_repr"], weight)
        else:
            if self.normalize:
                raise NotImplementedError()
            reward, _ = self.reward_model.gp_model.predict([info["gp_repr"]])

        if isinstance(reward, np.ndarray):
            assert reward.shape == (1,)
            reward = reward[0]
        info["inferred_reward"] = reward
        if self.debug:
            print("reward new", reward)
            print()
        return obs, reward, done, info

    @classmethod
    def load_from_model(cls, env, filename, debug=False):
        # importing here prevents some circular dependencies
        from active_reward_learning.reward_models.gaussian_process_linear import (
            LinearObservationGP,
        )

        gp_model = LinearObservationGP.load(filename)
        print(f"Loaded mode from {filename}")
        reward_model = SimpleNamespace()
        setattr(reward_model, "gp_model", gp_model)
        return cls(env, reward_model, debug=debug)
