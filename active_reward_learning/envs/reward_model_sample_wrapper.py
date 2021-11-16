from typing import Dict, Tuple, Union

import gym
import numpy as np

from active_reward_learning.util.helpers import pdf_multivariate_gauss


class RewardModelSampleWrapper(gym.RewardWrapper):
    def __init__(self, env: gym.Env, reward_model, normalize=False):
        self.normalize = normalize
        self.new_sample(reward_model)
        super().__init__(env)

    def step(self, action: int) -> Tuple[Union[int, np.ndarray], float, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        features = info["gp_repr"]
        if self.normalize:
            theta = self.theta / (np.linalg.norm(self.theta) + 1e-3)
        else:
            theta = self.theta
        reward = np.dot(features, theta)
        return obs, reward, done, info

    def _update_sample(self, reward_model, set_to_mean=False):
        mu = reward_model.gp_model.linear_predictive_mean
        cov = reward_model.gp_model.linear_predictive_cov
        if set_to_mean:
            self.theta = mu
        else:
            self.theta = np.random.multivariate_normal(mu, cov)
        posterior_prob = pdf_multivariate_gauss(self.theta, mu, cov)
        return posterior_prob

    def new_sample(self, reward_model):
        return self._update_sample(reward_model, set_to_mean=False)

    def set_sample_to_mean(self, reward_model):
        return self._update_sample(reward_model, set_to_mean=True)
