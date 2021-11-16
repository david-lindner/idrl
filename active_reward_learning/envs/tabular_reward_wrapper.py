from typing import Dict, Tuple, Union

import gym
import numpy as np


class TabularRewardWrapper(gym.Env):
    def __init__(self, tabular_env, rewards):
        self.tabular_env = tabular_env
        self.rewards = rewards
        self.observation_space = tabular_env.observation_space
        self.action_space = tabular_env.action_space
        self.N_states = self.tabular_env.N_states
        self.N_actions = self.tabular_env.N_actions
        self.get_observation = self.tabular_env.get_observation
        self.evaluate_policy = self.tabular_env.evaluate_policy
        self.current_state = self.tabular_env.current_state

    def step(self, action: int) -> Tuple[Union[int, np.ndarray], float, bool, Dict]:
        obs, reward, done, info = self.tabular_env.step(action)
        reward = self.rewards[self.tabular_env.current_state]
        self.current_state = self.tabular_env.current_state
        return obs, reward, done, info

    def reset(self) -> Union[int, np.ndarray]:
        obs = self.tabular_env.reset()
        reward = self.rewards[self.tabular_env.current_state]
        self.current_state = self.tabular_env.current_state
        return obs

    def set_rewards(self, rewards):
        self.rewards = rewards
