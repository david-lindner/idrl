import gym
import numpy as np

from active_reward_learning.util.helpers import softmax


class HierarchicalRLWrapper(gym.RewardWrapper):
    def __init__(self, env, policies):
        self.policies = policies
        self.n = len(policies) + 1  # last action for doing nothing
        super().__init__(env)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.n,))

    def step(self, action):
        # print(action)
        assert self.action_space.contains(action)
        wrapped_obs = self.env.wrapped_env._get_obs()

        action = softmax(action)

        combined_action = np.zeros(self.env.action_space.shape[0])

        for factor, policy in zip(action[:-1], self.policies):
            policy_action = policy.get_action(wrapped_obs)
            combined_action += factor * policy_action

        obs, reward, done, info = self.env.step(combined_action)
        return obs, reward, done, info
