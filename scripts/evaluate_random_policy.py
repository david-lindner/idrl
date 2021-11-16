import sys

import gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

import active_reward_learning
from active_reward_learning.drlhp.reward_model import RewardModelNN
from active_reward_learning.envs.time_feature_wrapper import TimeFeatureWrapper


def main():
    env_id = sys.argv[1]
    n_episodes = 1000

    env = gym.make(env_id)

    returns = []

    for i in range(n_episodes):
        ret = 0
        done = False
        obs = env.reset()
        print(f"Episode {i}...", end=" ")
        while not done:
            action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            ret += reward

            if done:
                print("Return: ", ret)
                returns.append(ret)

    print("Mean return of random policy: ", np.mean(returns))


if __name__ == "__main__":
    main()
