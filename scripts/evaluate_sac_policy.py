import sys

import gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import active_reward_learning
from active_reward_learning.envs.time_feature_wrapper import TimeFeatureWrapper


def main():
    policy_file = sys.argv[1]
    env_file = sys.argv[2]
    env_id = sys.argv[3]
    n_episodes = 1000

    env = gym.make(env_id)
    env = TimeFeatureWrapper(env)

    venv = DummyVecEnv([lambda: Monitor(env)])
    venv = VecNormalize.load(env_file, venv)
    venv.training = False
    venv.norm_reward = False

    model = SAC.load(policy_file)

    returns = []

    for i in range(n_episodes):
        ret = 0
        done = False
        obs = venv.reset()
        print(f"Episode {i}...", end=" ")
        while not done:
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, info = venv.step(action)
            ret += reward

            if done:
                print("Return: ", ret)
                returns.append(ret)

    print("Mean return of random policy: ", np.mean(returns))


if __name__ == "__main__":
    main()
