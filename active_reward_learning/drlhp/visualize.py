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
    policy_file = sys.argv[1]
    env_file = sys.argv[2]
    env_id = sys.argv[3]

    env = gym.make(env_id)
    env = TimeFeatureWrapper(env)

    venv = DummyVecEnv([lambda: Monitor(env)])
    venv = VecNormalize.load(env_file, venv)
    venv.training = False
    venv.norm_reward = False

    model = SAC.load(policy_file)

    obs = venv.reset()
    ret = 0
    t = 0
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=False)

        obs, reward, done, info = venv.step(action)
        ret += reward

        t += 1
        venv.render()

        print(f"t {t}  reward {reward}  return {ret}")

        if done:
            obs = venv.reset()
            print("Return", ret)
            ret = 0
            t = 0

    venv.close()


if __name__ == "__main__":
    main()
