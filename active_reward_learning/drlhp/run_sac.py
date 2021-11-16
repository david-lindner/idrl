import sys

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import active_reward_learning
from active_reward_learning.envs.time_feature_wrapper import TimeFeatureWrapper


def main():
    env_id = sys.argv[1]

    total_timesteps = 1e7

    env = gym.make(env_id)
    env = TimeFeatureWrapper(env)
    venv = DummyVecEnv([lambda: Monitor(env)])
    venv = VecNormalize(venv)

    model = SAC("MlpPolicy", venv, tensorboard_log="./tb_drlhp/", verbose=1)
    model.learn(
        total_timesteps=total_timesteps, log_interval=1, tb_log_name=f"sac_{env_id}"
    )
    model.save(f"sac_{env_id}_{total_timesteps}.zip")
    venv.save(f"sac_VecNormalize_{env_id}_{total_timesteps}.zip")

    obs = venv.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        venv.render()


if __name__ == "__main__":
    main()
