import gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import active_reward_learning
from active_reward_learning.drlhp.reward_model import (
    RewardModelEnvWrapper,
    RewardModelNN,
    TrueRewardCallback,
)


def main():
    env = gym.make("InvertedPendulum-Penalty-v2")

    reward_model = RewardModelNN.load("pendulum_reward_model.pt")
    env = RewardModelEnvWrapper(env, reward_model)
    env = Monitor(env, info_keywords=("true_reward",))

    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize(venv)

    callback = TrueRewardCallback()
    model = SAC("MlpPolicy", venv, verbose=1)
    model.learn(total_timesteps=30000, log_interval=1, callback=callback)
    model.save("sac_model_pendulum_30k.zip")

    obs = venv.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        venv.render()
        if done:
            obs = venv.reset()


if __name__ == "__main__":
    main()
