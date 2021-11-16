"""Adapted from https://github.com/rll/rllab."""

from active_reward_learning.envs.mujoco.ant import AntEnv
from active_reward_learning.envs.mujoco.maze_env import MazeEnv


class AntMazeEnv(MazeEnv):
    MODEL_CLASS = AntEnv
