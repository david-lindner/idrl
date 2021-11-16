"""Adapted from https://github.com/rll/rllab."""

from active_reward_learning.envs.mujoco.maze_env import MazeEnv
from active_reward_learning.envs.mujoco.point import PointEnv


class PointMazeEnv(MazeEnv):
    MODEL_CLASS = PointEnv
