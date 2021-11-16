import itertools

import gym
import numpy as np
from gym.envs.registration import register
from gym.wrappers import TimeLimit

from active_reward_learning.common.policy import LinearPolicy
from active_reward_learning.envs import Gridworld, JunctionMDP, LimitedActionChain
from active_reward_learning.envs.gym_env import GymInfoWrapper, MujocoDirectionWrapper
from active_reward_learning.envs.hierarchical_rl_wrapper import HierarchicalRLWrapper


def get_junction(N, M, observation_type, observation_noise=0):
    env = JunctionMDP(
        rewards_chain=np.zeros(N),
        rewards_junction_top=np.ones(M) * 0.8,
        rewards_junction_bottom=np.linspace(-1, -0.3, M) ** 2 * (-1) + 1,
        discount_factor=0.99,
        episode_length=100,
        observation_type=observation_type,
        observation_noise=observation_noise,
    )
    env = GymInfoWrapper(env)
    return env


def get_junction_ordinal(x, observation_type, observation_noise=0):
    assert 0 < x < 1
    env = JunctionMDP(
        rewards_chain=np.array([0]),
        rewards_junction_top=np.array([0, 1]),
        rewards_junction_bottom=np.array([x, x]),
        discount_factor=0.99,
        episode_length=20,
        observation_type=observation_type,
        observation_noise=observation_noise,
    )
    env = GymInfoWrapper(env)
    return env


def get_chain(rewards, observation_type, observation_noise=0, normalize=True):
    assert len(rewards.shape) == 1
    n = rewards.shape[0]
    min_r = rewards.min()
    max_r = rewards.max()
    rewards = (rewards - min_r) / (max_r - min_r)
    env = LimitedActionChain(
        rewards,
        discount_factor=0.99,
        episode_length=100,
        block_N=n // 2,
        observation_type=observation_type,
        observation_noise=observation_noise,
    )
    env = GymInfoWrapper(env)
    return env


def get_gridworld(env_seed, observation_noise):
    env = Gridworld(
        width=10,
        height=10,
        n_objects=10,
        n_per_object=2,
        episode_length=100,
        discount_factor=0.99,
        observation_noise=observation_noise,
        wall_probability=0.3,
        env_seed=env_seed,
        random_action_prob=0,
        gaussian_peaks_as_rewards=False,
        add_optimal_policy_to_candidates=False,
        observation_type="state",
    )
    min_r = env.rewards.min()
    max_r = env.rewards.max()
    env.rewards = (env.rewards - min_r) / (max_r - min_r)
    # env = GymInfoWrapper(env)
    return env


def get_mujoco_direction(direction, gym_id):
    env = gym.make(gym_id)
    env = MujocoDirectionWrapper(env, direction=direction)
    return env


def get_mujoco_maze_goal(gym_id, goal_x, goal_y, **kwargs):
    env = gym.make(gym_id)
    env.unwrapped.goal_x = goal_x
    env.unwrapped.goal_y = goal_y
    env.unwrapped.reward_w = env.unwrapped._get_reward_w()
    return env


def get_hierarchical(gym_id, policies):
    env = gym.make(gym_id, use_inner_obs=False)
    policies = [LinearPolicy.load(p, env) for p in policies]
    env = HierarchicalRLWrapper(env, policies=policies)
    return env


NM_list = [(5, 2), (15, 5), (20, 20), (150, 50), (1500, 500)]

for (N, M), observation_type in itertools.product(NM_list, ("raw", "state")):
    register(
        id=f"JunctionEI-N{N}-M{M}-{observation_type.capitalize()}-v0",
        entry_point="active_reward_learning:get_junction",
        kwargs={"N": N, "M": M, "observation_type": observation_type},
    )

for observation_type in ("raw", "state"):
    register(
        id=f"Chain-{observation_type.capitalize()}-v0",
        entry_point="active_reward_learning:get_chain",
        kwargs={"observation_type": observation_type},
    )

register(
    id="GridworldPointRandom-v0", entry_point="active_reward_learning:get_gridworld"
)

for observation_type in ("raw", "state"):
    register(
        id=f"JunctionOrdinal-{observation_type.capitalize()}-v0",
        entry_point="active_reward_learning:get_junction_ordinal",
        kwargs={"observation_type": observation_type},
    )

register(
    id="HighwayDriving-FixedReward-v0",
    entry_point="active_reward_learning.envs.highway_driving:get_highway",
    kwargs={"random_reward": False},
    max_episode_steps=50,
)

register(
    id="HighwayDriving-RandomReward-v0",
    entry_point="active_reward_learning.envs.highway_driving:get_highway",
    kwargs={"random_reward": True},
    max_episode_steps=50,
)

# Mujoco environments

try:
    import mujoco_py

    mujoco_available = True
except ImportError:
    mujoco_available = False
    print("Warning: Mujoco not available.")

if mujoco_available:
    from active_reward_learning.envs.mujoco.ant import AntEnv
    from active_reward_learning.envs.mujoco.point import PointEnv
    from active_reward_learning.envs.mujoco.swimmer import SwimmerEnv
    from active_reward_learning.envs.time_feature_wrapper import TimeFeatureWrapper

    register(
        id="InvertedPendulum-Penalty-v2",
        entry_point="active_reward_learning.envs.mujoco.inverted_pendulum:InvertedPendulumEnv",
        kwargs={"penalty": True},
        max_episode_steps=100,
    )
    register(
        id="InvertedDoublePendulum-Penalty-v2",
        entry_point="active_reward_learning.envs.mujoco.inverted_double_pendulum:InvertedDoublePendulumEnv",
        kwargs={"penalty": True},
        max_episode_steps=100,
    )

    register(
        id="InvertedPendulum-Penalty-Long-v2",
        entry_point="active_reward_learning.envs.mujoco.inverted_pendulum:InvertedPendulumEnv",
        kwargs={"penalty": True},
        max_episode_steps=1000,
    )
    register(
        id="InvertedDoublePendulum-Penalty-Long-v2",
        entry_point="active_reward_learning.envs.mujoco.inverted_double_pendulum:InvertedDoublePendulumEnv",
        kwargs={"penalty": True},
        max_episode_steps=1000,
    )

    register(
        id="HalfCheetah-Short-v3",
        entry_point="gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv",
        kwargs={
            "exclude_current_positions_from_observation": True,
        },
        max_episode_steps=300,
    )
    register(
        id="Hopper-Short-v3",
        entry_point="gym.envs.mujoco.hopper_v3:HopperEnv",
        kwargs={
            "terminate_when_unhealthy": False,
            "exclude_current_positions_from_observation": True,
        },
        max_episode_steps=300,
    )
    register(
        id="Walker2d-Short-v3",
        entry_point="gym.envs.mujoco.walker2d_v3:Walker2dEnv",
        kwargs={
            "terminate_when_unhealthy": False,
            "exclude_current_positions_from_observation": True,
        },
        max_episode_steps=300,
    )

    register(
        id="HalfCheetah-Long-v3",
        entry_point="gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv",
        kwargs={
            "exclude_current_positions_from_observation": True,
        },
        max_episode_steps=1000,
    )
    register(
        id="Swimmer-Long-v3",
        entry_point="gym.envs.mujoco.swimmer_v3:SwimmerEnv",
        kwargs={
            "exclude_current_positions_from_observation": True,
        },
        max_episode_steps=1000,
    )
    register(
        id="Hopper-Long-v3",
        entry_point="gym.envs.mujoco.hopper_v3:HopperEnv",
        kwargs={
            "terminate_when_unhealthy": False,
            "exclude_current_positions_from_observation": True,
        },
        max_episode_steps=1000,
    )
    register(
        id="Walker2d-Long-v3",
        entry_point="gym.envs.mujoco.walker2d_v3:Walker2dEnv",
        kwargs={
            "terminate_when_unhealthy": False,
            "exclude_current_positions_from_observation": True,
        },
        max_episode_steps=1000,
    )
    register(
        id="Ant-Long-v3",
        entry_point="gym.envs.mujoco.ant_v3:AntEnv",
        kwargs={
            "terminate_when_unhealthy": False,
            "exclude_current_positions_from_observation": True,
        },
        max_episode_steps=1000,
    )

    register(
        id="HalfCheetah-Long-WithPos-v3",
        entry_point="gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv",
        kwargs={
            "exclude_current_positions_from_observation": False,
        },
        max_episode_steps=1000,
    )
    register(
        id="Swimmer-Long-WithPos-v3",
        entry_point="gym.envs.mujoco.swimmer_v3:SwimmerEnv",
        kwargs={
            "exclude_current_positions_from_observation": False,
        },
        max_episode_steps=1000,
    )
    register(
        id="Hopper-Long-WithPos-v3",
        entry_point="gym.envs.mujoco.hopper_v3:HopperEnv",
        kwargs={
            "terminate_when_unhealthy": False,
            "exclude_current_positions_from_observation": False,
        },
        max_episode_steps=1000,
    )
    register(
        id="Walker2d-Long-WithPos-v3",
        entry_point="gym.envs.mujoco.walker2d_v3:Walker2dEnv",
        kwargs={
            "terminate_when_unhealthy": False,
            "exclude_current_positions_from_observation": False,
        },
        max_episode_steps=1000,
    )
    register(
        id="Ant-Long-WithPos-v3",
        entry_point="gym.envs.mujoco.ant_v3:AntEnv",
        kwargs={
            "terminate_when_unhealthy": False,
            "exclude_current_positions_from_observation": False,
        },
        max_episode_steps=1000,
    )

    ###

    register(
        id="PointEnv-v2",
        entry_point="active_reward_learning.envs.mujoco.point:PointEnv",
        kwargs={"random_orientation": True},
        max_episode_steps=1000,
    )
    register(
        id="SwimmerEnv-v2",
        entry_point="active_reward_learning.envs.mujoco.swimmer:SwimmerEnv",
        kwargs={},
        max_episode_steps=1000,
    )
    register(
        id="AntEnv-v2",
        entry_point="active_reward_learning.envs.mujoco.ant:AntEnv",
        kwargs={},
        max_episode_steps=1000,
    )

    register(
        id="PointMaze1D-v2",
        entry_point="active_reward_learning.envs.mujoco.maze_env:MazeEnv",
        kwargs={
            "model_cls": PointEnv,
            "maze_id": "Maze1D-Big",
            "manual_collision": True,
            "maze_size_scaling": 6,
            "init_position_probs": [1, 0],
            "use_inner_obs": False,
        },
        max_episode_steps=200,
    )

    register(
        id="SwimmerMaze1D-v2",
        entry_point="active_reward_learning.envs.mujoco.maze_env:MazeEnv",
        kwargs={
            "model_cls": SwimmerEnv,
            "maze_id": "Maze1D-Big",
            "manual_collision": True,
            "maze_size_scaling": 3,
            "init_position_probs": [1, 0],
            "use_inner_obs": False,
        },
        max_episode_steps=1000,
    )

    register(
        id="SwimmerMaze1D-Big-v2",
        entry_point="active_reward_learning.envs.mujoco.maze_env:MazeEnv",
        kwargs={
            "model_cls": SwimmerEnv,
            "maze_id": "Maze1D-Big",
            "manual_collision": True,
            "maze_size_scaling": 3,
            "init_position_probs": [1, 0],
            "use_inner_obs": False,
        },
        max_episode_steps=400,
    )

    register(
        id="AntMaze1D-v2",
        entry_point="active_reward_learning.envs.mujoco.maze_env:MazeEnv",
        kwargs={
            "model_cls": AntEnv,
            "maze_id": "Maze1D-Big",
            "manual_collision": True,
            "maze_size_scaling": 4,
            "init_position_probs": [1, 0],
            "use_inner_obs": False,
        },
        max_episode_steps=500,
    )

    register(
        id="Point-Right-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [1, 0], "gym_id": "PointEnv-v2"},
        max_episode_steps=100,
    )
    register(
        id="Point-Left-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [-1, 0], "gym_id": "PointEnv-v2"},
        max_episode_steps=100,
    )
    register(
        id="Point-Up-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [0, 1], "gym_id": "PointEnv-v2"},
        max_episode_steps=100,
    )
    register(
        id="Point-Down-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [0, -1], "gym_id": "PointEnv-v2"},
        max_episode_steps=100,
    )

    register(
        id="Swimmer-Right-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [1, 0], "gym_id": "SwimmerEnv-v2"},
        max_episode_steps=300,
    )
    register(
        id="Swimmer-Left-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [-1, 0], "gym_id": "SwimmerEnv-v2"},
        max_episode_steps=300,
    )
    register(
        id="Swimmer-Up-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [0, 1], "gym_id": "SwimmerEnv-v2"},
        max_episode_steps=300,
    )
    register(
        id="Swimmer-Down-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [0, -1], "gym_id": "SwimmerEnv-v2"},
        max_episode_steps=300,
    )

    register(
        id="Ant-Right-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [1, 0], "gym_id": "AntEnv-v2"},
        max_episode_steps=300,
    )
    register(
        id="Ant-Left-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [-1, 0], "gym_id": "AntEnv-v2"},
        max_episode_steps=300,
    )
    register(
        id="Ant-Up-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [0, 1], "gym_id": "AntEnv-v2"},
        max_episode_steps=300,
    )
    register(
        id="Ant-Down-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [0, -1], "gym_id": "AntEnv-v2"},
        max_episode_steps=300,
    )

    register(
        id="Ant-Up-Right-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [1, 1], "gym_id": "AntEnv-v2"},
        max_episode_steps=300,
    )
    register(
        id="Ant-Down-Right-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [1, -1], "gym_id": "AntEnv-v2"},
        max_episode_steps=300,
    )
    register(
        id="Ant-Down-Left-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [-1, -1], "gym_id": "AntEnv-v2"},
        max_episode_steps=300,
    )
    register(
        id="Ant-Up-Left-v2",
        entry_point="active_reward_learning:get_mujoco_direction",
        kwargs={"direction": [-1, 1], "gym_id": "AntEnv-v2"},
        max_episode_steps=300,
    )

    register(
        id="PointMaze1D-Hierarchical-v2",
        entry_point="active_reward_learning:get_hierarchical",
        kwargs={
            "gym_id": "PointMaze1D-v2",
            "policies": [
                "policies/policy_point_up.npy",
                "policies/policy_point_right.npy",
                "policies/policy_point_down.npy",
                "policies/policy_point_left.npy",
            ],
        },
        max_episode_steps=200,
    )
    register(
        id="SwimmerMaze1D-Hierarchical-v2",
        entry_point="active_reward_learning:get_hierarchical",
        kwargs={
            "gym_id": "SwimmerMaze1D-v2",
            "policies": [
                "policies/policy_swimmer_up.npy",
                "policies/policy_swimmer_right.npy",
                "policies/policy_swimmer_down.npy",
                "policies/policy_swimmer_left.npy",
            ],
        },
        max_episode_steps=1000,
    )
    register(
        id="AntMaze1D-Hierarchical-v2",
        entry_point="active_reward_learning:get_hierarchical",
        kwargs={
            "gym_id": "AntMaze1D-v2",
            "policies": [
                "policies/policy_ant_up.npy",
                "policies/policy_ant_right.npy",
                "policies/policy_ant_down.npy",
                "policies/policy_ant_left.npy",
            ],
        },
        max_episode_steps=500,
    )
