import gym

import active_reward_learning


def test_collisions():
    try:
        import mujoco_py

        mujoco_available = True
    except ModuleNotFoundError:
        mujoco_available = False

    if mujoco_available:
        env = gym.make("PointMazeT-v2")

        for i in range(10):
            env.reset()
            for t in range(10000):
                a = env.action_space.sample()
                _, _, _, _ = env.step(a)
                x, y = env.unwrapped._get_grid_xy()
                assert env.MAZE_STRUCTURE[y][x] != 1  # not inside wall
