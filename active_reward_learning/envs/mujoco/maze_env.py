"""Adapted from https://github.com/rll/rllab."""

import os
import tempfile
import xml.etree.ElementTree as ET
from functools import partial

import gym
import numpy as np

from active_reward_learning.envs.mujoco import maze_env_utils
from active_reward_learning.envs.mujoco.point import PointEnv

# Directory that contains mujoco xml files.
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

EPS = 0.001


def potential(goal_x, goal_y, x, y):
    r2 = (goal_x - x) ** 2 + (goal_y - y) ** 2
    c = 100
    p = c * np.exp(-r2)
    return p


# def potential(goal_x, goal_y, x, y):
#     r2 = (goal_x - x)**2 + (goal_y - y)**2
#     c = 100
#     return -c*r2

# def potential(goal_x, goal_y, x, y):
#     r2 = np.abs(goal_x - x) + np.abs(goal_y - y)
#     c = 100
#     return -r2

# def potential(goal_x, goal_y, x, y):
#     r2 = max((goal_x - x) ** 2 + (goal_y - y) ** 2, EPS)
#     c = 100
#     return c / r2


class MazeEnv(gym.Env):
    # MODEL_CLASS: Optional[object] = None

    MAZE_HEIGHT = None
    MAZE_SIZE_SCALING = None

    def __reduce__(self):
        return (
            MazeEnv,
            (
                self.model_cls,
                self._maze_id,
                self.MAZE_HEIGHT,
                self.MAZE_SIZE_SCALING,
                self._manual_collision,
                self._init_position_probs,
                self._use_inner_obs,
                self._only_inner_obs,
                self._potential_based_reward,
                self._random_reward,
            ),
        )

    def __init__(
        self,
        model_cls=PointEnv,
        maze_id=None,
        maze_height=0.3,
        maze_size_scaling=8,
        manual_collision=False,
        init_position_probs=None,
        use_inner_obs=True,
        only_inner_obs=False,
        potential_based_reward=False,
        random_reward=False,
        constant_feature=True,
        *args,
        **kwargs,
    ):
        self._maze_id = maze_id

        # model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise "MODEL_CLASS unspecified!"
        self.model_cls = model_cls
        xml_path = os.path.join(MODEL_DIR, model_cls.FILE)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        self.MAZE_HEIGHT = height = maze_height
        self.MAZE_SIZE_SCALING = size_scaling = maze_size_scaling
        self._manual_collision = manual_collision
        self._init_position_probs = init_position_probs
        self._use_inner_obs = use_inner_obs
        self._only_inner_obs = only_inner_obs
        self._potential_based_reward = potential_based_reward
        self._random_reward = random_reward
        self._constant_feature = constant_feature

        self.MAZE_STRUCTURE = structure = maze_env_utils.construct_maze(
            maze_id=self._maze_id
        )
        self.elevated = any(
            -1 in row for row in structure
        )  # Elevate the maze to allow for falling.
        self.blocks = any(
            any(maze_env_utils.can_move(r) for r in row) for row in structure
        )  # Are there any movable blocks?

        torso_x, torso_y = self._find_robot()
        if torso_x is None or torso_y is None:
            print("Warning: Robot not set")
            torso_x, torso_y = 0, 0
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y
        self._init_positions = [
            (x - torso_x, y - torso_y) for x, y in self._find_all_robots()
        ]

        if len(self._init_positions) > 1 and self._init_position_probs is not None:
            assert len(self._init_position_probs) == len(self._init_positions)
            assert sum(self._init_position_probs) == 1

        self.width = len(structure[0])
        self.height = len(structure)
        self.t = 0  # DL: fixes issues when accessing observation space before resetting

        self.goal_x, self.goal_y = self._find_goal()
        if self.goal_x is None or self.goal_y is None:
            print("Warning: Goal not set")
            self.goal_x, self.goal_y = 0, 0
        self.goal_x /= self.MAZE_SIZE_SCALING
        self.goal_y /= self.MAZE_SIZE_SCALING

        if self._potential_based_reward:
            self.potential_functions = []
            i = 0
            for x in range(self.width):
                for y in range(self.height):
                    print(i, x, y)
                    i += 1
                    potential_func = partial(potential, x, y)
                    self.potential_functions.append(potential_func)
        else:
            self.potential_functions = None

        self._xy_to_rowcol = lambda x, y: (
            2 + (y + size_scaling / 2) / size_scaling,
            2 + (x + size_scaling / 2) / size_scaling,
        )
        self._view = np.zeros(
            [5, 5, 3]
        )  # walls (immovable), chasms (fall), movable blocks

        height_offset = 0.0
        if self.elevated:
            # Increase initial z-pos of ant.
            height_offset = height * size_scaling
            torso = tree.find(".//body[@name='torso']")
            torso.set("pos", "0 0 %.2f" % (0.75 + height_offset))
        if self.blocks:
            # If there are movable blocks, change simulation settings to perform
            # better contact detection.
            default = tree.find(".//default")
            default.find(".//geom").set("solimp", ".995 .995 .01")

        self.movable_blocks = []
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                struct = structure[i][j]

                # for figure
                border = 0.1

                # black tiles for grid cells
                # ET.SubElement(
                #     worldbody,
                #     "geom",
                #     name="floor_%d_%d" % (i, j),
                #     pos="%f %f %f"
                #     % (j * size_scaling - torso_x, i * size_scaling - torso_y, 0.01,),
                #     size="%f %f %f"
                #     % (
                #         0.5 * size_scaling,
                #         0.5 * size_scaling,
                #         height / 2 * size_scaling,
                #     ),
                #     type="plane",
                #     material="",
                #     contype="1",
                #     conaffinity="1",
                #     rgba="0 0 0 1",
                # )

                if self.elevated and struct not in [-1]:
                    # Create elevated platform.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name="elevated_%d_%d" % (i, j),
                        pos="%f %f %f"
                        % (
                            j * size_scaling - torso_x,
                            i * size_scaling - torso_y,
                            height / 2 * size_scaling,
                        ),
                        size="%f %f %f"
                        % (
                            0.5 * size_scaling,
                            0.5 * size_scaling,
                            height / 2 * size_scaling,
                        ),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.9 0.9 1",
                    )
                if struct == 1:  # Unmovable block.
                    # Offset all coordinates so that robot starts at the origin.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f"
                        % (
                            j * size_scaling - torso_x,
                            i * size_scaling - torso_y,
                            height_offset + height / 2 * size_scaling,
                        ),
                        size="%f %f %f"
                        % (
                            0.5 * size_scaling,
                            0.5 * size_scaling,
                            height / 2 * size_scaling,
                        ),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1",
                    )
                if struct == "g":  # Unmovable block.
                    goal_scaling = size_scaling
                    # Offset all coordinates so that robot starts at the origin.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name="goal_%d_%d" % (i, j),
                        pos="%f %f %f"
                        % (
                            j * size_scaling - torso_x + border,
                            i * size_scaling - torso_y + border,
                            0.02,
                        ),
                        size="%f %f %f"
                        % (
                            0.5 * goal_scaling - border,
                            0.5 * goal_scaling - border,
                            height / 2 * goal_scaling,
                        ),
                        type="plane",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.1 0.4 0.1 1",
                    )
                elif maze_env_utils.can_move(struct):  # Movable block.
                    # The "falling" blocks are shrunk slightly and increased in mass to
                    # ensure that it can fall easily through a gap in the platform blocks.
                    name = "movable_%d_%d" % (i, j)
                    self.movable_blocks.append((name, struct))
                    falling = maze_env_utils.can_move_z(struct)
                    spinning = maze_env_utils.can_spin(struct)
                    x_offset = 0.25 * size_scaling if spinning else 0.0
                    y_offset = 0.0
                    shrink = 0.1 if spinning else 0.99 if falling else 1.0
                    height_shrink = 0.1 if spinning else 1.0
                    movable_body = ET.SubElement(
                        worldbody,
                        "body",
                        name=name,
                        pos="%f %f %f"
                        % (
                            j * size_scaling - torso_x + x_offset,
                            i * size_scaling - torso_y + y_offset,
                            height_offset + height / 2 * size_scaling * height_shrink,
                        ),
                    )
                    ET.SubElement(
                        movable_body,
                        "geom",
                        name="block_%d_%d" % (i, j),
                        pos="0 0 0",
                        size="%f %f %f"
                        % (
                            0.5 * size_scaling * shrink,
                            0.5 * size_scaling * shrink,
                            height / 2 * size_scaling * height_shrink,
                        ),
                        type="box",
                        material="",
                        mass="0.001" if falling else "0.0002",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.1 0.1 1",
                    )
                    if maze_env_utils.can_move_x(struct):
                        ET.SubElement(
                            movable_body,
                            "joint",
                            armature="0",
                            axis="1 0 0",
                            damping="0.0",
                            limited="true" if falling else "false",
                            range="%f %f" % (-size_scaling, size_scaling),
                            margin="0.01",
                            name="movable_x_%d_%d" % (i, j),
                            pos="0 0 0",
                            type="slide",
                        )
                    if maze_env_utils.can_move_y(struct):
                        ET.SubElement(
                            movable_body,
                            "joint",
                            armature="0",
                            axis="0 1 0",
                            damping="0.0",
                            limited="true" if falling else "false",
                            range="%f %f" % (-size_scaling, size_scaling),
                            margin="0.01",
                            name="movable_y_%d_%d" % (i, j),
                            pos="0 0 0",
                            type="slide",
                        )
                    if maze_env_utils.can_move_z(struct):
                        ET.SubElement(
                            movable_body,
                            "joint",
                            armature="0",
                            axis="0 0 1",
                            damping="0.0",
                            limited="true",
                            range="%f 0" % (-height_offset),
                            margin="0.01",
                            name="movable_z_%d_%d" % (i, j),
                            pos="0 0 0",
                            type="slide",
                        )
                    if maze_env_utils.can_spin(struct):
                        ET.SubElement(
                            movable_body,
                            "joint",
                            armature="0",
                            axis="0 0 1",
                            damping="0.0",
                            limited="false",
                            name="spinable_%d_%d" % (i, j),
                            pos="0 0 0",
                            type="ball",
                        )
                else:
                    pass
                    # for figure
                    # ET.SubElement(
                    #     worldbody,
                    #     "geom",
                    #     name="floor_inner_%d_%d" % (i, j),
                    #     pos="%f %f %f"
                    #     % (
                    #         j * size_scaling - torso_x + border,
                    #         i * size_scaling - torso_y + border,
                    #         0.02,
                    #     ),
                    #     size="%f %f %f"
                    #     % (
                    #         0.5 * size_scaling - border,
                    #         0.5 * size_scaling - border,
                    #         height / 2 * size_scaling,
                    #     ),
                    #     type="plane",
                    #     material="",
                    #     contype="1",
                    #     conaffinity="1",
                    #     rgba="1 1 1 1",
                    # )

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if "name" not in geom.attrib:
                raise Exception("Every geom of the torso must have a name " "defined")

        _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
        tree.write(file_path)

        self.wrapped_env = model_cls(*args, file_path=file_path, **kwargs)
        self.metadata = self.wrapped_env.metadata

        if self._random_reward:
            self.reward_w = self._sample_random_w()
        else:
            self.reward_w = self._get_reward_w()

    def _get_reward_w(self):
        if self._maze_id.startswith("Maze1D"):
            # remove border for features
            height = self.height - 2
            width = self.width - 2
            goal_x = self.goal_x - 1
            goal_y = self.goal_y - 1
        else:
            height = self.height
            width = self.width
            goal_x = self.goal_x
            goal_y = self.goal_y

        # note this doesn't work for general mazes which would require a way
        # to find the correct path to do the reward shaping
        # this works only for the T maze and the 1D maze
        if self._constant_feature:
            l = 3
        else:
            l = 2

        w = np.zeros(width * height * l)
        for robot_x in range(width):
            for robot_y in range(height):
                idx = np.ravel_multi_index([robot_y, robot_x], [height, width])
                # w[l * idx : l * (idx + 1)] = [1, 0, 0]
                # w[l * idx : l * (idx + 1)] = [1, 0]

                # sparse reward
                # if robot_x == goal_x and robot_y == goal_y:
                #     w[l * idx : l * (idx + 1)] = [0, 0, 1]
                # else:
                #     w[l * idx : l * (idx + 1)] = [0, 0, 0]

                # shaped reward
                if self._constant_feature:
                    if robot_x < goal_x:
                        w[l * idx : l * (idx + 1)] = [1, 0, 0]
                    elif robot_x > goal_x:
                        w[l * idx : l * (idx + 1)] = [-1, 0, 0]
                    elif robot_y < goal_y:
                        w[l * idx : l * (idx + 1)] = [0, 1, 0]
                    elif robot_y > goal_y:
                        w[l * idx : l * (idx + 1)] = [0, -1, 0]
                    else:  # robot is at goal
                        w[l * idx : l * (idx + 1)] = [0, 0, 0]  # [0, 0, 1]
                else:
                    if robot_x < goal_x:
                        w[l * idx : l * (idx + 1)] = [1, 0]
                    elif robot_x > goal_x:
                        w[l * idx : l * (idx + 1)] = [-1, 0]
                    elif robot_y < goal_y:
                        w[l * idx : l * (idx + 1)] = [0, 1]
                    elif robot_y > goal_y:
                        w[l * idx : l * (idx + 1)] = [0, -1]
                    else:  # robot is at goal
                        w[l * idx : l * (idx + 1)] = [0, 0]  # [0, 0, 1]
        return w

    def _sample_random_w(self):
        assert self._potential_based_reward
        from active_reward_learning.util.helpers import softmax

        # w = np.random.random(self.Ndim_repr)
        # w /= np.linalg.norm(w)
        w = np.random.multivariate_normal(
            np.zeros(self.Ndim_repr), np.identity(self.Ndim_repr)
        )
        w = softmax(w)

        w = np.zeros_like(w)
        w[10] = 1

        return w

    def get_ori(self):
        return self.wrapped_env.get_ori()

    def _xy_to_grid_coords(self, x, y):
        size_scaling = self.MAZE_SIZE_SCALING
        grid_x = (x + self._init_torso_x + 0.5 * size_scaling) / size_scaling
        grid_y = (y + self._init_torso_y + 0.5 * size_scaling) / size_scaling
        return grid_x, grid_y

    def _get_grid_xy(self, round=True):
        x, y = self.wrapped_env.get_xy()
        grid_x, grid_y = self._xy_to_grid_coords(x, y)
        if round:
            grid_x, grid_y = int(grid_x), int(grid_y)
        return grid_x, grid_y

    def _get_obs(self):
        if self._only_inner_obs:
            return self.wrapped_env._get_obs()

        if self._maze_id.startswith("Maze1D"):
            # remove border for features
            height = self.height - 2
            width = self.width - 2
        else:
            height = self.height
            width = self.width

        grid_x, grid_y = self._get_grid_xy(round=False)
        if self._maze_id.startswith("Maze1D"):
            grid_x -= 1
            grid_y -= 1

        ###
        # inner_obs = self.wrapped_env._get_obs()
        # inner_obs = np.array(list(inner_obs) + [1])
        #
        # len_obs = len(inner_obs)
        # obs = np.zeros(2 * len_obs)
        # obs[:len_obs] = inner_obs
        # if grid_x > self.goal_x:
        #     obs[len_obs:] = inner_obs
        ###

        ###
        inner_obs = self.wrapped_env._get_obs()
        inner_obs = np.array(list(inner_obs) + [1])
        len_obs = len(inner_obs)

        grid_x, grid_y = self._get_grid_xy(round=True)
        if self._maze_id.startswith("Maze1D"):
            grid_x -= 1
            grid_y -= 1
        idx = np.ravel_multi_index([grid_y, grid_x], [height, width])

        if self._use_inner_obs:
            obs = np.zeros(width * height * len_obs)
            obs[idx * len_obs : (idx + 1) * len_obs] = inner_obs
        else:
            obs = np.zeros(len_obs + width * height * len_obs)
            obs[:len_obs] = inner_obs
            obs[len_obs + idx] = 1
        ###

        return obs

    def _get_gp_repr(self, old_pos, new_pos):
        if self._potential_based_reward:
            x0, y0 = old_pos
            x1, y1 = new_pos
            # print("x0, y0 1", x0, y0)
            # print("x1, y1 1", x1, y1)
            x0, y0 = self._xy_to_grid_coords(x0, y0)
            x1, y1 = self._xy_to_grid_coords(x1, y1)
            # print("x0, y0 2", x0, y0)
            # print("x1, y1 2", x1, y1)
            gp_repr = []
            gamma = 1
            for potential_func in self.potential_functions:
                f = potential_func(x1, y1) - gamma * potential_func(x0, y0)
                gp_repr.append(f)
            return np.array(gp_repr)
        else:
            if self._maze_id.startswith("Maze1D"):
                # remove border for features
                height = self.height - 2
                width = self.width - 2
            else:
                height = self.height
                width = self.width

            if self._constant_feature:
                len_obs = 3
            else:
                len_obs = 2
            grid_x, grid_y = self._get_grid_xy()

            if self._maze_id.startswith("Maze1D"):
                grid_x -= 1
                grid_y -= 1

            idx = np.ravel_multi_index([grid_y, grid_x], [height, width])
            gp_repr = np.zeros(width * height * len_obs)
            dx = new_pos[0] - old_pos[0]
            dy = new_pos[1] - old_pos[1]

            if self._constant_feature:
                gp_repr[idx * len_obs : (idx + 1) * len_obs] = [dx, dy, 1]
            else:
                gp_repr[idx * len_obs : (idx + 1) * len_obs] = [dx, dy]
            return gp_repr

    def reset(self):
        self.t = 0
        self.trajectory = []
        self.wrapped_env.reset()
        if len(self._init_positions) > 1:
            i = np.random.choice(len(self._init_positions), p=self._init_position_probs)
            self.wrapped_env.set_xy(self._init_positions[i])
        return self._get_obs()

    @property
    def viewer(self):
        return self.wrapped_env.viewer

    def render(self, *args, **kwargs):
        return self.wrapped_env.render(*args, **kwargs)

    @property
    def observation_space(self):
        shape = self._get_obs().shape
        high = np.inf * np.ones(shape)
        low = -high
        return gym.spaces.Box(low, high)

    @property
    def Ndim_repr(self):
        pos = self.wrapped_env.get_xy()
        shape = self._get_gp_repr(pos, pos).shape
        assert len(shape) == 1
        return shape[0]

    @property
    def action_space(self):
        return self.wrapped_env.action_space

    def _find_robot(self):
        return self._find("r")

    def _find_goal(self):
        return self._find("g")

    def _find(self, symb):
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == symb:
                    return j * size_scaling, i * size_scaling
        return None, None

    def _find_all_robots(self):
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        coords = []
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == "r":
                    coords.append((j * size_scaling, i * size_scaling))
        return coords

    def step(self, action):
        self.t += 1
        old_pos = self.wrapped_env.get_xy()
        inner_next_obs, inner_reward, done, info = self.wrapped_env.step(action)
        if self._manual_collision:
            grid_x, grid_y = self._get_grid_xy()
            if self.MAZE_STRUCTURE[grid_y][grid_x] == 1:
                self.wrapped_env.set_xy(old_pos)
        else:
            inner_next_obs, inner_reward, done, info = self.wrapped_env.step(action)
        next_obs = self._get_obs()
        done = False

        new_pos = self.wrapped_env.get_xy()
        gp_repr = self._get_gp_repr(old_pos, new_pos)

        info["state"] = None
        info["gp_repr"] = gp_repr
        info["x"] = new_pos[0]
        info["y"] = new_pos[1]

        grid_x, grid_y = self._get_grid_xy()
        info["grid_x"] = grid_x
        info["grid_y"] = grid_y
        reward = np.dot(self.reward_w, gp_repr)

        # print("gp_repr", info["gp_repr"])
        # print("reward_w", self.reward_w)
        # print("reward", reward)

        return next_obs, reward, done, info

    def sample_features_rewards(self, n_samples):
        n_dim = self.Ndim_repr
        l = 3
        n_ind = n_dim // l
        min_val = np.array([-3, -3, 1] * (n_dim // l))
        max_val = np.array([3, 3, 1] * (n_dim // l))

        samples = min_val + (max_val - min_val) * np.random.sample((n_samples, n_dim))

        samples = maze_env_utils.get_random_free_cell(self.MAZE_STRUCTURE, n_samples)
        samples = [(x, 0) for x, y in samples]

        # infos = [{"x": x, "y": y} for x, y in samples]

        ind = np.array(
            [
                np.ravel_multi_index((y, x), (self.height - 2, self.width))
                for x, y in samples
            ]
        )
        ind = np.stack([l * ind + i for i in range(l)], axis=1)
        mask = np.zeros((n_samples, n_dim))
        np.put_along_axis(mask, ind, 1, axis=1)
        samples = samples * mask
        return samples, np.matmul(samples, self.reward_w.T)

    def plot_potential(self):
        assert self._potential_based_reward
        import matplotlib.pyplot as plt

        n = 100
        x = np.linspace(0, self.width, n)
        y = np.linspace(0, self.height, n)
        xx, yy = np.meshgrid(x, y)
        potential = np.zeros_like(xx)
        for k, potential_function in enumerate(self.potential_functions):
            for i in range(n):
                for j in range(n):
                    x, y = xx[i, j] - 0.5, yy[i, j] - 0.5
                    potential[i, j] += self.reward_w[k] * potential_function(x, y)
        plt.imshow(potential, extent=[0, self.width, 0, self.height], origin="lower")
        plt.colorbar()
