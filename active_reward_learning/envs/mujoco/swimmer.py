"""Adapted from https://github.com/rll/rllab."""

import os

import mujoco_py
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

FULL_FILE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "assets", "swimmer.xml"
)


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    FILE = "swimmer.xml"

    def __init__(self, expose_all_qpos=True, file_path=FULL_FILE_PATH):
        self._expose_all_qpos = expose_all_qpos
        frame_skip = 4
        # frame_skip = 10
        mujoco_env.MujocoEnv.__init__(self, file_path, frame_skip)
        utils.EzPickle.__init__(self)

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= "1.50":
            return self.sim
        else:
            return self.model

    def step(self, a):
        # a[0] *= 10
        # a[1] *= 10
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        if self._expose_all_qpos:
            return np.concatenate([qpos.flat, qvel.flat])
        else:
            return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )
        return self._get_obs()

    def set_xy(self, xy):
        qpos = np.copy(self.physics.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]
        qvel = self.physics.data.qvel
        self.set_state(qpos, qvel)

    def get_xy(self):
        return np.copy(self.physics.data.qpos[:2])
