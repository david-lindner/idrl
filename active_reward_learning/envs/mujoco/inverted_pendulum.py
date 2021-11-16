import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

FULL_FILE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "assets", "inverted_pendulum.xml"
)


class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, penalty=False):
        self.penalty = penalty
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, FULL_FILE_PATH, 2)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        if self.penalty:
            done = False
            reward -= np.square(ob[1])
        else:
            notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
            done = not notdone

        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
