import numpy as np

from .point import PointEnv


class PointEnvLinear(PointEnv):
    def __init__(self):
        self.Ndim_repr = 4
        self.reward_w = np.array([0, 0, 1, 0])
        super().__init__(expose_all_qpos=True, random_orientation=True)

    def _get_gp_repr(self, last_qpos, qpos):
        x, y = qpos[0], qpos[1]
        dx, dy = qpos[0] - last_qpos[0], qpos[1] - last_qpos[1]
        return np.array([x, y, dx, dy])

    def step(self, action):
        last_qpos = np.copy(self.physics.data.qpos)
        next_obs, reward, done, info = super().step(action)
        qpos = np.copy(self.physics.data.qpos)
        gp_repr = self._get_gp_repr(last_qpos, qpos)
        info["gp_repr"] = gp_repr
        reward = np.dot(self.reward_w, gp_repr)
        return next_obs, reward, done, info
