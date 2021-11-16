import gym
import numpy as np


class GymInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        lshape = len(env.observation_space.shape)
        assert lshape <= 1
        if lshape == 0:
            self.Ndim_repr = 1
        else:
            self.Ndim_repr = env.observation_space.shape[0]

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["state"] = None
        info["gp_repr"] = np.atleast_1d(np.array(obs))
        return obs, reward, done, info


class MujocoDirectionWrapper(gym.Wrapper):
    def __init__(self, env, direction=[1, 0]):
        self.direction = direction
        super().__init__(env)

    def step(self, a):
        xposbefore = self.sim.data.qpos[0]
        yposbefore = self.sim.data.qpos[1]
        obs, _, done, info = self.env.step(a)
        xposafter = self.sim.data.qpos[0]
        yposafter = self.sim.data.qpos[1]

        ctrl_cost_coeff = 0.0001
        reward_fwd = (
            self.direction[0] * (xposafter - xposbefore)
            + self.direction[1] * (yposafter - yposbefore)
        ) / self.dt
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()

        reward = reward_fwd + reward_ctrl
        return obs, reward, done, info
