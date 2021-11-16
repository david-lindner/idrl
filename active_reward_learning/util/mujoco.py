import numpy as np


def initialize_mujoco_from_com(env, q, v):
    """
    Initialize a given mujoco environment to a state conditioned on the center of mass
    position and velocities.

    Missing information in the observation is filled with zeros.
    """
    assert len(q.shape) == 1
    assert len(v.shape) == 1

    nq = env.model.nq
    qpos = np.zeros(nq)
    qpos[: q.shape[0]] = q

    nv = env.model.nv
    qvel = np.zeros(nv)
    qvel[: v.shape[0]] = v

    env.set_state(qpos, qvel)
    return env


def evaluate_mujoco_policy_from_s0(env, policy, q, v, N):
    res = 0
    for _ in range(N):
        obs = env.reset()
        env = initialize_mujoco_from_com(env, q, v)
        # FIXME: should get correct observation here
        done = False
        while not done:
            a = policy.get_action(obs)
            obs, reward, done, _ = env.step(a)
            res += reward
    return res / N
