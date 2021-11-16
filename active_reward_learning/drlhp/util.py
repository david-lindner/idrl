import numpy as np


def initialize_mujoco_from_obs(env, obs):
    """
    Initialize a given MuJoCo environment to a state conditioned on an observation.

    Missing information in the observation (which is usually the torso-coordinates)
    is filled with zeros.
    """
    env_id = env.unwrapped.spec.id
    if (
        "InvertedPendulum" in env_id
        or "InvertedDoublePendulum" in env_id
        or "Reacher" in env_id
        or "WithPos" in env_id
    ):
        nfill = 0
    elif "HalfCheetah" in env_id or "Hopper" in env_id or "Walker2d" in env_id:
        nfill = 1
    elif "Ant" in env_id or "Swimmer" in env_id:
        nfill = 2
    else:
        raise NotImplementedError(f"{env_id} not supported")

    nq = env.model.nq
    nv = env.model.nv
    if "InvertedDoublePendulum" in env_id:
        q0 = obs[0]
        s = obs[1 : nq + 1]
        c = obs[nq + 1 : 2 * nq + 1]
        obs_qpos = np.arctan2(c, s)
        obs_qvel = obs[2 * nq + 1 : 2 * nq + nv + 1]
    elif "Reacher" in env_id:
        c = obs[:2]
        s = obs[2:4]
        theta = np.arctan2(c, s)
        obs_qpos = np.zeros(nq)
        obs_qpos[:2] = theta
        obs_qpos[2:] = obs[4 : nq + 4 - 2]
        obs_qvel = np.zeros(nv)
        obs_qvel[:2] = obs[nq + 4 - 2 : nq + 4]
    else:
        obs_qpos = np.zeros(nq)
        obs_qpos[nfill:] = obs[: nq - nfill]
        obs_qvel = obs[nq - nfill : nq - nfill + nv]

    env.set_state(obs_qpos, obs_qvel)
    return env


def render_mujoco_from_obs(env, obs, **kwargs):
    env = initialize_mujoco_from_obs(env, obs)
    rgb = env.render(mode="rgb_array", **kwargs)
    return rgb
