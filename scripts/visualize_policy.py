"""

Code to load a policy and generate rollout data. Adapted from https://github.com/berkeleydeeprlcourse.
Example usage:
    python run_policy.py ../trained_policies/Humanoid-v1/policy_reward_11600/lin_policy_plus.npz Humanoid-v1 --render \
            --num_rollouts 20
"""
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines import PPO2, SAC

from active_reward_learning.common.policy import EpsGreedyPolicy, GaussianNoisePolicy
from active_reward_learning.common.policy import LinearPolicy, StableBaselinesPolicy
from active_reward_learning.envs import RewardModelMeanWrapper
from active_reward_learning.util.results import FileExperimentResults


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("policy_file", type=str)
    parser.add_argument("policy_type", type=str)
    parser.add_argument("envname", type=str)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gp_model_file", type=str)
    parser.add_argument(
        "--num_rollouts", type=int, default=1, help="Number of expert rollouts"
    )
    parser.add_argument("--mujoco_render_trajectory", action="store_true")
    parser.add_argument("--plot_reward", action="store_true")
    args = parser.parse_args()

    print("loading and building expert policy")

    env = gym.make(args.envname)

    if args.policy_type == "ars":
        policy = LinearPolicy.load(args.policy_file, env)
        print("w", policy.w)
        print("mean", policy.obs_mean)
        print("std", policy.obs_std)
    elif args.policy_type == "ars_experiment":
        ex = FileExperimentResults(args.policy_file)
        w = np.array(ex.info["weights"][-1])
        m = np.array(ex.info["means"][-1])
        s = np.array(ex.info["stds"][-1])
        print("w", w)
        print("mean", m)
        print("std", s)
        policy = LinearPolicy(w, m, s, env)
    elif args.policy_type == "ppo":
        model = PPO2.load(args.policy_file)
        policy = StableBaselinesPolicy(model)
    elif args.policy_type == "sac":
        model = SAC.load(args.policy_file)
        policy = StableBaselinesPolicy(model)
    else:
        raise ValueError(f"unkown policy type {args.policy_type}")

    if args.gp_model_file is not None:
        env = RewardModelMeanWrapper.load_from_model(
            env, args.gp_model_file, n_rff_features=None, debug=True
        )

    env.reset()
    env.render()
    # env.viewer.cam.trackbodyid = 0
    # env.viewer.cam.fixedcamid = 0

    returns = []
    observations = []
    for i in range(args.num_rollouts):
        if args.mujoco_render_trajectory:
            x, y = [], []
        if args.plot_reward:
            true_rewards, inferred_rewards, env_rewards = [], [], []
        print("iter", i)
        obs = env.reset()
        done = False
        totalr = 0.0
        steps = 0
        while not done:
            action = policy.get_action(obs)
            observations.append(obs)

            obs, r, done, info = env.step(action)
            done = False
            print("obs", obs)
            print("action", action)
            print("reward", r)
            # print("info", info)

            if args.plot_reward:
                true_reward = info["true_reward"] if "true_reward" in info else r
                inferred_reward = (
                    info["inferred_reward"] if "inferred_reward" in info else r
                )
                true_rewards.append(true_reward)
                inferred_rewards.append(inferred_reward)
                env_rewards.append(r)

            if args.mujoco_render_trajectory:
                xpos = env.unwrapped.sim.data.qpos[0]
                ypos = env.unwrapped.sim.data.qpos[1]
                x.append(xpos)
                y.append(ypos)

            totalr += r
            steps += 1
            if args.render:
                env.render()
                # import pdb; pdb.set_trace()
            if steps % 1 == 0:
                print(f"{steps}/{env.spec.max_episode_steps}")
            if steps >= min(10000, env.spec.max_episode_steps):
                break
        print("return", totalr)
        returns.append(totalr)

        if args.plot_reward:
            print("True return", np.sum(true_rewards))
            print("Inferred return", np.sum(inferred_rewards))
            print("Env return", np.sum(env_rewards))
            plt.figure(figsize=(12, 8))
            plt.plot(true_rewards, label="true")
            plt.plot(env_rewards, label="inferred")
            plt.legend()
            plt.xlabel("t")
            plt.ylabel("reward")
            plt.show()

        if args.mujoco_render_trajectory:
            plt.figure(figsize=(10, 10))
            x, y = np.array(x), np.array(y)
            plt.plot(x, y)
            m = max(np.abs(x).max(), np.abs(y).max())
            m *= 1.1
            plt.xlim(-m, m)
            plt.ylim(-m, m)
            plt.show()

    print("returns", returns)
    print("mean return", np.mean(returns))
    print("std of return", np.std(returns))


if __name__ == "__main__":
    main()
