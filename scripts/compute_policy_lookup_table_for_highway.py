import pickle

import gym
import numpy as np

import active_reward_learning


def main():
    env = gym.make("HighwayDriving-RandomReward-v0")

    N = 1000
    n_restarts = 10
    outfile = "highway_policies.npy"

    all_policies = []

    for i in range(N):
        print("\ni", i)

        reward_w = np.random.multivariate_normal(np.zeros(4), np.identity(4))
        reward_w = np.array(list(reward_w) + [1])

        policy = env.get_optimal_policy(w=reward_w, restarts=n_restarts)
        all_policies.append(policy)

    with open(outfile, "wb") as f:
        pickle.dump(all_policies, f)


if __name__ == "__main__":
    main()
