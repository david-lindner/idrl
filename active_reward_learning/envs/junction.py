from typing import Optional

import numpy as np

from active_reward_learning.envs.tabular_mdp import TabularMDP


class JunctionMDP(TabularMDP):
    #            -> x <-> x <-> x <-> x <-> x <-> x <-> x <-> x
    # x -> x -> |
    #            -> x <-> x <-> x <-> x <-> x <-> x <-> x <-> x
    def __init__(
        self,
        rewards_chain: np.ndarray,
        rewards_junction_top: np.ndarray,
        rewards_junction_bottom: np.ndarray,
        discount_factor: float,
        episode_length: int,
        init_agent_pos: Optional[int] = None,
        observation_type: str = "state",
        observation_noise: float = 0,
    ):
        # first N states are left chain (from 0 to N-1)
        # next M states are upper right chain (from N to N+M-1)
        # final M states are lower right chain (from N+M to N+2M-1)
        N = rewards_chain.shape[0]
        M = rewards_junction_top.shape[0]
        assert rewards_junction_bottom.shape[0] == M

        transitions = np.zeros((2, N + 2 * M, N + 2 * M))

        # first N states can only go right
        for i in range(N - 1):
            transitions[0, i, min(N - 1, i + 1)] = 1
            transitions[1, i, min(N - 1, i + 1)] = 1

        # at Nth state can go up or down
        transitions[0, N - 1, N] = 1
        transitions[1, N - 1, N + M] = 1

        # for both paths we then have random walks within the M states of the path
        for i in range(M):
            # upper path
            transitions[0, N + i, min(N + M - 1, N + i + 1)] += 0.5
            transitions[1, N + i, min(N + M - 1, N + i + 1)] += 0.5
            transitions[0, N + i, max(N, N + i - 1)] += 0.5
            transitions[1, N + i, max(N, N + i - 1)] += 0.5
            # lower path
            transitions[0, N + M + i, min(N + 2 * M - 1, N + M + i + 1)] += 0.5
            transitions[1, N + M + i, min(N + 2 * M - 1, N + M + i + 1)] += 0.5
            transitions[0, N + M + i, max(N + M, N + M + i - 1)] += 0.5
            transitions[1, N + M + i, max(N + M, N + M + i - 1)] += 0.5

        super().__init__(
            N + 2 * M,
            2,
            np.array(
                list(rewards_chain)
                + list(rewards_junction_top)
                + list(rewards_junction_bottom)
            ),
            transitions,
            discount_factor,
            [],
            episode_length,
            init_agent_pos,
            observation_type=observation_type,
            observation_noise=observation_noise,
        )
