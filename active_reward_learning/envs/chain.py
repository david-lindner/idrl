from typing import Optional

import numpy as np

from active_reward_learning.envs.tabular_mdp import TabularMDP


class SimpleChain(TabularMDP):
    # x <-> x <-> x <-> x <-> x <-> x <-> x <-> x
    def __init__(
        self,
        rewards: np.ndarray,
        discount_factor: float,
        init_agent_pos: Optional[int],
        episode_length: int,
        observation_type: str = "state",
        observation_noise: float = 0,
    ):
        N = rewards.shape[0]
        transitions = np.zeros((2, N, N))
        for i in range(N):
            transitions[0, i, max(0, i - 1)] = 1
            transitions[1, i, min(N - 1, i + 1)] = 1
        super().__init__(
            N,
            2,
            rewards,
            transitions,
            discount_factor,
            [],
            episode_length,
            init_agent_pos,
            observation_type=observation_type,
            observation_noise=observation_noise,
        )


class LimitedActionChain(TabularMDP):
    # x -> x -> x -> x <-> x <-> x <-> x
    def __init__(
        self,
        rewards: np.ndarray,
        discount_factor: float,
        episode_length: int,
        block_N: int,
        init_agent_pos: Optional[int] = None,
        use_sparse_transitions: bool = False,
        observation_type: str = "state",
        observation_noise: float = 0,
    ):
        N = rewards.shape[0]
        transitions = np.zeros((2, N, N))
        for i in range(block_N):
            transitions[0, i, min(N - 1, i + 1)] = 1
            transitions[1, i, min(N - 1, i + 1)] = 1
        for i in range(block_N, N):
            transitions[0, i, max(0, i - 1)] = 1
            transitions[1, i, min(N - 1, i + 1)] = 1
        super().__init__(
            N,
            2,
            rewards,
            transitions,
            discount_factor,
            [],
            episode_length,
            init_agent_pos,
            use_sparse_transitions=use_sparse_transitions,
            observation_type=observation_type,
            observation_noise=observation_noise,
        )


class FirstStateTrapChain(TabularMDP):
    # x <-> x <-> x <-> x <-> x <-> x <-> x <-> x
    def __init__(
        self,
        rewards: np.ndarray,
        discount_factor: float,
        init_agent_pos: Optional[int],
        episode_length: int,
        prob_to_first: float,
        observation_type: str = "state",
        observation_noise: float = 0,
    ):
        N = rewards.shape[0]
        transitions = np.zeros((2, N, N))
        for i in range(N):
            transitions[0, i, max(0, i - 1)] += 1 - prob_to_first
            transitions[0, i, 0] += prob_to_first
            transitions[1, i, min(N - 1, i + 1)] += 1 - prob_to_first
            transitions[1, i, 0] += prob_to_first
        super().__init__(
            N,
            2,
            rewards,
            transitions,
            discount_factor,
            [],
            episode_length,
            init_agent_pos,
            observation_type=observation_type,
            observation_noise=observation_noise,
        )
