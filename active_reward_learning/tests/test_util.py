import numpy as np

from active_reward_learning.util.helpers import (
    argmax_over_index_set,
    get_deterministic_policy_matrix,
    jaccard_index,
    mean_jaccard_distance,
)


def test_argmax_over_index_set():
    l = [0, 1, 1.5, 2, 3, 3.5, 4]
    idx = [0, 1, 2]
    assert argmax_over_index_set(l, idx) == [2]
    assert argmax_over_index_set(l, set(idx)) == [2]
    idx = [1, 2, 4, 6]
    assert argmax_over_index_set(l, idx) == [6]
    assert argmax_over_index_set(l, set(idx)) == [6]


def test_get_deterministic_policy_matrix():
    policy = get_deterministic_policy_matrix([0, 0, 3, 4, 2], 5)
    target_policy = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
        ]
    )
    assert np.all(policy == target_policy)
    policy = get_deterministic_policy_matrix(np.array([0, 1, 1, 4, 2]), 5)
    target_policy = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
        ]
    )
    assert np.all(policy == target_policy)


def test_jaccard_index():
    A = {1, 2, 3, 4, 7}
    B = {1, 4, 5, 7, 9}
    assert jaccard_index(A, B) == 3 / 7


def test_mean_jaccard_distance():
    A = {1, 2, 3, 4, 7}
    B = {1, 4, 5, 7, 9}
    C = {1, 3, 7, 8}
    assert (
        mean_jaccard_distance([A, B, C])
        == ((1 - 3 / 7) + (1 - 2 / 7) + (1 - 3 / 6)) / 3
    )

    assert mean_jaccard_distance([A]) == 0
