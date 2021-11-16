import cvxopt
import numpy as np

from active_reward_learning.common.policy import TabularPolicy
from active_reward_learning.envs import LimitedActionChain, TabularMDP
from active_reward_learning.util.helpers import get_deterministic_policy_matrix


def test_solver_consistency():
    np.random.seed(1)
    for _ in range(10):
        rewards = np.random.random(size=10)

        env = LimitedActionChain(rewards, 0.99, 10, 1, use_sparse_transitions=False)
        optimal_value_1 = env.get_lp_solution(return_value=True)
        optimal_policy_1 = env.get_lp_solution(return_value=False)
        optimal_value_2 = env.get_value_function(optimal_policy_1)
        optimal_policy_2 = env.get_greedy_policy_for_value_function(optimal_value_1)
        optimal_value_3 = env.get_value_function(optimal_policy_2)
        optimal_policy_3 = env.get_greedy_policy_for_value_function(optimal_value_2)

        optimal_value_4 = env.get_policy_iteration_solution(100, return_value=True)
        optimal_policy_4 = env.get_policy_iteration_solution(100, return_value=False)
        optimal_value_5 = env.get_value_function(optimal_policy_4)
        optimal_policy_5 = env.get_greedy_policy_for_value_function(optimal_value_4)
        optimal_value_6 = env.get_value_function(optimal_policy_5)
        optimal_policy_6 = env.get_greedy_policy_for_value_function(optimal_value_5)

        cvxopt.solvers.options["abstol"] = 0.000001
        cvxopt.solvers.options["feastol"] = 0.000001

        env_sp = LimitedActionChain(rewards, 0.99, 10, 1, use_sparse_transitions=True)
        env_sp.transitions = None
        sp_optimal_value_1 = env_sp.get_lp_solution(return_value=True)
        sp_optimal_policy_1 = env_sp.get_lp_solution(return_value=False)
        sp_optimal_value_2 = env_sp.get_value_function(optimal_policy_1)
        sp_optimal_policy_2 = env_sp.get_greedy_policy_for_value_function(
            optimal_value_1
        )
        sp_optimal_value_3 = env_sp.get_value_function(optimal_policy_2)
        sp_optimal_policy_3 = env_sp.get_greedy_policy_for_value_function(
            optimal_value_2
        )
        sp_optimal_value_4 = env_sp.get_policy_iteration_solution(
            100, return_value=True
        )
        sp_optimal_policy_4 = env_sp.get_policy_iteration_solution(
            100, return_value=False
        )
        sp_optimal_value_5 = env_sp.get_value_function(optimal_policy_4)
        sp_optimal_policy_5 = env_sp.get_greedy_policy_for_value_function(
            optimal_value_4
        )
        sp_optimal_value_6 = env_sp.get_value_function(optimal_policy_5)
        sp_optimal_policy_6 = env_sp.get_greedy_policy_for_value_function(
            optimal_value_5
        )

        assert sp_optimal_policy_1 == optimal_policy_1
        assert sp_optimal_policy_2 == optimal_policy_2
        assert sp_optimal_policy_3 == optimal_policy_3
        assert sp_optimal_policy_4 == optimal_policy_4
        assert sp_optimal_policy_5 == optimal_policy_5
        assert sp_optimal_policy_6 == optimal_policy_6

        assert np.allclose(sp_optimal_value_1, optimal_value_1, atol=1e-5)
        assert np.allclose(sp_optimal_value_2, optimal_value_2, atol=1e-5)
        assert np.allclose(sp_optimal_value_3, optimal_value_3, atol=1e-5)
        assert np.allclose(sp_optimal_value_4, optimal_value_4, atol=1e-5)
        assert np.allclose(sp_optimal_value_5, optimal_value_5, atol=1e-5)
        assert np.allclose(sp_optimal_value_6, optimal_value_6, atol=1e-5)

        assert np.allclose(sp_optimal_value_1, sp_optimal_value_2)
        assert np.allclose(sp_optimal_value_2, sp_optimal_value_3)
        assert np.allclose(sp_optimal_value_3, sp_optimal_value_4)
        assert np.allclose(sp_optimal_value_4, sp_optimal_value_5)
        assert np.allclose(sp_optimal_value_5, sp_optimal_value_6)

        assert optimal_policy_1 == optimal_policy_2
        assert optimal_policy_2 == optimal_policy_3
        assert optimal_policy_3 == optimal_policy_4
        assert optimal_policy_4 == optimal_policy_5
        assert optimal_policy_5 == optimal_policy_6

        assert np.allclose(optimal_value_1, optimal_value_2, atol=1e-5)
        assert np.allclose(optimal_value_2, optimal_value_3, atol=1e-5)
        assert np.allclose(optimal_value_3, optimal_value_4, atol=1e-5)
        assert np.allclose(optimal_value_4, optimal_value_5, atol=1e-5)
        assert np.allclose(optimal_value_5, optimal_value_6, atol=1e-5)

        assert np.allclose(
            env.evaluate_policy(optimal_policy_1),
            env.evaluate_policy(optimal_policy_2),
            atol=0.00001,
        )
        assert np.allclose(
            env.evaluate_policy(optimal_policy_2),
            env.evaluate_policy(optimal_policy_3),
            atol=0.00001,
        )
        assert np.allclose(
            env.evaluate_policy(optimal_policy_3),
            env.evaluate_policy(optimal_policy_4),
            atol=0.00001,
        )
        assert np.allclose(
            env.evaluate_policy(optimal_policy_4),
            env.evaluate_policy(optimal_policy_5),
            atol=0.00001,
        )
        assert np.allclose(
            env.evaluate_policy(optimal_policy_5),
            env.evaluate_policy(optimal_policy_6),
            atol=0.00001,
        )

        assert env.evaluate_policy(optimal_policy_1) >= env.evaluate_policy(
            TabularPolicy(np.zeros_like(optimal_policy_1.matrix))
        )
        assert env.evaluate_policy(optimal_policy_4) >= env.evaluate_policy(
            TabularPolicy(np.zeros_like(optimal_policy_1.matrix))
        )

        for i in range(env.N_states):
            one_hot = np.zeros(env.N_states, dtype=np.float32)
            one_hot[i] = 1
            assert np.allclose(
                env.evaluate_policy(optimal_policy_1, one_hot),
                optimal_value_1[i],
                atol=0.00001,
            )
            assert np.allclose(
                env.evaluate_policy(optimal_policy_2, one_hot),
                optimal_value_2[i],
                atol=0.00001,
            )
            assert np.allclose(
                env.evaluate_policy(optimal_policy_3, one_hot),
                optimal_value_3[i],
                atol=0.00001,
            )
            assert np.allclose(
                env.evaluate_policy(optimal_policy_4, one_hot),
                optimal_value_4[i],
                atol=0.00001,
            )
            assert np.allclose(
                env.evaluate_policy(optimal_policy_5, one_hot),
                optimal_value_5[i],
                atol=0.00001,
            )
            assert np.allclose(
                env.evaluate_policy(optimal_policy_6, one_hot),
                optimal_value_6[i],
                atol=0.00001,
            )


def test_solvers():
    T = np.zeros((2, 3, 3), dtype=float)
    T[0, :, :] = np.eye(3)
    T[1, [0, 1, 2], [1, 2, 0]] = [1, 1, 1]
    R = np.array([2, 10, 4], dtype=float)

    target_pi = get_deterministic_policy_matrix(np.array([1, 0, 1]), 2)
    target_V = np.array([100, 100, 92], dtype=float)

    env = TabularMDP(3, 2, R, T, 0.9, [], 10000, None)
    solution_pi = env.get_lp_solution(return_value=False)
    solution_V = env.get_lp_solution(return_value=True)
    assert np.allclose(solution_pi.matrix, target_pi)
    assert np.allclose(solution_V, target_V)

    env = TabularMDP(3, 2, R, T, 0.9, [], 10000, None, use_sparse_transitions=True)
    solution_pi = env.get_lp_solution(return_value=False)
    solution_V = env.get_lp_solution(return_value=True)
    assert np.allclose(solution_pi.matrix, target_pi)
    assert np.allclose(solution_V, target_V)

    env = TabularMDP(3, 2, R, T, 0.9, [], 10000, None)
    solution_pi = env.get_policy_iteration_solution(100, return_value=False)
    solution_V = env.get_policy_iteration_solution(100, return_value=True)
    assert np.allclose(solution_pi.matrix, target_pi)
    assert np.allclose(solution_V, target_V)

    env = TabularMDP(3, 2, R, T, 0.9, [], 10000, None, use_sparse_transitions=True)
    solution_pi = env.get_policy_iteration_solution(100, return_value=False)
    solution_V = env.get_policy_iteration_solution(100, return_value=True)
    assert np.allclose(solution_pi.matrix, target_pi)
    assert np.allclose(solution_V, target_V)
