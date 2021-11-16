import numpy as np

from active_reward_learning.envs import LimitedActionChain
from active_reward_learning.reward_models import BasicGPRewardModel
from active_reward_learning.reward_models.acquisition_functions import (
    acquisition_function_random,
)
from active_reward_learning.reward_models.kernels import RBFCustomDist
from active_reward_learning.reward_models.query import TrajectoryQuery
from active_reward_learning.solvers import LPSolver


def _get_not_observed_for_seed(seed):
    rewards = np.zeros(100)
    env = LimitedActionChain(rewards, 0.99, 100, 1)
    kernel = RBFCustomDist(input_dim=1)
    solver = LPSolver(env)
    np.random.seed(seed)
    reward_model = BasicGPRewardModel(
        env,
        acquisition_function_random,
        kernel,
        solver,
        initialize_candidate_policies=False,
    )
    not_observed = [s for s in range(env.N_states) if s not in reward_model.observed]
    return not_observed


def test_seed_initial_observation():
    seeds = [1, 42, 23, 555]
    N = 3
    for seed in seeds:
        last = _get_not_observed_for_seed(seed)
        for _ in range(N):
            next = _get_not_observed_for_seed(seed)
            assert last == next
            last = next


def test_trajectory_observations():
    # make sure sums are consistent
    np.random.seed(1)

    for _ in range(10):
        rewards = np.random.random(5)
        env = LimitedActionChain(rewards, 0.99, 100, 1)
        kernel = RBFCustomDist(input_dim=1)
        solver = LPSolver(env)
        reward_model = BasicGPRewardModel(
            env,
            acquisition_function_random,
            kernel,
            solver,
            use_comparisons=True,
            initialize_candidate_policies=False,
        )
        X = list(reward_model.gp_model.X_list)
        observed = []

        states1 = np.arange(3)
        states2 = np.arange(len(rewards))

        traj1 = [[i] for i in states1]
        traj2 = [[i] for i in states2]
        query1 = TrajectoryQuery(traj1, rewards[states1], dict())
        query2 = TrajectoryQuery(traj2, rewards[states2], dict())
        reward_model.update_reward_model(query1)
        reward_model.update_reward_model(query2)

        states3 = [i for i in states2 if i not in states1]
        traj3 = [[i] for i in states3]
        mu, cov = reward_model.gp_model.predict_multiple(
            [traj3], linear_combination=[[1] * len(traj3)]
        )
        assert np.isclose(mu[0], rewards[states3].sum())
        assert np.isclose(cov[0], 0)


def test_comparison_observations():
    # learns correct order of states
    rewards = np.arange(5) * 0.2
    env = LimitedActionChain(rewards, 0.99, 100, 1)
    kernel = RBFCustomDist(input_dim=1)
    solver = LPSolver(env)
    np.random.seed(1)
    reward_model = BasicGPRewardModel(
        env,
        acquisition_function_random,
        kernel,
        solver,
        use_comparisons=True,
        initialize_candidate_policies=False,
    )
    X = list(reward_model.gp_model.X_list)
    observed = []
    for query in reward_model.candidate_queries:
        for _ in range(50):
            # need multiple samples to learn correct order
            reward_model.update_reward_model(query)
    mu, cov = reward_model.gp_model.predict_multiple(np.arange(5))
    for i in range(1, 5):
        assert mu[i] > mu[i - 1]


def test_candidate_queries_gp_repr():
    """
    `get_candidate_queries_gp_repr` should return `gp_repr` and `linear_combination`
    such that it can be used with `predict_all` to make predictions for all the
    candidate queries.
    """
    rewards = np.zeros(10)
    env = LimitedActionChain(rewards, 0.99, 10, 1)
    kernel = RBFCustomDist(input_dim=1)
    solver = LPSolver(env)
    np.random.seed(1)

    for use_comparisons in (False, True):
        reward_model = BasicGPRewardModel(
            env,
            acquisition_function_random,
            kernel,
            solver,
            use_comparisons=use_comparisons,
            initialize_candidate_policies=False,
        )

        (
            gp_repr,
            linear_combination,
            gp_repr_idx,
        ) = reward_model.get_candidate_queries_gp_repr()

        mu_pred1, cov_pred1 = reward_model.gp_model.predict_multiple(
            gp_repr, linear_combination=linear_combination
        )
        sigma_pred1 = np.diagonal(cov_pred1)

        mu_pred2, sigma_pred2 = reward_model.get_candidate_queries_reward_predictions()

        assert np.allclose(mu_pred1, mu_pred2, atol=1e-3)
        assert np.allclose(sigma_pred1, sigma_pred2, atol=1e-3)
