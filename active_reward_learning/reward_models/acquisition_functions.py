"""
Collection of acquisition function which can be used for the `BasicGPRewardModel`.

All of the acquisition functions have the same type signature to increase modularity.
"""

from typing import List

import numpy as np
from scipy.stats import norm

from active_reward_learning.reward_models.basic_gp_reward_model import (
    BasicGPRewardModel,
)
from active_reward_learning.reward_models.query import LinearQuery, QueryBase
from active_reward_learning.solvers import LBFGSArgmaxSolver, LBFGSSolver
from active_reward_learning.util.helpers import argmax_over_index_set


def acquisition_function_random(gp_reward_model: BasicGPRewardModel) -> int:
    """
    Randomly select the states to query.
    """
    return np.random.randint(0, len(gp_reward_model.candidate_queries))


def acquisition_function_random_unobserved(gp_reward_model: BasicGPRewardModel) -> int:
    """
    Randomly select the states to query from all unobserved states.
    """
    candidate_queries: List[QueryBase] = []
    i = np.random.randint(0, len(gp_reward_model.candidate_queries))
    while gp_reward_model.observed_counter[gp_reward_model.candidate_queries[i]]:
        i = np.random.randint(0, len(gp_reward_model.candidate_queries))
    return i


def acquisition_function_variance(gp_reward_model: BasicGPRewardModel) -> int:
    """
    Maximize the variance of the GP prediction.
    """
    (
        candidate_queries_gp_repr,
        candidate_queries_linear_combination,
        candidate_queries_gp_repr_idx,
    ) = gp_reward_model.get_candidate_queries_gp_repr()
    # mu_pred, sigma_pred = gp_reward_model.get_candidate_queries_reward_predictions()
    mu_pred, cov_pred = gp_reward_model.gp_model.predict_multiple(
        candidate_queries_gp_repr,
        linear_combination=candidate_queries_linear_combination,
    )
    sigma_pred = np.diag(cov_pred)

    argmax_variance = argmax_over_index_set(
        sigma_pred, range(len(candidate_queries_gp_repr_idx))
    )
    return candidate_queries_gp_repr_idx[np.random.choice(argmax_variance)]


def acquisition_function_expected_volume_removal(
    gp_reward_model: BasicGPRewardModel,
) -> int:
    """
    Maximize expected volume removal as described in [1].

    [1] Sadigh, Dorsa, et al. "Active Preference-Based Learning of Reward Functions."
    Robotics: Science and Systems. 2017.
    """
    assert gp_reward_model.use_comparisons

    # DL: This assumes the same observation model for each query which we might
    # want to change at some point
    query0 = gp_reward_model.candidate_queries[0]
    response = query0.response

    (
        candidate_queries_gp_repr,
        candidate_queries_linear_combination,
        candidate_queries_gp_repr_idx,
    ) = gp_reward_model.get_candidate_queries_gp_repr()
    # mu_diff, _ = gp_reward_model.get_candidate_queries_reward_predictions()
    mu_diff, _ = gp_reward_model.gp_model.predict_multiple(
        candidate_queries_gp_repr,
        linear_combination=candidate_queries_linear_combination,
    )

    if response == "bernoulli":
        prob = (1 + np.clip(mu_diff, -1, 1)) / 2
    elif response == "deterministic":
        prob = np.sign(mu_diff)
    elif response == "probit":
        prob = norm.cdf(mu_diff / (np.sqrt(2) * query0.sigma))
    else:
        raise NotImplementedError(f"evr for {response}")

    volume_removal = np.minimum(1 - prob, prob)

    argmax_volume_removal = argmax_over_index_set(
        volume_removal, range(len(candidate_queries_gp_repr_idx))
    )
    return candidate_queries_gp_repr_idx[np.random.choice(argmax_volume_removal)]


def acquisition_function_probability_of_improvement(
    gp_reward_model: BasicGPRewardModel,
) -> int:
    """
    Probability of improvement (see, e.g., [1]):
    $$
        \\mathrm{PI}(x) = \\Phi \\left( \\frac{\\mu(x) - f(x^*) - \\xi}{\\sigma(x)} \\right)
    $$
    where $x^*$ is the best sample in the training set and $\\Phi$ is the normal
    cumulative density function. The exploration-exploitation tradeoff parameter
    $\\xi$ has to be chosen manually.

    [1] Daniel, Christian, et al. "Active reward learning with a novel acquisition function."
        Autonomous Robots 39.3 (2015): 389-405.
    """
    if not gp_reward_model.observed:
        return acquisition_function_random(gp_reward_model)
    xi = 0.001
    (
        candidate_queries_gp_repr,
        candidate_queries_linear_combination,
        candidate_queries_gp_repr_idx,
    ) = gp_reward_model.get_candidate_queries_gp_repr()
    # mu_pred, sigma_pred = gp_reward_model.get_candidate_queries_reward_predictions()
    mu_pred, cov_pred = gp_reward_model.gp_model.predict_multiple(
        candidate_queries_gp_repr,
        linear_combination=candidate_queries_linear_combination,
    )
    sigma_pred = np.diag(cov_pred)
    mu_max = -float("inf")
    for query in gp_reward_model.observed:
        assert isinstance(query, LinearQuery)
        obs = query.reward
        if obs > mu_max:
            mu_max = obs
    M = (mu_pred - mu_max - xi) / sigma_pred
    PI = norm.cdf(M)
    argmax_pi = argmax_over_index_set(PI, range(len(candidate_queries_gp_repr_idx)))
    return candidate_queries_gp_repr_idx[np.random.choice(argmax_pi)]


def acquisition_function_expected_improvement(
    gp_reward_model: BasicGPRewardModel,
) -> int:
    """
    Expected improvement (see, e.g., [1]):
    $$
        \\mathrm{EI}(x) = (\\mu(x) - f(x^*) - \\xi) \\Phi(M) + \\sigma(x) \\rho(M)
    $$
    if $\\sigma > 0$ and zero otherwise, where where $x^*$ is the best sample
    in the training set, $\\Phi$ is the normal cumulative density function and
    $\\rho(x)$ is the normal probability density function. The exploration-
    exploitation tradeoff parameter $\\xi$ has to be chosen manually. $M$ is
    given by
    $$
        M = \\frac{\\mu(x) - f(x^*) - \\xi}{\\sigma(x)}
    $$

    [1] Daniel, Christian, et al. "Active reward learning with a novel acquisition function."
        Autonomous Robots 39.3 (2015): 389-405.
    """
    if not gp_reward_model.observed:
        return acquisition_function_random(gp_reward_model)
    xi = 0.001
    (
        candidate_queries_gp_repr,
        candidate_queries_linear_combination,
        candidate_queries_gp_repr_idx,
    ) = gp_reward_model.get_candidate_queries_gp_repr()
    # mu_pred, sigma_pred = gp_reward_model.get_candidate_queries_reward_predictions()
    mu_pred, cov_pred = gp_reward_model.gp_model.predict_multiple(
        candidate_queries_gp_repr,
        linear_combination=candidate_queries_linear_combination,
    )
    sigma_pred = np.diag(cov_pred)

    mu_max = -float("inf")
    for query in gp_reward_model.observed:
        assert isinstance(query, LinearQuery)
        obs = query.reward
        if obs > mu_max:
            mu_max = obs
    M = (mu_pred - mu_max - xi) / sigma_pred
    EI = M * sigma_pred * norm.cdf(M) + sigma_pred * norm.pdf(M)
    argmax_ei = argmax_over_index_set(EI, range(len(candidate_queries_gp_repr_idx)))
    return candidate_queries_gp_repr_idx[np.random.choice(argmax_ei)]


def _get_reward_model_policy(gp_reward_model, temporary_observation=None):
    if (
        not gp_reward_model.use_tabular_solver
        and "HighwayDriving" in gp_reward_model.env.spec.id
    ):
        assert isinstance(gp_reward_model.solver, (LBFGSSolver, LBFGSArgmaxSolver))
        if temporary_observation is not None:
            w_pred, _ = gp_reward_model.gp_model.make_temporary_observation_and_predict(
                temporary_observation[0],
                temporary_observation[1],
                None,
                predictive_mean=True,
            )
        else:
            w_pred = gp_reward_model.gp_model.linear_predictive_mean
        policy = gp_reward_model.env.get_optimal_policy(
            w=w_pred, restarts=gp_reward_model.solver_iterations
        )
    elif gp_reward_model.use_tabular_solver:
        states = gp_reward_model.env.get_all_states_repr()
        if temporary_observation is not None:
            (
                mu_pred,
                _,
            ) = gp_reward_model.gp_model.make_temporary_observation_and_predict(
                temporary_observation[0], temporary_observation[1], states
            )
        else:
            mu_pred, _ = gp_reward_model.gp_model.predict_multiple(states)

        policy = gp_reward_model.solver.solve(
            gp_reward_model.solver_iterations, rewards=mu_pred
        )
    else:
        raise NotImplementedError("EPD only works for tabular environments and driving")
    return policy


def acquisition_function_expected_policy_divergence(
    gp_reward_model: BasicGPRewardModel,
) -> int:
    """
    Acquisition function for discrete MDPs that is analogous to the `expected
    policy divergence` approach introduced by [1].

    Simulates an observation at the upper (and lower) confidence bounds of the
    GP and then counds the states in which the policy changes according to this
    (simulated) observations. Ultimately the state is chosen that maximally
    changes the policy in this way.

    [1] maximize the KL-divergence of the last policy and the next policy. Here,
    we imitate this by measuring the difference by counting states.

    [1] Daniel, Christian, et al. "Active reward learning with a novel acquisition function."
        Autonomous Robots 39.3 (2015): 389-405.
    """
    (
        candidate_queries_gp_repr,
        candidate_queries_linear_combination,
        candidate_queries_gp_repr_idx,
    ) = gp_reward_model.get_candidate_queries_gp_repr()

    # mu_pred, sigma_pred = gp_reward_model.get_candidate_queries_reward_predictions()
    mu_pred, cov_pred = gp_reward_model.gp_model.predict_multiple(
        candidate_queries_gp_repr,
        linear_combination=candidate_queries_linear_combination,
    )
    sigma_pred = np.diag(cov_pred)

    if gp_reward_model.environment_is_tabular:

        def policy_distance(policy1, policy2):
            return np.sum(policy1.matrix != policy2.matrix)

    else:

        def policy_distance(policy1, policy2):
            return np.sum(np.square(policy1.matrix - policy2.matrix))

    max_diff = 0
    orig_policy = _get_reward_model_policy(gp_reward_model)
    next_x = [0]
    for i in range(len(candidate_queries_gp_repr)):
        gp_repr = candidate_queries_gp_repr[i]
        linear_combination = candidate_queries_linear_combination[i]
        obs = (gp_repr, linear_combination)

        # print(i)
        policy_upper = _get_reward_model_policy(
            gp_reward_model, temporary_observation=(obs, mu_pred[i] + sigma_pred[i])
        )

        diff_i = policy_distance(policy_upper, orig_policy)

        lower_bound = True
        if lower_bound:
            policy_lower = _get_reward_model_policy(
                gp_reward_model,
                temporary_observation=(obs, mu_pred[i] - sigma_pred[i]),
            )
            diff_i += policy_distance(policy_lower, orig_policy)

        if diff_i > max_diff:
            max_diff = diff_i
            next_x = [i]
        elif diff_i == max_diff:
            next_x.append(i)

    return candidate_queries_gp_repr_idx[np.random.choice(next_x)]


# dictionary to get acquisition functions from short labels
ACQUISITION_FUNCTIONS = {
    "rand": acquisition_function_random,
    "rand_unobs": acquisition_function_random_unobserved,
    "var": acquisition_function_variance,
    "pi": acquisition_function_probability_of_improvement,
    "ei": acquisition_function_expected_improvement,
    "epd": acquisition_function_expected_policy_divergence,
    "evr": acquisition_function_expected_volume_removal,
}
