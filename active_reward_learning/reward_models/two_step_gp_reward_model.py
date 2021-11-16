import itertools
from typing import Any, Callable, Collection, Dict, List

import numpy as np

from active_reward_learning.envs import TabularMDP
from active_reward_learning.reward_models.basic_gp_reward_model import (
    BasicGPRewardModel,
)
from active_reward_learning.reward_models.query import ComparisonQueryLinear
from active_reward_learning.solvers import BaseSolver
from active_reward_learning.util.helpers import (
    get_dict_assert,
    get_dict_default,
    pdf_multivariate_gauss,
)


def get_policy_W(gp_reward_model: BasicGPRewardModel, policy_i: int):
    """
    Get state visitation frequencies of a single policy.
    """
    assert gp_reward_model.candidate_policies is not None
    policy = gp_reward_model.candidate_policies[policy_i]

    # we just need the covariance of rewards that are supported in W
    if gp_reward_model.use_trajectories_to_evaluate_policy:
        freq = gp_reward_model.state_visitation_frequencies[policy_i]
        all_states = [
            np.fromstring(k, dtype=gp_reward_model.state_repr_dtype)
            for k in freq.keys()
        ]
        W = np.array([freq[repr.tostring()] for repr in all_states])
    else:
        assert gp_reward_model.environment_is_tabular
        W = gp_reward_model.env.get_return_trafo_for_policy(policy)
        all_states = gp_reward_model.env.get_all_states_repr()

    support = W != 0
    states_support = [s for i, s in enumerate(all_states) if support[i]]
    return W, support, states_support, all_states


def get_multiple_policies_W(
    gp_reward_model: BasicGPRewardModel, candidate_policy_indices: Collection[int]
):
    """
    Get state visitation frequencies of multiple policies.
    """
    assert candidate_policy_indices is not None
    assert len(candidate_policy_indices) >= 2
    W_list = []
    states_in_W_repr_list = []
    all_states = set()
    for policy_i in candidate_policy_indices:
        W, support, states_support, states_in_W = get_policy_W(
            gp_reward_model, policy_i
        )
        states_in_W_repr = [state_repr.tostring() for state_repr in states_in_W]
        all_states.update(states_in_W_repr)
        W_list.append(W)
        states_in_W_repr_list.append(states_in_W_repr)

    N_states = len(all_states)
    all_states_idx = dict(zip(list(all_states), range(N_states)))
    W_list_new = []
    support_any = np.zeros(N_states, dtype=np.bool)
    states_support = [None] * N_states
    for W, states_in_W_repr in zip(W_list, states_in_W_repr_list):
        W_new = np.zeros(N_states)
        for W_val, state_repr in zip(W, states_in_W_repr):
            if W_val > 0:
                i = all_states_idx[state_repr]
                W_new[i] = W_val
                if not support_any[i]:
                    support_any[i] = True
                    states_support[i] = np.fromstring(
                        state_repr, dtype=gp_reward_model.state_repr_dtype
                    )
        W_list_new.append(W_new)

    states_support = [s for s in states_support if s is not None]

    return W_list_new, support_any, states_support


class TwoStepGPRewardModel(BasicGPRewardModel):
    """
    Implements a GP reward model with a two-step acquisition function.

    Points to query are selected by:
        1. selecting a policy to learn about
        2. select a point that is informative about the policy
    Steps 1 and 2 can be customized by specifying the `policy_selection_function`
    and the `state_selection_function`.
    """

    def __init__(
        self,
        env: TabularMDP,
        kernel_function,
        solver: BaseSolver,
        policy_selection_function: Callable[
            [List[np.ndarray], BasicGPRewardModel, Dict[str, Any]], List[int]
        ],
        state_selection_function: Callable[
            [np.ndarray, BasicGPRewardModel, Dict[str, Any]], int
        ],
        obs_var: float = 0,
        arguments: Dict[str, Any] = {},
        **kwargs,
    ):
        acquisition_function = self.get_acquisition_function(
            policy_selection_function, state_selection_function, arguments
        )
        super().__init__(
            env, acquisition_function, kernel_function, solver, obs_var, **kwargs
        )

    def get_acquisition_function(
        self,
        policy_selection_function: Callable[
            [List[np.ndarray], BasicGPRewardModel, Dict[str, Any]], List[int]
        ],
        state_selection_function: Callable[
            [np.ndarray, BasicGPRewardModel, Dict[str, Any]], int
        ],
        arguments: Dict[str, Any],
    ) -> Callable[[BasicGPRewardModel], int]:
        """
        Return an acquisition function that first selects a policy / set of policies
        according to the `policy_selection_function` and then a state to query according
        to the `state_selection_function`. The acquisition function can then simply
        be used with a `BasicGPRewardModel`.
        """

        def acquisition_function(gp_reward_model: BasicGPRewardModel) -> int:
            assert gp_reward_model.candidate_policies is not None
            policies_idx = policy_selection_function(
                gp_reward_model.candidate_policies, gp_reward_model, arguments
            )
            state = state_selection_function(policies_idx, gp_reward_model, arguments)
            return state

        return acquisition_function


def policy_selection_none(
    candidate_policies: List[np.ndarray],
    gp_reward_model: BasicGPRewardModel,
    arguments: Dict[str, Any] = {},
) -> List[int]:
    return list(range(len(candidate_policies)))


def policy_selection_maximum_regret(
    candidate_policies: List[np.ndarray],
    gp_reward_model: BasicGPRewardModel,
    arguments: Dict[str, Any] = {},
) -> List[int]:
    """
    Implementation of [1].

    [1] Wilde, Nils, Dana Kulic, and Stephen L. Smith.
        "Active preference learning using maximum regret."
        https://arxiv.org/pdf/2005.04067.pdf
    """
    assert gp_reward_model.candidate_rewards is not None
    assert gp_reward_model.use_comparisons
    if gp_reward_model.environment_is_tabular:
        raise NotImplementedError(
            "Maximum regret acquisition is not implemented for tabular environments"
        )

    simple_model = get_dict_default(arguments, "simple_model", False)
    gp = gp_reward_model.gp_model

    if simple_model:
        uncertainty_p = get_dict_assert(arguments, "uncertainty_p")
        assert 0.5 < uncertainty_p < 1, uncertainty_p
        reward_probs = np.ones(len(gp_reward_model.candidate_rewards))
        for x, y in zip(gp.X_list[1:], gp.Y_list[1:]):  # first one is grounding
            assert y == 1 or y == -1
            for i, reward_w in enumerate(gp_reward_model.candidate_rewards):
                features_1, features_2 = x
                reward_1 = np.dot(features_1, reward_w)
                reward_2 = np.dot(features_2, reward_w)
                if (reward_1 > reward_2 and y == 1) or (
                    reward_1 <= reward_2 and y == -1
                ):
                    reward_probs[i] *= uncertainty_p
                else:
                    reward_probs[i] *= 1 - uncertainty_p
        reward_probs /= np.sum(reward_probs)
    else:
        mu = gp.linear_predictive_mean
        cov = gp.linear_predictive_cov
        reward_probs = []
        for reward_w in gp_reward_model.candidate_rewards:
            reward_prob = pdf_multivariate_gauss(reward_w, mu, cov)
            reward_probs.append(reward_prob)

    max_reg = -float("inf")
    best_ij = [0, 0]

    for query in gp_reward_model.candidate_queries:
        assert isinstance(query, ComparisonQueryLinear)

        i, j = query.info["policy_i1"], query.info["policy_i2"]
        features_i, features_j = query.gp_repr_list

        reward_i = gp_reward_model.candidate_rewards[i]
        reward_j = gp_reward_model.candidate_rewards[j]

        p_i = reward_probs[i]
        p_j = reward_probs[j]

        G_pi_i_w_i = np.dot(features_i, reward_i)
        G_pi_j_w_j = np.dot(features_j, reward_j)
        G_pi_i_w_j = np.dot(features_i, reward_j)
        G_pi_j_w_i = np.dot(features_j, reward_i)

        # old implementation (wrong)
        # regret = - p_i * p_j * (G_pi_i_w_j / G_pi_j_w_j + G_pi_j_w_i / G_pi_i_w_i)
        # ratio based regret
        # regret = p_i * p_j * (2 - G_pi_i_w_j / G_pi_j_w_j - G_pi_j_w_i / G_pi_i_w_i)

        # difference based
        R1 = max(G_pi_j_w_j - G_pi_i_w_j, 0)
        R2 = max(G_pi_i_w_i - G_pi_j_w_i, 0)
        regret = p_i * p_j * (R1 + R2)

        # prints for debugging
        print(f"i: {i}   j: {j}")
        print(
            f"\tG_pi_i_w_j: {G_pi_i_w_j:.2f}   G_pi_j_w_j: {G_pi_j_w_j:.2f}   "
            f"G_pi_j_w_i: {G_pi_j_w_i:.2f}   G_pi_i_w_i: {G_pi_i_w_i:.2f}   "
        )
        print(f"\tR1: {R1:.2f}   R2: {R2:.2f}   p_i: {p_i}   p_j: {p_j}")
        print("\tregret", regret)
        ###

        if regret > max_reg:
            max_reg = regret
            best_ij = [i, j]

    print("max_reg", max_reg)
    print("best_ij", best_ij)

    return best_ij


def policy_selection_most_uncertain_pair_of_plausible_maximizers(
    candidate_policies: List[np.ndarray],
    gp_reward_model: BasicGPRewardModel,
    arguments: Dict[str, Any] = {},
) -> List[int]:
    """
    Selects two plausible maximizers that define the most uncertain direction.

    First determines the set of plausible maximizer policies, by comparing their
    confidence bounds. Then compares each pair of policies from the set of plausible
    maximizers to find the pair that has the highest variance in the difference of their
    expected returns.
    """
    n_policies = get_dict_default(arguments, "n_policies", 2)
    assert n_policies == 2

    if gp_reward_model.use_trajectories_to_evaluate_policy is not None:
        W_list, support_all, states_support = get_multiple_policies_W(
            gp_reward_model, list(range(len(candidate_policies)))
        )
        mu_support, sigma_support = gp_reward_model.gp_model.predict_multiple(
            states_support
        )
    else:
        raise NotImplementedError()

    # don't determine plausible maximizers
    plausible_maximizers = policy_indices = np.arange(len(candidate_policies))

    max_ij = [0, 1]
    max_var = -float("inf")
    for i, j in itertools.combinations(range(len(plausible_maximizers)), 2):
        policy_i1, policy_i2 = policy_indices[i], policy_indices[j]
        W_1, W_2 = W_list[policy_i1], W_list[policy_i2]
        G_pi_diff_var = np.dot(
            W_1[support_all] - W_2[support_all],
            np.dot(sigma_support, W_1[support_all] - W_2[support_all]),
        )

        if G_pi_diff_var > max_var:
            max_ij = [policy_i1, policy_i2]
            max_var = G_pi_diff_var

    # if tuple(max_ij) == (0, 1023):
    #     import ipdb; ipdb.set_trace()

    print("max_ij", max_ij)
    return max_ij


def state_selection_MI_diff(
    policy_idx: List[int],
    gp_reward_model: BasicGPRewardModel,
    arguments: Dict[str, Any] = {},
) -> int:
    """
    Select a state to query to maximize the mutual information between the states
    reward function and the difference between the expected returns of the two
    selected policies from the first step.

    Note that maximizing I(G^\\pi, (s, r(s))) is equivalent to minimizing
    H(G^\\pi | r(s)) (because H(G^\\pi, r(s)) is constant). Hence, maximizing
    mutual information is approximated by 'hallucinating' reward observations
    for each state and then finding the state that minimizes the conditional entropy.
    """
    assert gp_reward_model.candidate_policies is not None
    assert len(policy_idx) == 2
    policy_i1 = policy_idx[0]
    policy_i2 = policy_idx[1]

    (W_1, W_2), support, states_support = get_multiple_policies_W(
        gp_reward_model, (policy_i1, policy_i2)
    )

    min_var = float("inf")
    min_var_states = [0]

    (
        candidate_queries_gp_repr,
        candidate_queries_linear_combinations,
        candidate_queries_gp_repr_idx,
    ) = gp_reward_model.get_candidate_queries_gp_repr()

    for i in range(len(candidate_queries_gp_repr)):
        gp_repr = candidate_queries_gp_repr[i]
        linear_combination = candidate_queries_linear_combinations[i]
        query = (gp_repr, linear_combination)

        (
            _,
            sigma_support,
        ) = gp_reward_model.gp_model.make_temporary_observation_and_predict(
            query, 0, states_support
        )

        var = np.dot(
            W_1[support] - W_2[support],
            np.dot(sigma_support, W_1[support] - W_2[support]),
        )
        idx = candidate_queries_gp_repr_idx[i]

        if var < min_var:
            min_var = var
            min_var_states = [idx]
        elif var == min_var:
            min_var_states.append(idx)

    return np.random.choice(min_var_states)


def state_selection_MI(
    policy_idx: List[int],
    gp_reward_model: BasicGPRewardModel,
    arguments: Dict[str, Any] = {},
) -> int:
    """
    Select a state to query to maximize the mutual information between the states
    reward function and the expected return of the policy selected in the first step.

    Note that maximizing I(G^\\pi, (s, r(s))) is equivalent to minimizing
    H(G^\\pi | r(s)) (because H(G^\\pi, r(s)) is constant). Hence, maximizing
    mutual information is approximated by 'hallucinating' reward observations
    for each state and then finding the state that minimizes the conditional entropy.
    """
    assert gp_reward_model.candidate_policies is not None
    assert len(policy_idx) == 1
    policy_i1 = policy_idx[0]

    W_1, support, states_support, _ = get_policy_W(gp_reward_model, policy_i1)

    min_H_cond = float("inf")
    min_H_cond_states = [0]

    (
        candidate_queries_gp_repr,
        candidate_queries_linear_combinations,
        candidate_queries_gp_repr_idx,
    ) = gp_reward_model.get_candidate_queries_gp_repr()
    print("len(candidate_queries_gp_repr)", len(candidate_queries_gp_repr))

    for i in range(len(candidate_queries_gp_repr)):
        gp_repr = candidate_queries_gp_repr[i]
        linear_combination = candidate_queries_linear_combinations[i]
        query = (gp_repr, linear_combination)

        (
            _,
            sigma_support,
        ) = gp_reward_model.gp_model.make_temporary_observation_and_predict(
            query, 0, states_support
        )

        var = np.dot(
            W_1[support],
            np.dot(sigma_support, W_1[support]),
        )

        # H_cond = 0.5 * np.log(2 * np.pi * np.e * var)
        H_cond = var

        i = candidate_queries_gp_repr_idx[i]

        if H_cond < min_H_cond:
            min_H_cond = H_cond
            min_H_cond_states = [i]
        elif H_cond == min_H_cond:
            min_H_cond_states.append(i)

    return np.random.choice(min_H_cond_states)


def query_selection_policy_idx(
    policy_idx: List[int],
    gp_reward_model: BasicGPRewardModel,
    arguments: Dict[str, Any] = {},
) -> int:
    """
    Assumes that info dict contains which policies the trajectories are from.

    Hence, this currently only works with generating queries form Thompson sampled
    policies in the highway environment.
    """
    assert gp_reward_model.candidate_policies is not None
    assert len(policy_idx) == 2
    assert gp_reward_model.n_rollouts_for_states == 1
    policy_i1, policy_i2 = policy_idx
    idx = set()
    for i, query in enumerate(gp_reward_model.candidate_queries):
        if "policy_i1" not in query.info or "policy_i2" not in query.info:
            raise NotImplementedError()

        if (
            policy_i1 == query.info["policy_i1"]
            and policy_i2 == query.info["policy_i2"]
        ) or (
            policy_i1 == query.info["policy_i2"]
            and policy_i2 == query.info["policy_i1"]
        ):
            idx.add(i)

    return np.random.choice(list(idx))
