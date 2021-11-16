import os
import pickle
import time
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import scipy.sparse as sp

from active_reward_learning.common.policy import (
    BasePolicy,
    CombinedPolicy,
    EpsGreedyPolicy,
    GaussianNoisePolicy,
    LinearPolicy,
)
from active_reward_learning.envs import RewardModelMeanWrapper, TabularMDP
from active_reward_learning.envs.reward_model_sample_wrapper import (
    RewardModelSampleWrapper,
)
from active_reward_learning.reward_models.gaussian_process_linear import (
    LinearObservationGP,
)
from active_reward_learning.reward_models.kernels import LinearKernel
from active_reward_learning.reward_models.query import (
    ComparisonQueryLinear,
    LinearQuery,
    PointQuery,
    QueryBase,
    StateComparisonQueryLinear,
    StateQuery,
    TrajectoryQuery,
)
from active_reward_learning.solvers import (
    ArgmaxSolver,
    BaseSolver,
    LBFGSArgmaxSolver,
    LPSolver,
)
from active_reward_learning.util.helpers import (
    get_hash,
    mean_jaccard_distance,
    np_to_tuple,
    subsample_sequence,
    timing,
)


class BasicGPRewardModel:
    """
    Implements the basic GP reward modelling framework.

    The general procedure is:
        (i) Select state to query with acquisition function
        (ii) Query state and observe the reward
        (iii) Update GP model with observation
        (iv) Update policy to be optimal w.r.t new reward predictions from GP

    Attributes
    -------------
    env (gym.Env): environment
    gp_model (GP): gaussian process model of the reward function
    observed (set): states that have already been observed
    acquisition_function (callable): function that takes a BasicGPRewardModel
                                     object as an argument and returns the next
                                     state to query according to some
                                     acquisition function
    _last_pred_mu (np.ndarray): last cached means from GP reward prediction
    _last_pred_var (np.ndarray): last cached variances from GP reward prediction
    _last_predictions_up_to_date (bool): if True, the cached values are still
                                         up to date
    """

    def __init__(
        self,
        env: gym.Env,
        acquisition_function: Callable[["BasicGPRewardModel"], int],
        kernel,
        solver: BaseSolver,
        obs_var: float = 0,
        use_trajectories_to_evaluate_policy: bool = False,
        solver_iterations: int = 100,
        optimize_gp_parameters_every_n: Optional[int] = None,
        use_thompson_sampling_for_candidate_policies: bool = True,
        update_candidate_policies_every_n: Optional[int] = 10,
        n_candidate_policies_thompson_sampling: Optional[int] = 10,
        n_rollouts_for_states: int = 1,
        n_rollouts_for_eval: int = 1,
        candidate_queries_from: str = "fixed",
        initialize_candidate_policies: bool = True,
        use_mean_for_candidate_policies: bool = False,
        gp_num_inducing: Optional[int] = None,
        rollout_horizon: Optional[int] = None,
        subsampling_candidate_queries_n: Optional[int] = None,
        use_comparisons: bool = False,
        comparison_response: str = "bernoulli",
        subsample_traj_for_queries_len: Optional[int] = None,
        n_policies_initial_ts: Optional[int] = None,
        candidate_queries_file: Optional[Union[str, list]] = None,
        trajectory_clip_length: Optional[int] = None,
        trajectory_n_clips: Optional[int] = None,
        af_label: Optional[str] = None,
        observation_batch_size: int = 1,
    ):
        self.env = env
        self.environment_is_tabular = isinstance(env, TabularMDP) or isinstance(
            env.unwrapped, TabularMDP
        )
        self.use_tabular_solver = isinstance(solver, LPSolver) or (
            self.environment_is_tabular and isinstance(solver, ArgmaxSolver)
        )

        # initial observation
        env.reset()
        self.env.step(self.env.action_space.sample())
        obs, reward1, done, info1 = self.env.step(self.env.action_space.sample())

        self.use_comparisons = use_comparisons
        self.comparison_response = comparison_response

        self.gp_model = LinearObservationGP(kernel, obs_var=obs_var)

        if self.use_comparisons:
            # ground model in simulated observation at 0
            self.gp_model.observe((info1["gp_repr"], [1]), 0, obs_noise=0.001)

        if self.environment_is_tabular:
            self.last_query = info1["state"]
        else:
            self.last_query = None

        self.acquisition_function = acquisition_function

        self.observed: List[QueryBase] = []
        self.observed_dict: list = []
        self.observed_counter: Dict[QueryBase, int] = Counter()

        self._last_pred_mu = None
        self._last_pred_var = None
        self._last_pred_cov = None
        self._last_predictions_up_to_date = False
        self.selected_policy = None

        self.rollout_horizon = rollout_horizon
        self.use_trajectories_to_evaluate_policy = use_trajectories_to_evaluate_policy
        self.optimize_gp_parameters_every_n = optimize_gp_parameters_every_n
        self.use_thompson_sampling_for_candidate_policies = (
            use_thompson_sampling_for_candidate_policies
        )
        self.use_mean_for_candidate_policies = use_mean_for_candidate_policies
        self.update_candidate_policies_every_n = update_candidate_policies_every_n
        self.n_candidate_policies_thompson_sampling = (
            n_candidate_policies_thompson_sampling
        )

        self.candidate_queries_from = candidate_queries_from
        self.n_rollouts_for_states = n_rollouts_for_states
        self.n_rollouts_for_eval = n_rollouts_for_eval

        self.subsampling_candidate_queries_n = subsampling_candidate_queries_n
        self.subsample_traj_for_queries_len = subsample_traj_for_queries_len

        assert observation_batch_size >= 1
        self.observation_batch_size = observation_batch_size
        self.run_time = 0

        self.candidate_policies: Optional[List[np.ndarray]]
        self.candidate_rewards: Optional[List[np.ndarray]]
        self.candidate_policy_posterior_probabilities: Optional[List[float]]
        self.candidate_policy_mean_jaccard_dist: Optional[float] = None

        if initialize_candidate_policies:
            self.candidate_policies = env.get_candidate_policies()
            assert self.candidate_policies is not None
            assert len(self.candidate_policies) > 0
            n_policies = len(self.candidate_policies)
            self.candidate_policy_posterior_probabilities = [
                1 / n_policies
            ] * n_policies
            self.updated_candidate_policies_in_last_step = True
        else:
            self.candidate_policies = None
            self.candidate_policy_posterior_probabilities = None
            self.updated_candidate_policies_in_last_step = False

        self.solver = solver
        if not self.use_tabular_solver:
            if isinstance(kernel, LinearKernel):
                self.solver.set_env(RewardModelSampleWrapper(env, self))
            else:
                assert not self.use_thompson_sampling_for_candidate_policies
                self.solver.set_env(RewardModelMeanWrapper(env, self))
        self.solver_iterations = solver_iterations

        self.candidate_queries: list = []
        self.candidate_queries_gp_repr: Optional[List[Tuple]] = None
        self.candidate_queries_gp_repr_idx: Optional[List[int]] = None
        self.candidate_queries_linear_combinations = None
        self.state_visitation_frequencies: List[np.ndarray] = []
        self.state_repr_dtype = info1["gp_repr"].dtype
        if self.candidate_queries_from == "fixed":
            unique_repr = set()
            if self.environment_is_tabular:
                if use_comparisons:
                    for s1 in range(env.N_states):
                        for s2 in range(s1 + 1, env.N_states):
                            gp_repr1 = env.get_state_repr(s1)
                            gp_repr2 = env.get_state_repr(s2)
                            if list(gp_repr1) != list(gp_repr2):
                                gp_repr = tuple(list(gp_repr1) + list(gp_repr2))

                                if gp_repr not in unique_repr:
                                    unique_repr.add(gp_repr)
                                    self.candidate_queries.append(
                                        StateComparisonQueryLinear(
                                            gp_repr1,
                                            gp_repr2,
                                            env.get_reward(s1),
                                            env.get_reward(s2),
                                            dict(),
                                            response=comparison_response,
                                        )
                                    )
                else:
                    for s in range(env.N_states):
                        gp_repr = tuple(env.get_state_repr(s))
                        if gp_repr not in unique_repr:
                            unique_repr.add(gp_repr)
                            self.candidate_queries.append(
                                StateQuery(
                                    s,
                                    gp_repr,
                                    env.get_reward(s),
                                    dict(),
                                    obs=env.get_observation(s),
                                )
                            )
            else:
                # uniformly sample candidates for non-tabular environments
                n_samples = self.n_rollouts_for_states * self.env.spec.max_episode_steps
                x_test, y_test = self.env.sample_features_rewards(n_samples)
                if self.use_comparisons:
                    for i in range(len(x_test)):
                        x1 = x_test[i]
                        r1 = y_test[i]
                        for j in range(i + 1, len(x_test)):
                            x2 = x_test[j]
                            r2 = y_test[j]
                            info: Dict[object, object] = dict()
                            self.candidate_queries.append(
                                ComparisonQueryLinear(
                                    x1, x2, r1, r2, info, response=comparison_response
                                )
                            )
                else:
                    for i in range(len(x_test)):
                        x = x_test[i]
                        r = y_test[i]
                        info = dict()
                        self.candidate_queries.append(PointQuery(x, r, info))
        elif self.candidate_queries_from == "rollouts_fixed":
            if self.candidate_policies is None:
                assert (self.n_candidate_policies_thompson_sampling is not None) or (
                    n_policies_initial_ts is not None
                )
                self.candidate_policies = []
                self.candidate_rewards = []
                self.candidate_policy_posterior_probabilities = []
                print("\tinitial thompson sampling")
                if n_policies_initial_ts is None:
                    assert self.n_candidate_policies_thompson_sampling is not None
                    n_policies = self.n_candidate_policies_thompson_sampling
                else:
                    n_policies = n_policies_initial_ts
                self.update_candidate_policies_using_thompson_sampling(n_policies)

            self.collect_candidate_policy_rollouts(True)

            candidate_policies_features = [None] * len(self.candidate_policies)
            for query in self.candidate_queries:
                i, j = query.info["policy_i1"], query.info["policy_i2"]
                features_i, features_j = query.gp_repr_list
                if candidate_policies_features[i] is None:
                    candidate_policies_features[i] = features_i
                if candidate_policies_features[j] is None:
                    candidate_policies_features[j] = features_j

            assert self.candidate_policies is not None
            assert self.candidate_rewards is not None
            assert self.candidate_policy_posterior_probabilities is not None

            for i in range(len(self.candidate_policies)):
                for j in range(len(self.candidate_policies)):
                    reward_i = self.candidate_rewards[i]
                    features_i = candidate_policies_features[i]
                    features_j = candidate_policies_features[j]
                    G_pi_i_w_i = np.dot(features_i, reward_i)
                    G_pi_j_w_i = np.dot(features_j, reward_i)
                    if G_pi_j_w_i > G_pi_i_w_i:
                        self.candidate_rewards[i] = self.candidate_rewards[j]
                        self.candidate_policies[i] = self.candidate_policies[j]
                        candidate_policies_features[i] = candidate_policies_features[j]
                        self.candidate_policy_posterior_probabilities[
                            i
                        ] = self.candidate_policy_posterior_probabilities[j]

            self.collect_candidate_policy_rollouts(True)

            if "Highway" in self.env.spec.id:
                print("Update LBFGSArgmaxSolver...", end=" ")

                if isinstance(self.solver, LBFGSArgmaxSolver):
                    print("Appending candidate policies.")
                    cand = self.solver.candidate_policies
                    cand = cand + self.candidate_policies
                else:
                    print("Using candidate policies.")
                    cand = self.candidate_policies

                self.solver = LBFGSArgmaxSolver(
                    self.solver.env,
                    candidate_policies=cand,
                    debug=False,
                )

        elif self.candidate_queries_from == "query_file":
            assert isinstance(candidate_queries_file, str)
            assert candidate_queries_file is not None
            assert candidate_queries_file.endswith(".pkl")
            print(f"Loading candidate queries from '{candidate_queries_file}'...")
            with open(candidate_queries_file, "rb") as f:
                self.candidate_queries = pickle.load(f)
        elif self.candidate_queries_from in ("random_rollouts", "policy_file"):
            assert trajectory_clip_length is not None
            assert trajectory_n_clips is not None

            expl_policies: List[BasePolicy]
            if self.candidate_queries_from == "random_rollouts":
                print(f"Randomly exploring...")
                policy: BasePolicy
                policy = LinearPolicy(np.zeros(2))  # will never be used
                policy = EpsGreedyPolicy(policy, 1, env.action_space)
                expl_policies = [policy]
            else:
                assert candidate_queries_file is not None
                assert isinstance(candidate_queries_file, (str, list))
                if isinstance(candidate_queries_file, str):
                    candidate_queries_file = [candidate_queries_file]
                expl_policies = []
                for policy_file in candidate_queries_file:
                    assert policy_file.endswith(".npy")
                    print(f"Loading exploration policy from '{policy_file}'...")
                    policy = LinearPolicy.load(policy_file, env)
                    eps = 0.1
                    policy = EpsGreedyPolicy(policy, eps, env.action_space)
                    expl_policies.append(policy)

            policy_idx = 0

            trajectories = []
            for i_rollout in range(self.n_rollouts_for_states):
                obs = self.env.reset()
                done = False
                t = 0
                gp_repr_list, reward_list, info_list = [], [], []

                print(policy_idx)
                expl_policy = expl_policies[policy_idx]
                policy_idx = (policy_idx + 1) % len(expl_policies)

                while not done and (
                    self.rollout_horizon is None or t <= self.rollout_horizon
                ):
                    t += 1
                    a = expl_policy.get_action(obs)
                    obs, reward, done, info = self.env.step(a)
                    gp_repr_list.append(info["gp_repr"])
                    reward_list.append(reward)
                    info_list.append({k: v for k, v in info.items() if k != "gp_repr"})
                trajectories.append((gp_repr_list, reward_list, info_list))

            for gp_repr_list, reward_list, info_list in trajectories:
                info = {"info_list": tuple(info_list)}
                L = trajectory_clip_length
                N = trajectory_n_clips
                for _ in range(N):
                    start = np.random.randint(0, len(gp_repr_list) - L)
                    self.candidate_queries.append(
                        TrajectoryQuery(
                            gp_repr_list[start : start + L],
                            reward_list[start : start + L],
                            info,
                        )
                    )

        self.timing: Dict[str, float] = dict()

    def run(self, iterations, callback=None, print_timing=False):
        iteration = 0

        self.run_time = 0
        while iteration < iterations:
            current_time = time.time()
            print("\tquery x")

            if (
                (
                    self.use_thompson_sampling_for_candidate_policies
                    or self.use_mean_for_candidate_policies
                )
                and self.update_candidate_policies_every_n is not None
                and iteration % self.update_candidate_policies_every_n == 0
            ):
                self.update_candidate_policies()
                self.updated_candidate_policies_in_last_step = True
                update_candidate_queries = (
                    self.candidate_queries_from == "rollouts_updated"
                )
            else:
                self.updated_candidate_policies_in_last_step = False
                update_candidate_queries = False

            if (
                update_candidate_queries or self.use_trajectories_to_evaluate_policy
            ) and self.candidate_policies is not None:
                print("\trolling out policies")
                self.collect_candidate_policy_rollouts(update_candidate_queries)

            self.query_reward(iteration, iterations)
            iteration += 1

            if (
                self.optimize_gp_parameters_every_n is not None
                and self.optimize_gp_parameters_every_n > 0
                and iteration % self.optimize_gp_parameters_every_n == 0
            ):
                self.gp_model.optimize_parameters()

            new_current_time = time.time()
            self.run_time += new_current_time - current_time
            current_time = new_current_time

            if callback:
                callback(locals(), globals())

            if print_timing:
                print("\t\tTiming")
                for key, value in self.timing.items():
                    print("\t\t\t'{}': {:.2f} seconds".format(key, value))

    @timing
    def update_candidate_policies(self):
        self.candidate_policies = []
        self.candidate_rewards = []
        self.candidate_policy_posterior_probabilities = []
        if self.use_thompson_sampling_for_candidate_policies:
            print("\tthompson sampling")
            self.update_candidate_policies_using_thompson_sampling()
        if self.use_mean_for_candidate_policies:
            print("\tget new mean-optimal candidate policy")
            self.update_candidate_policies_using_mean()

        print("candidate_policies", self.candidate_policies)
        print("posterior_probabilities", self.candidate_policy_posterior_probabilities)

    @timing
    def update_reward_model(self, query: QueryBase, y) -> None:
        """
        Update the GP reward model with a specific observation.

        Args:
        -----------
        query (QueryBase): query at which the reward observation was made
        optimize_gp (bool): if set to true the parameters of the GP will be
                            optimize by max log-likelihood
        """
        self.observed.append(query)
        self.observed_dict.append((dict(query._asdict()), y))
        self.observed_counter[query] += 1
        assert isinstance(self.gp_model, LinearObservationGP)
        assert isinstance(query, LinearQuery)
        print("query.gp_repr_list", query.gp_repr_list)
        print("query.linear_combination", query.linear_combination)
        self.gp_model.observe([query.gp_repr_list, query.linear_combination], y)
        self._last_predictions_up_to_date = False

    @timing
    def query_reward(self, iteration, iterations) -> None:
        """
        Query the reward of a query determined by the acquistion function and update
        the GP reward model.
        """
        print("\trunning acquisition function")

        for sample_i in range(self.observation_batch_size):
            print(sample_i, end=" ")
            i = self.acquisition_function(self)
            query = self.candidate_queries[i]
            y = query.reward

            self.update_reward_model(query, y)
        self.last_query = query

    def get_candidate_queries_gp_repr(self):
        """
        Return the features and linear weights for every candidate query.

        If subsampling is activated returns them for a random subsample of candidates.
        For this case a list of indices is returned that maps the subsampled queries
        to positions in the full list of candidate queries.
        """
        if (
            self.candidate_queries_gp_repr is None
            or self.candidate_queries_gp_repr_idx is None
            or self.candidate_queries_linear_combinations is None
        ) or self.subsampling_candidate_queries_n is not None:
            sample_idx = np.arange(len(self.candidate_queries))
            if self.subsampling_candidate_queries_n is not None:
                # sample a subset of candidate queries
                sample_idx = np.random.choice(
                    sample_idx,
                    self.subsampling_candidate_queries_n,
                )

            # dictionary to contain unique set of queries
            candidate_queries_dict = dict()
            for i in sample_idx:
                query = self.candidate_queries[i]
                gp_repr_list = query.gp_repr_list
                linear_combination = query.linear_combination
                gp_repr_tuple = tuple(
                    [np_to_tuple(gp_repr) for gp_repr in gp_repr_list]
                )
                linear_combination_tuple = tuple(linear_combination)
                key = (gp_repr_tuple, linear_combination_tuple)

                if key not in candidate_queries_dict:
                    candidate_queries_dict[key] = i

            # convert unique set of queries into list
            candidate_queries_gp_repr = []
            candidate_queries_linear_combinations = []
            candidate_queries_gp_repr_idx = []

            for key, idx in candidate_queries_dict.items():
                gp_repr_tuple, linear_combination_tuple = key
                gp_repr_list = [list(x) for x in gp_repr_tuple]
                linear_combination = list(linear_combination_tuple)
                candidate_queries_gp_repr.append(gp_repr_list)
                candidate_queries_linear_combinations.append(linear_combination)
                candidate_queries_gp_repr_idx.append(idx)

            if self.subsampling_candidate_queries_n is not None:
                return (
                    candidate_queries_gp_repr,
                    candidate_queries_linear_combinations,
                    candidate_queries_gp_repr_idx,
                )
            else:
                self.candidate_queries_gp_repr = candidate_queries_gp_repr
                self.candidate_queries_linear_combinations = (
                    candidate_queries_linear_combinations
                )
                self.candidate_queries_gp_repr_idx = candidate_queries_gp_repr_idx
        return (
            self.candidate_queries_gp_repr,
            self.candidate_queries_linear_combinations,
            self.candidate_queries_gp_repr_idx,
        )

    @timing
    def get_candidate_queries_reward_predictions(
        self, get_full_cov: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get mean and variance of reward predictions for the candidate states from the GP.

        Uses caching to avoid querying the GP model if no update happened since the last call
        fo this function.

        Note: currently this removes duplicates, which is not obvious.
        """
        if (
            self._last_predictions_up_to_date
            and self.subsampling_candidate_queries_n is None
        ):
            if get_full_cov:
                return self._last_pred_mu, self._last_pred_cov
            else:
                return self._last_pred_mu, self._last_pred_var
        else:
            (
                candidate_query_repr,
                candidate_queries_linear_combinations,
                candidate_queries_gp_repr_idx,
            ) = self.get_candidate_queries_gp_repr()

            mu_pred, cov_pred = self.gp_model.predict_multiple(
                candidate_query_repr,
                linear_combination=candidate_queries_linear_combinations,
            )
            var_pred = np.diag(cov_pred)

            self._last_pred_mu = mu_pred
            self._last_pred_cov = cov_pred
            self._last_pred_var = var_pred
            self._last_predictions_up_to_date = True
            if get_full_cov:
                return mu_pred, cov_pred
            else:
                return mu_pred, var_pred

    @timing
    def get_policy_return_mean_var(
        self,
        policy_i: int,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Returns mean and variance of a policy given the current GP reward model.

        Assuming the reward is Gaussian, the expected return of a policy is also
        Gaussian. The mean and variance of this gaussian are returned by this method.
        """
        assert self.candidate_policies is not None
        assert 0 <= policy_i < len(self.candidate_policies)
        if self.use_trajectories_to_evaluate_policy:
            freq = self.state_visitation_frequencies[policy_i]
            state_repr = [
                np.fromstring(k, dtype=self.state_repr_dtype) for k in freq.keys()
            ]
            W = np.array([freq[repr.tostring()] for repr in state_repr])
            mu, sigma = self.gp_model.predict_multiple(state_repr)
            E_G_pi = np.dot(W, mu)
            Var_G_pi = np.dot(W, np.dot(sigma, W))
        else:
            assert self.environment_is_tabular
            policy = self.candidate_policies[policy_i]
            W = self.env.get_return_trafo_for_policy(policy)
            all_states = self.env.get_all_states_repr()
            mu, sigma = self.gp_model.predict_multiple(all_states)
            E_G_pi = W.dot(mu)
            Var_G_pi = W.dot(sp.csr_matrix(sigma).dot(W))
        return E_G_pi, Var_G_pi

    @timing
    def update_candidate_policies_using_thompson_sampling(
        self,
        n_policies=None,
    ) -> None:
        """
        Uses Thompson sampling to get a new set of candidate policies based on
        the current reward model.

        Performs two steps n_policies times:
        1. Sample a reward function from the model
        2. Finds the optimal policy for this function
        """
        assert self.candidate_policies is not None
        assert self.candidate_rewards is not None
        assert self.candidate_policy_posterior_probabilities is not None

        if n_policies is None:
            assert self.n_candidate_policies_thompson_sampling is not None
            n_policies = self.n_candidate_policies_thompson_sampling

        if self.use_tabular_solver:
            all_states = np.asarray(self.env.get_all_states_repr())
            samples = self.gp_model.sample_y_from_posterior(all_states, n_policies)
            for i in range(n_policies):
                print("\tTS Policy (tabular) {}".format(i))
                rewards = samples[:, i]
                policy = self.solver.solve(self.solver_iterations, rewards=rewards)
                self.candidate_policies.append(policy)
                self.candidate_rewards.append(rewards)
        else:
            for i in range(n_policies):
                print("\tTS Policy {}".format(i))
                posterior_prob = self.solver.env.new_sample(self)

                policy = self.solver.solve(self.solver_iterations)
                self.candidate_policies.append(policy)
                self.candidate_rewards.append(self.solver.env.theta)
                self.candidate_policy_posterior_probabilities.append(posterior_prob)

        self._last_predictions_up_to_date = False
        print("\tresampled candidate policies")

    @timing
    def update_candidate_policies_using_mean(self) -> None:
        """
        Will add the current mean-optimal policy to the set of candidates.
        """
        assert self.candidate_policies is not None
        assert self.candidate_rewards is not None
        assert self.candidate_policy_posterior_probabilities is not None
        if self.use_tabular_solver:
            all_states = np.asarray(self.env.get_all_states_repr())
            mu_pred, _ = self.gp_model.predict_multiple(all_states)
            rewards = mu_pred
            policy = self.solver.solve(self.solver_iterations, rewards=mu_pred)
        else:
            if isinstance(self.solver.env, RewardModelSampleWrapper):
                posterior_prob = self.solver.env.set_sample_to_mean(self)
                self.candidate_policy_posterior_probabilities.append(posterior_prob)
                rewards = self.solver.env.theta
            elif isinstance(self.gp_model.kernel, LinearKernel):
                rewards = self.gp_model.linear_predictive_mean
            else:
                rewards = None
            policy = self.solver.solve(self.solver_iterations)
        self.candidate_policies.append(policy)
        self.candidate_rewards.append(rewards)
        self._last_predictions_up_to_date = False
        print("resampled mean candidate policy")

    @timing
    def collect_candidate_policy_rollouts(self, update_candidate_queries) -> None:
        """
        Performs `n_rollouts` for each policy in `candidate_policy` and updates
        `candidate_queries` to contain all distinct states that were visited
        during the rollouts.
        """
        assert self.candidate_policies is not None
        assert self.use_trajectories_to_evaluate_policy or update_candidate_queries

        if self.use_trajectories_to_evaluate_policy:
            self.state_visitation_frequencies = []

        if update_candidate_queries:
            self.candidate_queries = []
            self.candidate_queries_gp_repr = None
            self.candidate_queries_linear_combinations = None
            self.candidate_queries_gp_repr_idx = None

        if update_candidate_queries:
            n_rollouts = max(self.n_rollouts_for_eval, self.n_rollouts_for_states)
        else:
            n_rollouts = self.n_rollouts_for_eval

        state_action_per_policy = []
        trajectories = []
        for policy_i, policy in enumerate(self.candidate_policies):
            # print("policy_i", policy_i)

            state_actions = set()
            if self.use_trajectories_to_evaluate_policy:
                W_pi: Dict[str, float] = dict()
                count = 0
            for rollout_i in range(n_rollouts):
                # print("rollout_i", rollout_i)
                # print("rollout")
                obs = self.env.reset()
                done = False
                t = 0
                gp_repr_list, reward_list, info_list = [], [], []
                while not done and (
                    self.rollout_horizon is None or t <= self.rollout_horizon
                ):
                    t += 1
                    a = policy.get_action(obs)

                    state_actions.add((get_hash(obs), get_hash(a)))
                    obs, reward, done, info = self.env.step(a)

                    if update_candidate_queries:
                        gp_repr_list.append(info["gp_repr"])
                        reward_list.append(reward)
                        info_list.append(
                            {k: v for k, v in info.items() if k != "gp_repr"}
                        )
                    if self.use_trajectories_to_evaluate_policy:
                        # print(f'DEBUG: x {info["x"]},  y {info["y"]}')
                        gp_repr = info["gp_repr"].tostring()
                        if self.state_repr_dtype is None:
                            self.state_repr_dtype = info["gp_repr"].dtype
                        elif self.state_repr_dtype != info["gp_repr"].dtype:
                            raise Exception(
                                "Inconsistent datatypes in gp_representation: "
                                "{} and {}".format(
                                    self.state_repr_dtype, info["gp_repr"].dtype
                                )
                            )
                        if gp_repr not in W_pi:
                            W_pi[gp_repr] = 0
                        W_pi[gp_repr] += 1
                        count += 1

                if self.subsample_traj_for_queries_len is not None:
                    old_len = len(gp_repr_list)
                    new_len = self.subsample_traj_for_queries_len
                    a, b = subsample_sequence(old_len, new_len)
                    assert len(gp_repr_list) == old_len
                    assert len(reward_list) == old_len
                    assert len(info_list) == old_len
                    gp_repr_list = gp_repr_list[a:b]
                    reward_list = reward_list[a:b]
                    info_list = info_list[a:b]
                    assert len(gp_repr_list) == new_len
                    assert len(reward_list) == new_len
                    assert len(info_list) == new_len

                trajectories.append((gp_repr_list, reward_list, info_list, policy_i))

            state_action_per_policy.append(state_actions)

            if self.use_trajectories_to_evaluate_policy:
                for k in W_pi.keys():
                    W_pi[k] /= n_rollouts
                self.state_visitation_frequencies.append(W_pi)

        if update_candidate_queries:
            if self.use_comparisons:
                # Note: this currently only works for linear reward functions and has only
                # been tested for HighwayDriving.
                if not "Highway" in self.env.spec.id:
                    raise NotImplementedError(
                        "Comparisons from Thompson sampled trajectories "
                        "is currently only implemented for highway environment."
                    )

                ## TODO: make this a proper parameter
                normalize_features = False

                if not normalize_features:
                    # normalize rewards
                    rewards = []
                    for i in range(len(trajectories)):
                        gp_repr_list1, _, _, _ = trajectories[i]
                        gp_repr1 = np.sum(gp_repr_list1, axis=0) / len(gp_repr_list1)
                        reward1 = np.dot(gp_repr1, self.env.reward_w)
                        rewards.append(reward1)
                    max_reward = np.max(rewards)
                    min_reward = np.min(rewards)

                for i in range(len(trajectories)):
                    gp_repr_list1, reward_list1, info_list1, policy_i1 = trajectories[i]
                    gp_repr1 = np.sum(gp_repr_list1, axis=0) / len(gp_repr_list1)

                    # Normalize features to ensure rewards are between 0 and 1
                    if normalize_features:
                        gp_repr1[-1] = 0
                        gp_repr1 /= np.linalg.norm(gp_repr1)
                        gp_repr1[-1] = 1

                    reward1 = np.dot(gp_repr1, self.env.reward_w)

                    if not normalize_features:
                        reward1 = (reward1 - min_reward) / (max_reward - min_reward)

                    for j in range(i + 1, len(trajectories)):
                        (
                            gp_repr_list2,
                            reward_list2,
                            info_list2,
                            policy_i2,
                        ) = trajectories[j]
                        gp_repr2 = np.sum(gp_repr_list2, axis=0) / len(gp_repr_list2)

                        if normalize_features:
                            gp_repr2[-1] = 0
                            gp_repr2 /= np.linalg.norm(gp_repr2)
                            gp_repr2[-1] = 1

                        reward2 = np.dot(gp_repr2, self.env.reward_w)

                        if not normalize_features:
                            reward2 = (reward2 - min_reward) / (max_reward - min_reward)

                        info = {
                            "info_list1": tuple(info_list1),
                            "info_list2": tuple(info_list2),
                            "policy_i1": policy_i1,
                            "policy_i2": policy_i2,
                        }
                        self.candidate_queries.append(
                            ComparisonQueryLinear(
                                gp_repr1,
                                gp_repr2,
                                reward1,
                                reward2,
                                info,
                                response=self.comparison_response,
                            )
                        )
            else:
                for gp_repr_list, reward_list, info_list, policy_i in trajectories:
                    info = {"info_list": tuple(info_list)}
                    self.candidate_queries.append(
                        TrajectoryQuery(
                            gp_repr_list,
                            reward_list,
                            info,
                        )
                    )

        jac_dist = mean_jaccard_distance(state_action_per_policy)
        self.candidate_policy_mean_jaccard_dist = jac_dist
