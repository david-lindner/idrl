import datetime
import functools
import os
import pickle
import time
from typing import Callable, Optional

import gym
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver
from scipy.stats import pearsonr

from active_reward_learning.common.constants import BASE_PATH, LABELS, PLOTS_PATH
from active_reward_learning.common.policy import BasePolicy, LinearPolicy
from active_reward_learning.envs import Gridworld, RewardModelMeanWrapper, TabularMDP
from active_reward_learning.reward_models import (
    BasicGPRewardModel,
    TwoStepGPRewardModel,
    policy_selection_maximum_regret,
    policy_selection_most_uncertain_pair_of_plausible_maximizers,
    policy_selection_none,
    query_selection_policy_idx,
    state_selection_MI,
    state_selection_MI_diff,
)
from active_reward_learning.reward_models.acquisition_functions import (
    ACQUISITION_FUNCTIONS,
)
from active_reward_learning.reward_models.gaussian_process_linear import (
    draw_random_rewards_from_GP,
)
from active_reward_learning.reward_models.kernels import LinearKernel, RBFCustomDist
from active_reward_learning.reward_models.query import (
    ComparisonQueryLinear,
    TrajectoryQuery,
)
from active_reward_learning.solvers import (
    AugmentedRandomSearchSolver,
    BaseSolver,
    get_standard_solver,
)
from active_reward_learning.util.helpers import (
    gaussian_log_likelihood,
    get_dict_assert,
    get_dict_default,
    get_unique_x_y,
)
from active_reward_learning.util.plotting import plot_query_state
from active_reward_learning.util.results import Artifact
from active_reward_learning.util.video import record_gym_video


# changes the run _id and thereby the path that the FileStorageObserver
# writes the results
# cf. https://github.com/IDSIA/sacred/issues/174
class SetID(RunObserver):
    priority = 50  # very high priority to set id

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        seed = np.random.randint(100000, 999999)
        np.random.seed()  # reset seed to not get conflicts between runs with the same
        rand = np.random.randint(100000, 999999)
        np.random.seed(seed)
        custom_id = f"{timestamp}_{rand}"
        return custom_id  # started_event returns the _run._id


ex = Experiment("gym_env_experiment")
ex.observers = [SetID(), FileStorageObserver("results/gym_env_experiments")]

if len(ex.observers) == 0:
    print("---------------------------------")
    print("WARNING: No observer specified!")
    print("---------------------------------")


def log_metric(_run, iteration, label, value):
    print(label, value)
    _run.log_scalar(label, value, iteration)


def _get_reward_from_gp_repr(env, gp_repr):
    return np.dot(env.reward_w, gp_repr)


def junction_dist(N, M, s1, s2):
    if (s1 < N + M and s2 < N + M) or (s1 >= N + M and s2 >= N + M):
        return abs(s1 - s2) + 1
    else:
        if s1 > s2:
            s1, s2 = s2, s1
        return abs(N - 1 - s1) + abs(N + M - s2) + 2


def get_gridworld_dist(gridworld: Gridworld) -> Callable[[int, int], float]:
    def gridworld_dist(s1: int, s2: int) -> float:
        x1, y1 = gridworld._get_agent_pos_from_state(s1)
        x2, y2 = gridworld._get_agent_pos_from_state(s2)
        assert gridworld.tiles is not None
        t1 = gridworld.tiles[y1, x1]
        t2 = gridworld.tiles[y2, x2]
        return float(t1 != t2)

    return gridworld_dist


def abs_value_distance(x, y):
    return abs(x - y)


def get_kernel(mdp, ndim):
    if mdp["label"] == "gym":
        gym_label = mdp["gym_label"]
        if "Linear" in gym_label or "Maze" in gym_label:
            kernel = LinearKernel(ndim, variances=np.ones(ndim) * 3)
        elif "HighwayDriving" in gym_label:
            kernel = LinearKernel(ndim, variances=3 * np.ones(ndim))
        else:
            raise NotImplementedError()
    elif mdp["label"] in ("gridworld", "gridworld_point"):
        if mdp["label"] == "gridworld_point":
            kernel = RBFCustomDist(input_dim=ndim, variance=2, lengthscale=1)
        else:
            kernel = RBFCustomDist(input_dim=2, variance=2, lengthscale=1)
    elif mdp["label"] == "limited_chain":
        if mdp["observation_type"] == "state":
            kernel = RBFCustomDist(
                input_dim=ndim, variance=2, lengthscale=3, distance=abs_value_distance
            )
        elif mdp["observation_type"] == "raw":
            kernel = LinearKernel(input_dim=ndim, ARD=False)
        else:
            raise Exception(f"Unrecognized observation_type: '{observation_type}'")
    elif mdp["label"] in ("junction_ei"):
        N, M = mdp["N"], mdp["M"]
        if mdp["observation_type"] == "state":
            kernel = RBFCustomDist(
                input_dim=ndim,
                variance=2,
                lengthscale=3,
                distance=functools.partial(junction_dist, N, M),
            )
        else:
            raise Exception(f"Unrecognized observation_type: '{observation_type}'")
    elif mdp["label"] in ("junction_ordinal"):
        if mdp["observation_type"] == "state":
            kernel = RBFCustomDist(
                input_dim=ndim,
                variance=2,
                lengthscale=3,
                distance=functools.partial(junction_dist, 1, 2),
            )
        else:
            raise Exception(f"Unrecognized observation_type: '{observation_type}'")
    else:
        raise Exception("Unrecognized mdp label")
    return kernel


def run_active_learning(
    _run,
    env: gym.Env,
    reward_model: BasicGPRewardModel,
    evaluation_solver: BaseSolver,
    stop_after: int,
    label: str = "test",
    plot_query: bool = False,
    plot_policy: bool = False,
    print_timing: bool = True,
    fix_candidate_policy_to_optimal: bool = False,
    evaluate_policy: bool = True,
    n_episodes_for_policy_eval: int = 10,
    save_models_path: Optional[str] = None,
    evaluation_policy_path: Optional[str] = None,
    n_rollouts_for_states: int = 1,
    get_evaluation_set_from: str = "uniform",
    env_gym_label: str = "",
    use_comparisons: bool = False,
    comparison_response: Optional[str] = None,
    candidate_queries_from: Optional[str] = None,
) -> None:
    policy: BasePolicy

    if candidate_queries_from in ("fixed", "rollouts_fixed"):
        print(
            f"Writing out fixed {len(reward_model.candidate_queries)} "
            "candidate queries to 'fixed_candidate_queries.pkl'"
        )
        with Artifact("fixed_candidate_queries.pkl", "wb", _run) as f:
            pickle.dump(reward_model.candidate_queries, f)

    evaluation_solver.set_env(env)
    if evaluation_policy_path is not None:
        assert isinstance(evaluation_solver, AugmentedRandomSearchSolver)
        policy = LinearPolicy.load(evaluation_policy_path, env)
    else:
        policy = evaluation_solver.solve(reward_model.solver_iterations)
    evaluation_policy_return = policy.evaluate(env)
    print("evaluation_policy_return", evaluation_policy_return)
    _run.info["evaluation_policy_return"] = evaluation_policy_return

    if fix_candidate_policy_to_optimal:
        reward_model.candidate_policies = [
            policy,
            policy,
        ]

    os.makedirs(os.path.join(PLOTS_PATH, label), exist_ok=True)

    wrapped_eval_env = RewardModelMeanWrapper(env, reward_model)
    evaluation_solver.set_env(wrapped_eval_env)

    global t
    global x_test_uniq, y_test_uniq
    t = 0  # type: ignore
    x_test_uniq, y_test_uniq = None, None  # type: ignore

    def callback(locals, globals):
        global t
        global x_test_uniq, y_test_uniq
        _run.log_scalar("time", time.time() - t)
        model = locals["self"]
        it = locals["iteration"]
        print("Iteration", it)

        if it == 1:
            if get_evaluation_set_from == "uniform":
                if isinstance(env.unwrapped, TabularMDP):
                    episode_length = env.episode_length
                else:
                    episode_length = env.spec.max_episode_steps
                n_samples = n_rollouts_for_states * episode_length
                x_test, y_test = reward_model.env.sample_features_rewards(n_samples)
                x_test_uniq, y_test_uniq = get_unique_x_y(x_test, y_test)
            else:
                # get evaluation states from queries
                x_test, y_test = [], []
                for query in reward_model.candidate_queries:

                    if isinstance(query, ComparisonQueryLinear):
                        x_test.append(query.gp_repr1)
                        y_test.append(query.reward1)
                        x_test.append(query.gp_repr2)
                        y_test.append(query.reward2)
                    elif isinstance(query, TrajectoryQuery):
                        for gp_repr, reward in zip(
                            query.gp_repr_list, query.reward_list
                        ):
                            x_test.append(gp_repr)
                            y_test.append(reward)
                    else:
                        x_test.append(query.gp_repr)
                        y_test.append(query.reward)
                x_test = np.array(x_test)
                y_test = np.array(y_test)
                x_test_uniq, y_test_uniq = get_unique_x_y(x_test, y_test)
            if len(x_test_uniq) > 1000:
                idx = np.random.choice(np.arange(len(x_test_uniq)), 1000)
                x_test_uniq = x_test_uniq[idx]
                y_test_uniq = y_test_uniq[idx]

        if save_models_path is not None:
            path = save_models_path + "_" + str(it)
            model.gp_model.save(path)
            print("saved", path)

        print("\tevaluation")
        t0 = time.time()
        if evaluate_policy:
            evaluation_solver.set_env(wrapped_eval_env)  # resets solver
            eval_policy = evaluation_solver.solve(reward_model.solver_iterations)
            ret = evaluation_solver.evaluate(env, N=n_episodes_for_policy_eval)
            log_metric(_run, it, "return", ret)
            log_metric(_run, it, "regret", evaluation_policy_return - ret)

        if model.candidate_policy_mean_jaccard_dist is not None:
            log_metric(
                _run,
                it,
                "candidate_policy_mean_jaccard_dist",
                model.candidate_policy_mean_jaccard_dist,
            )

        # evaluate selected policy
        if model.selected_policy is not None:
            selected_policy_ret = model.selected_policy.evaluate(
                env, N=n_episodes_for_policy_eval
            )
            log_metric(_run, it, "selected_policy_return", selected_policy_ret)

        _run.log_scalar("n_observed", len(model.observed))

        if "evaluate_policy_time" not in model.timing:
            model.timing["evaluate_policy_time"] = 0
        model.timing["evaluate_policy_time"] += time.time() - t0

        # evaluate reward model
        mu, cov = model.gp_model.predict_multiple(
            x_test_uniq, random_fourier_features=False
        )

        # GP's predictive likelihood
        log_metric(
            _run,
            it,
            "cand_log_likelihood",
            gaussian_log_likelihood(y_test_uniq, mu, cov),
        )

        # Mean squared error
        log_metric(_run, it, "cand_mse", np.mean(np.square(y_test_uniq - mu)))
        log_metric(
            _run, it, "cand_mape", np.mean(np.abs((y_test_uniq - mu) / y_test_uniq))
        )

        # EPIC distance
        corr = pearsonr(mu, y_test_uniq)[0]
        epic = np.sqrt((1 - corr) / 2)
        log_metric(_run, it, "epic", epic)

        # percentage of correct sign
        sign1 = np.sign(y_test_uniq)
        sign2 = np.sign(mu)
        log_metric(_run, it, "cand_correct_sign_perc", np.mean(sign1 == sign2))

        # Total variance
        log_metric(_run, it, "cand_total_var", np.sum(np.diag(cov)))

        # for linear kernel can directly log misspecification
        if isinstance(model.gp_model.kernel, LinearKernel) and (
            "Maze1D" in env_gym_label
            or "HighwayDriving" in env_gym_label
            or "Linear" in env_gym_label
        ):
            # Note: deprecated for old swimmer envs
            w_true = env.reward_w
            w_gp = model.gp_model.linear_predictive_mean
            print("learned_linear_weights", w_gp)
            alignment_w = np.dot(w_true, w_gp) / (
                np.linalg.norm(w_true) * np.linalg.norm(w_gp)
            )
            log_metric(_run, it, "learned_linear_weight_alignment", alignment_w)
            mse_w = np.mean(np.square(w_true - w_gp))
            log_metric(_run, it, "learned_linear_weight_mse", mse_w)
            for i, val in enumerate(w_gp):
                log_metric(_run, it, f"learned_linear_weight_{i}", val)

        if model.updated_candidate_policies_in_last_step:
            # evaluate the candidate policies
            pol_ret = []
            for policy in model.candidate_policies:
                policy_return = policy.evaluate(env, N=n_episodes_for_policy_eval)
                pol_ret.append(policy_return)
            pol_ret = np.array(pol_ret)

            if model.candidate_policy_posterior_probabilities is not None and len(
                model.candidate_policy_posterior_probabilities
            ) == len(model.candidate_policies):
                probs = np.array(model.candidate_policy_posterior_probabilities)
                bayes_expected_return = np.sum(pol_ret * probs) / np.sum(probs)
                log_metric(_run, it, "bayes_expected_return", bayes_expected_return)

            if model.use_mean_for_candidate_policies:
                log_metric(
                    _run, it, "cand_policy_for_mean_of_model_return", pol_ret[-1]
                )
                inferred_ret = model.candidate_policies[-1].evaluate(
                    wrapped_eval_env, N=n_episodes_for_policy_eval
                )
                log_metric(
                    _run,
                    it,
                    "cand_policy_for_mean_of_model_inferred_return",
                    inferred_ret,
                )

            # mean and variance of candidate policy returns
            log_metric(_run, it, "cand_policy_returns_mean", np.mean(pol_ret))
            log_metric(_run, it, "cand_policy_returns_median", np.median(pol_ret))
            log_metric(_run, it, "cand_policy_returns_min", np.min(pol_ret))
            log_metric(_run, it, "cand_policy_returns_max", np.max(pol_ret))
            log_metric(
                _run, it, "cand_policy_returns_25perc", np.percentile(pol_ret, 25)
            )
            log_metric(
                _run, it, "cand_policy_returns_75perc", np.percentile(pol_ret, 75)
            )
            log_metric(_run, it, "cand_policy_returns_var", np.var(pol_ret))

        _run.info["queries"] = model.observed_dict

        print("---")
        print("queried")
        if not ("Square" in env_gym_label or "Maze" in env_gym_label):
            for gp_repr in model.last_query.gp_repr_list:
                print(gp_repr)
        print("reward", model.last_query.reward)
        print("---")

        if model.last_query is not None:
            if plot_query:
                plot_folder = os.path.join(PLOTS_PATH, label)
                filename = "query_{}.png".format(it)
                mean, var = model.gp_model.predict(model.last_query.gp_repr)
                plot_query_state(
                    env, model.last_query, ret, mean, var, plot_folder, filename
                )
            if plot_policy:
                filename = "policy_{}.gif".format(it)
                plot_policy_path = os.path.join(PLOTS_PATH, label, filename)
                record_gym_video(env, eval_policy, plot_policy_path)

    reward_model.run(stop_after, callback=callback, print_timing=print_timing)

    # DL: Deactivated model saving due to an issue in pickeling the linear GP.
    with Artifact("final_gp_model.pkl", None, _run) as f:
        reward_model.gp_model.save(f)
        print("saved", f)


@ex.config
def cfg():
    acquisition_function = {}
    plot_query = False
    plot_policy = False
    observation_variance = 0.1
    mdp = {}
    stop_after = None
    experiment_label = "None"
    print_timing = True
    fix_candidate_policy_to_optimal = False
    evaluate_policy = True
    evaluation_solver_key = None
    n_episodes_for_policy_eval = 10
    save_models_path = None
    evaluation_policy_path = None
    get_evaluation_set_from = "queries"
    n_rollouts_for_states = 1
    use_comparisons = False
    comparison_response = None


@ex.automain
def run(
    _run,
    seed,
    acquisition_function,
    plot_query,
    plot_policy,
    observation_variance,
    mdp,
    stop_after,
    experiment_label,
    print_timing,
    fix_candidate_policy_to_optimal,
    evaluate_policy,
    evaluation_solver_key,
    n_episodes_for_policy_eval,
    save_models_path,
    evaluation_policy_path,
    get_evaluation_set_from,
    n_rollouts_for_states,
    use_comparisons,
    comparison_response,
):
    params = locals()
    print("--------")
    for key, val in params.items():
        print(key, val)
    print("--------")

    assert comparison_response is not None or not use_comparisons
    assert (
        not fix_candidate_policy_to_optimal
        or not acquisition_function["use_thompson_sampling_for_candidate_policies"]
    )

    if mdp["label"] == "gym":
        assert "observation_noise" in mdp
        assert "gym_label" in mdp
        env_gym_label = mdp["gym_label"]
        env = gym.make(mdp["gym_label"])

        if mdp["gym_label"] == "HighwayDriving-v0":
            assert solver_name == "lbfgsb_solver", "Need to use lbfgs for highway"

        print(
            "Loaded environment '{}' with observation shape {} and action shape {}".format(
                env_gym_label, env.observation_space.shape, env.action_space.shape
            )
        )
    elif mdp["label"] in ("gridworld", "gridworld_point"):
        observation_noise = get_dict_default(mdp, "observation_noise", 0)
        gaussian_peaks_as_rewards = mdp["label"] == "gridworld"
        if gaussian_peaks_as_rewards:
            raise NotImplementedError()
        env_seed = np.random.randint(1, 100000)
        print("Gridworld environment seed:", env_seed)
        env_gym_label = "GridworldPointRandom-v0"
        env = gym.make(
            env_gym_label, env_seed=env_seed, observation_noise=observation_noise
        )
    elif mdp["label"] == "limited_chain":
        assert "N_states" in mdp
        observation_noise = get_dict_default(mdp, "observation_noise", 0)
        env_seed = np.random.randint(1, 100000)
        print("LimitedActionChain environment seed:", env_seed)
        ndim = 1 if mdp["observation_type"] == "state" else mdp["N_states"]
        kernel = get_kernel(mdp, ndim)
        rewards = draw_random_rewards_from_GP(
            mdp["N_states"], kernel, reward_seed=env_seed
        )
        observation_type = mdp["observation_type"].capitalize()
        env_gym_label = f"Chain-{observation_type}-v0"
        env = gym.make(
            env_gym_label, rewards=rewards, observation_noise=observation_noise
        )
    elif mdp["label"] in ("junction_ei"):
        assert "N" in mdp
        assert "M" in mdp
        assert "observation_type" in mdp
        observation_noise = get_dict_default(mdp, "observation_noise", 0)
        N, M = mdp["N"], mdp["M"]
        observation_type = mdp["observation_type"].capitalize()
        env_gym_label = f"JunctionEI-N{N}-M{M}-{observation_type}-v0"
        env = gym.make(env_gym_label, observation_noise=observation_noise)
    elif mdp["label"] in ("junction_ordinal"):
        assert "observation_type" in mdp
        observation_noise = get_dict_default(mdp, "observation_noise", 0)
        observation_type = mdp["observation_type"].capitalize()
        env_gym_label = f"JunctionOrdinal-{observation_type}-v0"
        x = np.random.random()
        env = gym.make(env_gym_label, x=x, observation_noise=observation_noise)
    else:
        raise Exception("Unrecognized mdp label")

    env_horizon = get_dict_default(mdp, "horizon", None)
    if env_horizon is not None:
        env = gym.wrappers.TimeLimit(env, env_horizon)

    kernel = get_kernel(mdp, env.Ndim_repr)

    assert "solver_key" in acquisition_function
    solver_key = acquisition_function["solver_key"]
    if evaluation_solver_key is None:
        evaluation_solver_key = solver_key

    if (evaluation_solver_key == "lp_solver" or solver_key == "lp_solver") and not (
        isinstance(env, TabularMDP) or isinstance(env.unwrapped, TabularMDP)
    ):
        raise ValueError(
            "'lp_solver' can only be used with TabularMDP "
            "(eval_solver {}, solver {})".format(evaluation_solver_key, solver_key)
        )

    label = LABELS[acquisition_function["label"]]

    if label == "thompson_sampling":
        update_candidate_policies_every_n = 1
        n_candidate_policies_thompson_sampling = 1
    elif label == "two_step_ts_kl":
        update_candidate_policies_every_n = 1
        n_candidate_policies_thompson_sampling = 2
    else:
        update_candidate_policies_every_n = get_dict_assert(
            acquisition_function, "update_candidate_policies_every_n"
        )
        n_candidate_policies_thompson_sampling = get_dict_assert(
            acquisition_function, "n_candidate_policies_thompson_sampling"
        )

    optimize_gp_parameters_every_n = get_dict_default(
        acquisition_function, "optimize_gp_parameters_every_n", 0
    )

    gp_num_inducing = get_dict_default(acquisition_function, "gp_num_inducing", None)
    rollout_horizon = get_dict_default(acquisition_function, "rollout_horizon", None)

    subsampling_candidate_queries_n = get_dict_default(
        acquisition_function, "subsampling_candidate_queries_n", None
    )
    use_mean_for_candidate_policies = get_dict_default(
        acquisition_function, "use_mean_for_candidate_policies", False
    )
    use_thompson_sampling_for_candidate_policies = get_dict_default(
        acquisition_function, "use_thompson_sampling_for_candidate_policies", True
    )
    initialize_candidate_policies = get_dict_default(
        acquisition_function, "initialize_candidate_policies", False
    )
    candidate_queries_from = get_dict_default(
        acquisition_function,
        "candidate_queries_from",
        "fixed",
    )
    candidate_queries_file = get_dict_default(
        acquisition_function, "candidate_queries_file", None
    )
    n_policies_initial_ts = get_dict_default(
        acquisition_function, "n_policies_initial_ts", None
    )
    trajectory_clip_length = get_dict_default(
        acquisition_function, "trajectory_clip_length", None
    )
    trajectory_n_clips = get_dict_default(
        acquisition_function, "trajectory_n_clips", None
    )

    observation_batch_size = get_dict_default(
        acquisition_function, "observation_batch_size", 1
    )
    stop_after = stop_after // observation_batch_size

    assert (
        (candidate_queries_from not in ("rollouts_fixed", "rollouts_updated"))
        or use_thompson_sampling_for_candidate_policies
        or n_policies_initial_ts is not None
        or initialize_candidate_policies
    )
    use_trajectories_to_evaluate_policy = get_dict_default(
        acquisition_function, "use_trajectories_to_evaluate_policy", True
    )

    subsample_traj_for_queries_len = get_dict_default(
        acquisition_function, "subsample_traj_for_queries_len", None
    )

    if fix_candidate_policy_to_optimal:
        use_thompson_sampling_for_candidate_policies = False
        use_mean_for_candidate_policies = False

    solver_policies_file = get_dict_default(
        acquisition_function, "solver_policies_file", None
    )
    if solver_key == "lbfgsb_argmax_solver":
        assert solver_policies_file is not None, "need solver_policies_file"

    solver = get_standard_solver(
        env,
        solver_key,
        solver_policies_file=solver_policies_file,
    )
    solver_iterations = get_dict_default(acquisition_function, "solver_iterations", 100)
    print("Initialized solver", solver_key)

    common_kwargs = {
        "solver_iterations": solver_iterations,
        "initialize_candidate_policies": initialize_candidate_policies,
        "optimize_gp_parameters_every_n": optimize_gp_parameters_every_n,
        "use_thompson_sampling_for_candidate_policies": use_thompson_sampling_for_candidate_policies,
        "n_rollouts_for_states": n_rollouts_for_states,
        "candidate_queries_from": candidate_queries_from,
        "update_candidate_policies_every_n": update_candidate_policies_every_n,
        "n_candidate_policies_thompson_sampling": n_candidate_policies_thompson_sampling,
        "use_mean_for_candidate_policies": use_mean_for_candidate_policies,
        "gp_num_inducing": gp_num_inducing,
        "rollout_horizon": rollout_horizon,
        "subsampling_candidate_queries_n": subsampling_candidate_queries_n,
        "use_trajectories_to_evaluate_policy": use_trajectories_to_evaluate_policy,
        "use_comparisons": use_comparisons,
        "comparison_response": comparison_response,
        "subsample_traj_for_queries_len": subsample_traj_for_queries_len,
        "n_policies_initial_ts": n_policies_initial_ts,
        "candidate_queries_file": candidate_queries_file,
        "trajectory_clip_length": trajectory_clip_length,
        "trajectory_n_clips": trajectory_n_clips,
        "af_label": label,
        "observation_batch_size": observation_batch_size,
    }

    if label == "thompson_sampling":
        reward_model = TwoStepGPRewardModel(
            env,
            kernel,
            solver,
            policy_selection_none,
            state_selection_MI,
            observation_variance,
            arguments={},
            **common_kwargs,
        )
    elif label == "directed_information_gain":
        reward_model = TwoStepGPRewardModel(
            env,
            kernel,
            solver,
            policy_selection_most_uncertain_pair_of_plausible_maximizers,
            state_selection_MI_diff,
            observation_variance,
            arguments={
                "n_policies": 2,
            },
            **common_kwargs,
        )
    elif label == "maximum_regret":
        simple_model = get_dict_default(acquisition_function, "simple_model", False)
        if simple_model:
            uncertainty_p = get_dict_assert(acquisition_function, "uncertainty_p")
        else:
            uncertainty_p = None
        reward_model = TwoStepGPRewardModel(
            env,
            kernel,
            solver,
            policy_selection_maximum_regret,
            query_selection_policy_idx,
            observation_variance,
            arguments={
                "n_policies": 2,
                "simple_model": simple_model,
                "uncertainty_p": uncertainty_p,
            },
            **common_kwargs,
        )
    else:
        reward_model = BasicGPRewardModel(
            env,
            ACQUISITION_FUNCTIONS[acquisition_function["label"]],
            kernel,
            solver,
            observation_variance,
            **common_kwargs,
        )

    evaluation_solver = get_standard_solver(
        env,
        evaluation_solver_key,
        solver_policies_file=solver_policies_file,
        debug=False,
    )

    print("Testing {} acquisition".format(label), flush=True)
    run_active_learning(
        _run,
        env,
        reward_model,
        evaluation_solver,
        stop_after,
        label=experiment_label,
        plot_query=plot_query,
        plot_policy=plot_policy,
        print_timing=print_timing,
        fix_candidate_policy_to_optimal=fix_candidate_policy_to_optimal,
        evaluate_policy=evaluate_policy,
        n_episodes_for_policy_eval=n_episodes_for_policy_eval,
        save_models_path=save_models_path,
        evaluation_policy_path=evaluation_policy_path,
        n_rollouts_for_states=n_rollouts_for_states,
        get_evaluation_set_from=get_evaluation_set_from,
        env_gym_label=env_gym_label,
        use_comparisons=use_comparisons,
        comparison_response=comparison_response,
        candidate_queries_from=candidate_queries_from,
    )

    _run.info["run_time_without_evaluation"] = reward_model.run_time
    print("Run time without evaluations:", reward_model.run_time)
