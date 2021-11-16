import datetime
import os
import time
from typing import Callable

import gym
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver

from active_reward_learning.common.constants import LABELS, PLOTS_PATH
from active_reward_learning.envs import (
    FirstStateTrapChain,
    Gridworld,
    JunctionMDP,
    LimitedActionChain,
    TabularMDP,
)
from active_reward_learning.reward_models import (
    BasicGPRewardModel,
    TwoStepGPRewardModel,
    policy_selection_most_uncertain_pair_of_plausible_maximizers,
    policy_selection_none,
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
from active_reward_learning.reward_models.query import ComparisonQueryLinear
from active_reward_learning.solvers import (
    BaseSolver,
    LPSolver,
    get_observations_required_for_solver,
    get_standard_solver,
)
from active_reward_learning.util.helpers import get_dict_assert, get_dict_default
from active_reward_learning.util.plotting import plot_reward_value_policy


# changes the run _id and thereby the path that the FileStorageObserver
# writes the results
# cf. https://github.com/IDSIA/sacred/issues/174
class SetID(RunObserver):
    priority = 50  # very high priority to set id

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        custom_id = "{}".format(timestamp)
        return custom_id  # started_event returns the _run._id


ex = Experiment("tabular_mdp_experiment")
ex.observers = [SetID(), FileStorageObserver("results/tabular")]


def run_active_learning(
    _run,
    env: TabularMDP,
    reward_model: BasicGPRewardModel,
    evaluation_solver: BaseSolver,
    stop_after: int,
    label: str = "test",
    plot: bool = True,
    optimize_gp_parameters_every_n: int = 0,
    print_timing: bool = True,
) -> None:
    assert isinstance(env, TabularMDP)
    os.makedirs(os.path.join(PLOTS_PATH, label), exist_ok=True)

    print("\tfind optimal policy")
    lp_solver = LPSolver(env)
    policy = lp_solver.solve()
    optimal_return = policy.evaluate(env)
    print("optimal_return", optimal_return)

    global t
    t = 0  # type: ignore

    def callback(locals, globals):
        global t
        _run.log_scalar("time", time.time() - t)

        print("\nIteration", locals["iteration"])
        model = locals["self"]
        print("\tevaluation")
        # update policy
        print("\tget reward predictions")

        print("len(candidate_queries)", len(model.candidate_queries))

        if isinstance(env, TabularMDP):
            all_states = env.get_all_states_repr()
            mu_pred, sigma_pred = model.gp_model.predict_multiple(all_states)
            sigma_pred = np.diag(sigma_pred)

            print("\tfind policy wrt mean reward")
            t0 = time.time()

            policy = evaluation_solver.solve(rewards=mu_pred)

            if "evaluate_model_time" not in model.timing:
                model.timing["evaluate_model_time"] = 0
            model.timing["evaluate_model_time"] += time.time() - t0
        else:
            raise NotImplementedError()

        print("\tevaluate policy")
        t0 = time.time()
        # evaluate optimal policy wrt reward model
        policy_return = policy.evaluate(env)
        regret = optimal_return - policy_return
        _run.log_scalar("return", policy_return, locals["iteration"])
        _run.log_scalar("regret", regret, locals["iteration"])
        print("regret", regret)

        # evaluate selected policy
        if model.selected_policy is not None:
            selected_policy_return = model.selected_policy.evaluate(env)
            selected_policy_regret = optimal_return - selected_policy_return
            _run.log_scalar(
                "selected_policy_return", selected_policy_return, locals["iteration"]
            )
            _run.log_scalar(
                "selected_policy_regret", selected_policy_regret, locals["iteration"]
            )
            print("selected_policy_regret", selected_policy_regret)

        _run.log_scalar("n_observed", len(model.observed))

        if "evaluate_policy_time" not in model.timing:
            model.timing["evaluate_policy_time"] = 0
        model.timing["evaluate_policy_time"] += time.time() - t0

        if isinstance(model.last_query, ComparisonQueryLinear):
            print()
            print("queried")
            print(model.last_query.gp_repr1)
            print(model.last_query.gp_repr2)
            print("answer:", model.last_query.reward)
            print("gp_repr_list:", model.last_query.gp_repr_list)
            print("linear_combination:", model.last_query.linear_combination)
            print()

        if plot:
            print("\tcreate plots")
            t0 = time.time()
            value_function = env.get_value_function(policy)
            plot_reward_value_policy(
                env,
                mu_pred,
                sigma_pred,
                value_function,
                policy,
                policy_return,
                model.last_query,
                os.path.join(PLOTS_PATH, label),
                "reward_value_policy_%i.png" % locals["iteration"],
                None,  # model.gp_model.model,
            )
            if "plot_time" not in model.timing:
                model.timing["plot_time"] = 0
            model.timing["plot_time"] += time.time() - t0

    reward_model.run(stop_after, callback=callback, print_timing=print_timing)


def get_junction_dist(N: int, M: int) -> Callable[[int, int], float]:
    def junction_dist(s1, s2):
        if (s1 < N + M and s2 < N + M) or (s1 >= N + M and s2 >= N + M):
            return abs(s1 - s2) + 1
        else:
            if s1 > s2:
                s1, s2 = s2, s1
            return abs(N - 1 - s1) + abs(N + M - s2) + 2

    return junction_dist


def get_gridworld_dist(gridworld: Gridworld) -> Callable[[int, int], float]:
    def gridworld_dist(s1: int, s2: int) -> float:
        x1, y1 = gridworld._get_agent_pos_from_state(s1)
        x2, y2 = gridworld._get_agent_pos_from_state(s2)
        assert gridworld.tiles is not None
        t1 = gridworld.tiles[y1, x1]
        t2 = gridworld.tiles[y2, x2]
        return float(t1 != t2)

    return gridworld_dist


@ex.named_config
def test():

    acquisition_function = {
        "label": "var",
        "optimize_gp_parameters_every_n": 1,
        "solver_key": "lp_solver",
        "solver_iterations": 1,
    }

    # acquisition_function = {
    #     "label": "idrl",
    #     "optimize_gp_parameters_every_n": 1,
    #     "solver_key": "lp_solver",
    #     "solver_iterations": 1,
    #     "candidate_queries_from": "fixed",
    #     "n_rollouts": 1,
    #     "use_mean_for_candidate_policies": False,
    #     "update_candidate_policies_every_n": None,
    #     "n_candidate_policies_thompson_sampling": None,
    #     "use_thompson_sampling_for_candidate_policies": False,
    #     "gp_num_inducing": None,
    # }

    # acquisition_function = {
    #     "label": "idrl",
    #     "optimize_gp_parameters_every_n": 1,
    #     "solver_key": "lp_solver",
    #     "solver_iterations": 1,
    #     "candidate_queries_from": "fixed",
    #     "n_rollouts": 1,
    #     "use_mean_for_candidate_policies": False,
    #     "update_candidate_policies_every_n": 1,
    #     "n_candidate_policies_thompson_sampling": 5,
    #     "use_thompson_sampling_for_candidate_policies": True,
    #     "gp_num_inducing": None,
    # }

    # acquisition_function = {
    #     "label": "2s_ts_kl",
    #     "optimize_gp_parameters_every_n": 1,
    #     "solver_key": "lp_solver",
    #     "solver_iterations": 1,
    # }

    # acquisition_function = {
    #     # "label": "2s_ucb_mi_diff",
    #     # "label": "2s_ucb_mi",
    #     "label": "2s_ts_kl",
    #     "optimize_gp_parameters_every_n": null,
    # }

    # mdp = {
    #     "label": "limited_chain",
    #     "N_states": 10,
    #     "block_N": 5,
    #     "observation_noise": 0,
    #     "discount_factor": 0.99,
    #     "episode_length": 100,
    # }

    # mdp = {
    #     "label": "junction_ordinal",
    #     "x": 0.9,
    #     "observation_noise": 0,
    # }

    # mdp = {
    #     "label": "first_state_trap_chain",
    #     "N_states": 10,
    #     "observation_noise": 0.1,
    #     "discount_factor": 0.99,
    #     "episode_length": 100,
    #     "prob_to_first": 0.99,
    # }

    # mdp = {
    #     "label": "junction",
    #     "N": 15,
    #     "M": 5,
    #     "observation_noise": 0,
    #     "discount_factor": 0.99,
    #     "episode_length": 100,
    # }

    # mdp = {
    #     "label": "junction_ei",
    #     "N": 15,
    #     "M": 5,
    #     "observation_noise": 0,
    #     "discount_factor": 0.99,
    #     "episode_length": 100,
    #     "observation_type": "raw",
    # }

    mdp = {
        "label": "gridworld_point",
        "height": 10,
        "width": 10,
        "n_objects": 10,
        "n_per_object": 2,
        "episode_length": 100,
        "discount_factor": 0.99,
        "observation_noise": 0,
        "wall_probability": 0.3,
        "random_action_prob": 0,
        "add_optimal_policy_to_candidates": False,
        "observation_type": "state",
    }

    # mdp = {
    #     "label": "highway_1",
    #     "robots": "left_to_middle",
    #     "use_feature_representation": True,
    # }
    # mdp = {"label": "simple_highway_1", "observation_noise": 0}
    # mdp = {"label": "minecraft_stone_basic", "observation_noise": 0.1}

    use_comparisons = False
    observation_variance = 0.1
    stop_after = 100
    plot = True
    experiment_label = None
    print_timing = True
    env_seed = None


@ex.config
def cfg():
    acquisition_function = {}
    plot = False
    observation_variance = 0.1
    mdp = {}
    stop_after = None
    experiment_label = None
    print_timing = True
    use_comparisons = False
    env_seed = None


@ex.automain
def run(
    _run,
    seed,
    acquisition_function,
    plot,
    observation_variance,
    mdp,
    stop_after,
    experiment_label,
    print_timing,
    use_comparisons,
    env_seed,
):
    args = locals()
    print("--------")
    for key, val in args.items():
        print(key, val)
    print("--------")

    solver_key = get_dict_default(acquisition_function, "solver_key", "lp_solver")
    observation_type = get_observations_required_for_solver(solver_key)

    if mdp["label"] == "limited_chain":
        # FIXME: missing observation_type
        assert "N_states" in mdp
        assert "block_N" in mdp
        assert "episode_length" in mdp
        assert "discount_factor" in mdp
        kernel = RBFCustomDist(
            input_dim=1, variance=2, lengthscale=3, distance=lambda x, y: abs(x - y)
        )
        rewards = draw_random_rewards_from_GP(
            mdp["N_states"], kernel, reward_seed=env_seed
        )
        # normalize rewards
        min_r = rewards.min()
        max_r = rewards.max()
        rewards = (rewards - min_r) / (max_r - min_r)

        # rewards = np.array([0.1, 0.2, 0.15])
        env = LimitedActionChain(
            rewards, mdp["discount_factor"], mdp["episode_length"], mdp["block_N"]
        )
    elif mdp["label"] == "first_state_trap_chain":
        # FIXME: missing observation_type
        assert "N_states" in mdp
        assert "prob_to_first" in mdp
        assert "episode_length" in mdp
        assert "discount_factor" in mdp
        kernel = RBFCustomDist(
            input_dim=1, variance=2, lengthscale=3, distance=lambda x, y: abs(x - y)
        )
        rewards = draw_random_rewards_from_GP(
            mdp["N_states"], kernel, reward_seed=env_seed
        )
        prob_to_first = 0.1
        env = FirstStateTrapChain(
            rewards,
            mdp["discount_factor"],
            0,
            mdp["episode_length"],
            mdp["prob_to_first"],
        )
    elif mdp["label"] == "junction":
        # FIXME: missing observation_type
        assert "N" in mdp
        assert "M" in mdp
        assert "episode_length" in mdp
        assert "discount_factor" in mdp
        N, M = mdp["N"], mdp["M"]
        kernel = RBFCustomDist(
            input_dim=1, variance=2, lengthscale=3, distance=get_junction_dist(N, M)
        )

        rewards = draw_random_rewards_from_GP(N + M, kernel, reward_seed=env_seed)

        # normalize rewards
        min_r = rewards.min()
        max_r = rewards.max()
        rewards = (rewards - min_r) / (max_r - min_r)

        env = JunctionMDP(
            rewards[:N],
            np.minimum(rewards[N:] + 0.01, 1),
            rewards[N:],
            mdp["discount_factor"],
            mdp["episode_length"],
        )
    elif mdp["label"] == "junction_ei":
        assert "N" in mdp
        assert "M" in mdp
        assert "episode_length" in mdp
        assert "discount_factor" in mdp
        N, M = mdp["N"], mdp["M"]
        kernel = RBFCustomDist(
            input_dim=1, variance=2, lengthscale=3, distance=get_junction_dist(N, M)
        )
        reward_left = np.zeros(N)
        rewards_right_1 = np.ones(M) * 0.8
        rewards_right_2 = np.linspace(-1, -0.3, M) ** 2 * (-1) + 1
        # rewards_right_2[-1] = 1
        observation_type = get_dict_default(mdp, "observation_type", observation_type)
        env = JunctionMDP(
            reward_left,
            rewards_right_1,
            rewards_right_2,
            mdp["discount_factor"],
            mdp["episode_length"],
            observation_type=observation_type,
        )
    elif mdp["label"] == "junction_ordinal":
        assert "x" in mdp
        x = mdp["x"]
        kernel = RBFCustomDist(
            input_dim=1, variance=2, lengthscale=3, distance=get_junction_dist(1, 2)
        )
        observation_type = get_dict_default(mdp, "observation_type", observation_type)
        env = gym.make(
            f"JunctionOrdinal-{observation_type.capitalize()}-v0", x=x
        ).unwrapped
    elif mdp["label"] in ("gridworld", "gridworld_point"):
        assert "width" in mdp
        assert "height" in mdp
        assert "n_objects" in mdp
        assert "n_per_object" in mdp
        assert "wall_probability" in mdp
        assert "random_action_prob" in mdp
        assert "add_optimal_policy_to_candidates" in mdp
        assert "observation_noise" in mdp
        assert "episode_length" in mdp
        assert "discount_factor" in mdp

        gaussian_peaks_as_rewards = mdp["label"] == "gridworld"

        if env_seed is None:
            env_seed = np.random.randint(1, 100000)
        print("Gridworld environment seed:", env_seed)
        observation_type = get_dict_default(mdp, "observation_type", observation_type)
        env = Gridworld(
            mdp["width"],
            mdp["height"],
            mdp["n_objects"],
            mdp["n_per_object"],
            mdp["episode_length"],
            mdp["discount_factor"],
            observation_noise=mdp["observation_noise"],
            wall_probability=mdp["wall_probability"],
            env_seed=env_seed,
            random_action_prob=mdp["random_action_prob"],
            gaussian_peaks_as_rewards=gaussian_peaks_as_rewards,
            add_optimal_policy_to_candidates=mdp["add_optimal_policy_to_candidates"],
            observation_type=observation_type,
        )

        # normalize rewards
        min_r = env.rewards.min()
        max_r = env.rewards.max()
        env.rewards = (env.rewards - min_r) / (max_r - min_r)

        reachable_tiles = env.get_number_of_reachable_tiles()
        _run.info["reachable_tiles"] = reachable_tiles
        print("reachable_tiles", reachable_tiles)

        if mdp["label"] == "gridworld_point":
            kernel = LinearKernel(env.Ndim_repr, variances=np.ones(env.Ndim_repr))
        else:
            kernel = RBFCustomDist(input_dim=env.Ndim_repr, variance=2, lengthscale=1)
    else:
        raise Exception("Unrecognized mdp label")

    solver = get_standard_solver(env, solver_key)
    solver_iterations = get_dict_default(acquisition_function, "solver_iterations", 100)
    evaluation_solver = get_standard_solver(env, "lp_solver")

    label = LABELS[acquisition_function["label"]]

    use_thompson_sampling_for_candidate_policies = get_dict_default(
        acquisition_function, "use_thompson_sampling_for_candidate_policies", False
    )
    use_mean_for_candidate_policies = get_dict_default(
        acquisition_function, "use_mean_for_candidate_policies", False
    )
    if use_thompson_sampling_for_candidate_policies or use_mean_for_candidate_policies:
        update_candidate_policies_every_n = get_dict_assert(
            acquisition_function, "update_candidate_policies_every_n"
        )
    else:
        update_candidate_policies_every_n = None

    if use_thompson_sampling_for_candidate_policies:
        n_candidate_policies_thompson_sampling = get_dict_assert(
            acquisition_function, "n_candidate_policies_thompson_sampling"
        )
    else:
        n_candidate_policies_thompson_sampling = None

    use_trajectories_to_evaluate_policy = get_dict_default(
        acquisition_function, "use_trajectories_to_evaluate_policy", False
    )

    candidate_queries_from = get_dict_default(
        acquisition_function, "candidate_queries_from", "fixed"
    )

    n_rollouts_for_states = get_dict_default(acquisition_function, "n_rollouts", 1)

    optimize_gp_parameters_every_n = get_dict_default(
        acquisition_function, "optimize_gp_parameters_every_n", 0
    )

    gp_num_inducing = get_dict_default(acquisition_function, "gp_num_inducing", None)

    if label == "thompson_sampling":
        use_thompson_sampling_for_candidate_policies = True
        update_candidate_policies_every_n = 1
        n_candidate_policies_thompson_sampling = 1
        if n_rollouts_for_states is None:
            n_rollouts_for_states = 1
        reward_model = TwoStepGPRewardModel(
            env,
            kernel,
            solver,
            policy_selection_none,
            state_selection_MI,
            obs_var=observation_variance,
            arguments={},
            use_trajectories_to_evaluate_policy=use_trajectories_to_evaluate_policy,
            solver_iterations=solver_iterations,
            initialize_candidate_policies=not use_thompson_sampling_for_candidate_policies,
            optimize_gp_parameters_every_n=optimize_gp_parameters_every_n,
            use_thompson_sampling_for_candidate_policies=use_thompson_sampling_for_candidate_policies,
            update_candidate_policies_every_n=update_candidate_policies_every_n,
            n_candidate_policies_thompson_sampling=n_candidate_policies_thompson_sampling,
            n_rollouts_for_states=n_rollouts_for_states,
            candidate_queries_from=candidate_queries_from,
            use_mean_for_candidate_policies=use_mean_for_candidate_policies,
            gp_num_inducing=gp_num_inducing,
            use_comparisons=use_comparisons,
        )
    elif label == "directed_information_gain":
        reward_model = TwoStepGPRewardModel(
            env,
            kernel,
            solver,
            policy_selection_most_uncertain_pair_of_plausible_maximizers,
            state_selection_MI_diff,
            obs_var=observation_variance,
            arguments={
                "n_policies": 2,
            },
            use_trajectories_to_evaluate_policy=use_trajectories_to_evaluate_policy,
            solver_iterations=solver_iterations,
            initialize_candidate_policies=not use_thompson_sampling_for_candidate_policies,
            optimize_gp_parameters_every_n=optimize_gp_parameters_every_n,
            use_thompson_sampling_for_candidate_policies=use_thompson_sampling_for_candidate_policies,
            update_candidate_policies_every_n=update_candidate_policies_every_n,
            n_candidate_policies_thompson_sampling=n_candidate_policies_thompson_sampling,
            n_rollouts_for_states=n_rollouts_for_states,
            candidate_queries_from=candidate_queries_from,
            use_mean_for_candidate_policies=use_mean_for_candidate_policies,
            gp_num_inducing=gp_num_inducing,
            use_comparisons=use_comparisons,
        )
    else:
        reward_model = BasicGPRewardModel(
            env,
            ACQUISITION_FUNCTIONS[acquisition_function["label"]],
            kernel,
            solver,
            obs_var=observation_variance,
            solver_iterations=solver_iterations,
            optimize_gp_parameters_every_n=optimize_gp_parameters_every_n,
            use_thompson_sampling_for_candidate_policies=use_thompson_sampling_for_candidate_policies,
            update_candidate_policies_every_n=update_candidate_policies_every_n,
            n_candidate_policies_thompson_sampling=n_candidate_policies_thompson_sampling,
            n_rollouts_for_states=n_rollouts_for_states,
            candidate_queries_from=candidate_queries_from,
            use_mean_for_candidate_policies=use_mean_for_candidate_policies,
            gp_num_inducing=gp_num_inducing,
            use_comparisons=use_comparisons,
        )

    if stop_after > env.N_states:
        print(
            "Warning:",
            "stop_after ({}) > env.N_states ({}),".format(stop_after, env.N_states),
            "this might be a problem.",
            flush=True,
        )

    print("Testing {} acquisition".format(label), flush=True)
    run_active_learning(
        _run,
        env,
        reward_model,
        evaluation_solver,
        stop_after,
        label,
        plot,
        print_timing,
    )
