import numpy as np

from active_reward_learning.tabular_experiment import ex as tabular_experiment


def test_all_methods_run_without_error_on_simple_env():
    acquisition_functions = [
        {
            "label": "idrl",
            "use_thompson_sampling_for_candidate_policies": False,
        },
        {"label": "2s_ts", "n_rollouts": 1},
        {
            "label": "2s_ts",
            "n_rollouts": 1,
            "optimize_gp_parameters_every_n": 1,
        },
        {"label": "epd"},
        {"label": "rand"},
        {"label": "var"},
        {"label": "ei"},
        {"label": "pi"},
        {
            "label": "rand",
            "optimize_gp_parameters_every_n": 1,
        },
    ]
    mdps = [
        {
            "label": "limited_chain",
            "N_states": 5,
            "block_N": 3,
            "discount_factor": 0.99,
            "episode_length": 10,
        },
        {
            "label": "junction",
            "N": 3,
            "M": 3,
            "discount_factor": 0.99,
            "episode_length": 10,
        },
        {
            "label": "gridworld_point",
            "discount_factor": 0.99,
            "episode_length": 10,
            "width": 5,
            "height": 5,
            "n_objects": 5,
            "n_per_object": 2,
            "wall_probability": 0.3,
            "random_action_prob": 0.3,
            "add_optimal_policy_to_candidates": False,
            "observation_noise": 0,
        },
        {
            "label": "gridworld",
            "discount_factor": 0.99,
            "episode_length": 10,
            "width": 5,
            "height": 5,
            "n_objects": 5,
            "n_per_object": 2,
            "wall_probability": 0.3,
            "random_action_prob": 0.3,
            "add_optimal_policy_to_candidates": True,
            "observation_noise": 0,
        },
    ]
    tabular_experiment.observers = []

    np.random.seed(1)

    for mdp in mdps:
        seed = np.random.randint(1, 100000)
        for acquisition_function in acquisition_functions:
            print(mdp, acquisition_function)
            tabular_experiment.run(
                config_updates={
                    "acquisition_function": acquisition_function,
                    "seed": seed,
                    "plot": False,
                    "mdp": mdp,
                    "stop_after": 3,
                }
            )


def test_all_methods_run_without_error_on_simple_env_comparisons():
    acquisition_functions = [
        {
            "label": "idrl",
            "use_thompson_sampling_for_candidate_policies": False,
        },
        {"label": "epd"},
        {"label": "rand"},
        {"label": "var"},
        {
            "label": "rand",
            "optimize_gp_parameters_every_n": 1,
        },
    ]
    mdps = [
        {
            "label": "limited_chain",
            "N_states": 5,
            "block_N": 3,
            "discount_factor": 0.99,
            "episode_length": 10,
        },
        {
            "label": "junction",
            "N": 3,
            "M": 3,
            "discount_factor": 0.99,
            "episode_length": 10,
        },
        # {
        #     "label": "gridworld_point",
        #     "discount_factor": 0.99,
        #     "episode_length": 10,
        #     "width": 5,
        #     "height": 5,
        #     "n_objects": 5,
        #     "n_per_object": 2,
        #     "wall_probability": 0.3,
        #     "random_action_prob": 0.3,
        #     "add_optimal_policy_to_candidates": False,
        #     "observation_noise": 0,
        # },
        # {
        #     "label": "gridworld",
        #     "discount_factor": 0.99,
        #     "episode_length": 10,
        #     "width": 5,
        #     "height": 5,
        #     "n_objects": 5,
        #     "n_per_object": 2,
        #     "wall_probability": 0.3,
        #     "random_action_prob": 0.3,
        #     "add_optimal_policy_to_candidates": True,
        #     "observation_noise": 0,
        # },
    ]
    tabular_experiment.observers = []

    np.random.seed(1)

    for mdp in mdps:
        seed = np.random.randint(1, 100000)
        for acquisition_function in acquisition_functions:
            print(mdp, acquisition_function)
            tabular_experiment.run(
                config_updates={
                    "acquisition_function": acquisition_function,
                    "seed": seed,
                    "plot": False,
                    "mdp": mdp,
                    "stop_after": 3,
                    "use_comparisons": True,
                }
            )
