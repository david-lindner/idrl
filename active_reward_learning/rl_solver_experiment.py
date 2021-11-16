import time

import gym
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

from active_reward_learning.envs import Gridworld, RewardModelMeanWrapper
from active_reward_learning.solvers import (
    get_observations_required_for_solver,
    get_standard_solver,
)
from active_reward_learning.util.video import record_gym_video

ex = Experiment("rl_solver_experiment")
ex.observers = [FileStorageObserver("results/rl_solver_experiments")]

t = 0
i = 0
dt = 0


@ex.named_config
def point_right():
    env_str = "Point-Right-v2"
    solver_str = "ars"
    iterations = 100
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_point_right.npy"


@ex.named_config
def point_left():
    env_str = "Point-Left-v2"
    solver_str = "ars"
    iterations = 100
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_point_left.npy"


@ex.named_config
def point_up():
    env_str = "Point-Up-v2"
    solver_str = "ars"
    iterations = 100
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_point_up.npy"


@ex.named_config
def point_down():
    env_str = "Point-Down-v2"
    solver_str = "ars"
    iterations = 100
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_point_down.npy"


@ex.named_config
def swimmer_right():
    env_str = "Swimmer-Right-v2"
    solver_str = "ars"
    iterations = 100
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_swimmer_right.npy"


@ex.named_config
def swimmer_left():
    env_str = "Swimmer-Left-v2"
    solver_str = "ars"
    iterations = 100
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_swimmer_left.npy"


@ex.named_config
def swimmer_up():
    env_str = "Swimmer-Up-v2"
    solver_str = "ars"
    iterations = 100
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_swimmer_up.npy"


@ex.named_config
def swimmer_down():
    env_str = "Swimmer-Down-v2"
    solver_str = "ars"
    iterations = 100
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_swimmer_down.npy"


@ex.named_config
def ant_right():
    env_str = "Ant-Right-v2"
    solver_str = "ars"
    iterations = 500
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_ant_right.npy"


@ex.named_config
def ant_left():
    env_str = "Ant-Left-v2"
    solver_str = "ars"
    iterations = 500
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_ant_left.npy"


@ex.named_config
def ant_up():
    env_str = "Ant-Up-v2"
    solver_str = "ars"
    iterations = 500
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_ant_up.npy"


@ex.named_config
def ant_up_right():
    env_str = "Ant-Up-Right-v2"
    solver_str = "ars"
    iterations = 500
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_ant_up_right.npy"


@ex.named_config
def ant_down_right():
    env_str = "Ant-Down-Right-v2"
    solver_str = "ars"
    iterations = 500
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_ant_down_right.npy"


@ex.named_config
def ant_down_left():
    env_str = "Ant-Down-Left-v2"
    solver_str = "ars"
    iterations = 500
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_ant_down_left.npy"


@ex.named_config
def ant_up_left():
    env_str = "Ant-Up-Left-v2"
    solver_str = "ars"
    iterations = 500
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_ant_up_left.npy"


@ex.named_config
def ant_down():
    env_str = "Ant-Down-v2"
    solver_str = "ars"
    iterations = 500
    evaluate_every = 10
    n_jobs = 1
    horizon = 1000
    policy_out_file = "policy_ant_down.npy"


@ex.named_config
def test():
    env_str = "Ant-Right-v2"

    solver_str = "ars"
    iterations = 30
    evaluate_every = 10

    load_dqn_file = None
    experiment_label = ""
    render_gym = False

    n_jobs = 1
    horizon = None


@ex.config
def cfg():
    iterations = 1000
    env_str = None
    solver_str = "dqn_linear"
    evaluate_every = 1
    load_dqn_file = None
    experiment_label = ""
    render_gym = False
    gp_model_file = None
    feature_net_file = None
    n_jobs = 1
    horizon = None
    policy_out_file = "ars_policy.npy"


@ex.automain
def run(
    _run,
    seed,
    iterations,
    env_str,
    solver_str,
    evaluate_every,
    load_dqn_file,
    render_gym,
    gp_model_file,
    feature_net_file,
    n_jobs,
    horizon,
    policy_out_file,
):
    observation_type = get_observations_required_for_solver(solver_str)
    if env_str is None:
        env_str = load_dqn_file.split("/")[-1].split("_")[0]

    if env_str == "small_gridworld_0":
        env_base = Gridworld(
            5,
            5,
            10,
            1,
            100,
            0.99,  # discount factor
            0,  # wall prob
            env_seed=seed,
            observation_type=observation_type,
            gaussian_peaks_as_rewards=True,
        )
    elif env_str == "small_gridworld_30":
        env_base = Gridworld(
            5,
            5,
            10,
            1,
            100,
            0.99,  # discount factor
            0.3,  # wall prob
            env_seed=seed,
            observation_type=observation_type,
            gaussian_peaks_as_rewards=True,
        )
    elif env_str == "small_gridworld_50":
        env_base = Gridworld(
            5,
            5,
            10,
            1,
            100,
            0.99,  # discount factor
            0.5,  # wall prob
            env_seed=seed,
            observation_type=observation_type,
            gaussian_peaks_as_rewards=True,
        )
    elif env_str in (
        "Swimmer-Right-v2",
        "Swimmer-Left-v2",
        "Swimmer-Up-v2",
        "Swimmer-Down-v2",
    ):
        env_base = gym.make(env_str)
    elif "Swimmer-" in env_str and "v2" in env_str:
        env_base = gym.make(env_str, None, horizon)
    else:
        # print("Available envs:")
        # print(gym.envs.registry.all())
        print("Warning unknown env_str '{}'".format(env_str))
        env_base = gym.make(env_str)

    if gp_model_file is not None:
        env = RewardModelMeanWrapper.load_from_model(
            env_base, gp_model_file, debug=False
        )
    else:
        env = env_base

    print("observation_space", env.observation_space)
    print("action_space", env.action_space)

    solver = get_standard_solver(
        env, solver_str, load_dqn_file=load_dqn_file, n_jobs=n_jobs
    )
    if solver_str == "ars":
        solver.policy_out_file = policy_out_file

    if solver_str == "dqn_linear":
        params = solver.dqn.model.get_parameters()
        weights = params[
            "deepq/target_q_func/model/action_value/fully_connected/weights:0"
        ]
        biases = params[
            "deepq/target_q_func/model/action_value/fully_connected/biases:0"
        ]
        print("weights", weights)
        print("biases", biases)

    global t
    global i
    t = time.time()
    i = 0
    n_eval = 10

    def logging_callback(locals, globals):
        global t
        global dt
        global i
        if evaluate_every is not None and i % evaluate_every == 0:
            print()
            dt += time.time() - t
            solver.update_policy()

            if solver_str == "dqn_linear":
                print("dqn_return", solver.dqn.evaluate(N=n_eval))
                # print(np.sum(solver.linear_q.w), np.sum(solver.linear_q.w ** 2))
                # print(solver.linear_q.w)
                print("α={}, ε={}".format(locals["alpha"], locals["eps"]))
                print(
                    "Σw =",
                    np.sum(solver.linear_q.w),
                    "   Σw^2 =",
                    np.sum(solver.linear_q.w ** 2),
                )
            elif solver_str == "linear_q_learning":
                print("α={}, ε={}".format(locals["alpha"], locals["eps"]))
                print("Σw =", np.sum(solver.w), "   Σw^2 =", np.sum(solver.w ** 2))
                # print(solver.w)

            inferred_return = solver.evaluate(env, N=n_eval)
            env_return = solver.evaluate(env_base, N=n_eval)
            print(
                "{}  time {}  inferred return {:.3f}  true return {:.3f}".format(
                    i, int(dt), inferred_return, env_return
                ),
                flush=True,
            )

            _run.log_scalar("time", dt, i)
            _run.log_scalar("true_return", env_return, i)
            _run.log_scalar("inferred_return", inferred_return, i)

            if solver_str == "ars":
                if "weights" not in _run.info:
                    _run.info["weights"] = []
                if "means" not in _run.info:
                    _run.info["means"] = []
                if "stds" not in _run.info:
                    _run.info["stds"] = []
                _run.info["weights"].append(solver.w)
                _run.info["means"].append(solver.mean)
                _run.info["stds"].append(solver.std)
            t = time.time()
            print()
        i += 1
        return True  # continue training

    solver.solve(iterations, logging_callback=logging_callback)

    if solver_str == "ppo":
        solver.policy.model.save("ppo_policy")

    if solver_str == "sac":
        solver.policy.model.save("sac_policy")

    if solver_str == "lp_solver":
        env_return = solver.evaluate(rollout=True, N=n_eval)
    else:
        env_return = solver.evaluate(N=n_eval)
    print("Final return", env_return)

    if render_gym:
        record_gym_video(env, solver.policy, "out.mp4")
