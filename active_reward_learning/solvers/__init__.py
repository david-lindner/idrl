from active_reward_learning.util.helpers import get_dict_default

from .argmax_solver import ArgmaxSolver
from .ars import AugmentedRandomSearchSolver
from .base import BaseSolver
from .lbfgsb_argmax_solver import LBFGSArgmaxSolver
from .lbfgsb_solver import LBFGSSolver
from .lp_solver import LPSolver

try:
    from stable_baselines import DQN, PPO2, SAC
    from stable_baselines.common.policies import CnnPolicy as CommonCnnPolicy
    from stable_baselines.common.policies import MlpPolicy as CommonMlpPolicy
    from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy
    from stable_baselines.sac.policies import CnnPolicy as SACCnnPolicy
    from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy

    from .stable_baselines_solver import StableBaselinesSolver

    drl_available = True
except ImportError as e:
    print(e)
    print("Warning: stable-baselines package does not seem to be installed.")
    drl_available = False


def get_standard_solver(env, key, **kwargs):
    if key == "lp_solver":
        return LPSolver(env)
    elif key == "argmax_solver":
        candidate_policies = get_dict_default(kwargs, "candidate_policies", None)
        return ArgmaxSolver(
            env,
            n_eval=100,
            candidate_policies=candidate_policies,
        )
    elif key == "lbfgsb_solver":
        return LBFGSSolver(env)
    elif key == "lbfgsb_argmax_solver":
        solver_policies_file = get_dict_default(kwargs, "solver_policies_file", None)
        debug = get_dict_default(kwargs, "debug", None)
        return LBFGSArgmaxSolver(
            env,
            solver_policies_file=solver_policies_file,
            debug=debug,
        )
    elif key == "ppo":
        return StableBaselinesSolver(env, PPO2, CommonMlpPolicy)
    elif key == "ppo_cnn":
        return StableBaselinesSolver(env, PPO2, CommonCnnPolicy)
    elif key == "sac":
        return StableBaselinesSolver(env, SAC, SACMlpPolicy)
    elif key == "sac_cnn":
        return StableBaselinesSolver(env, SAC, SACCnnPolicy)
    elif key == "dqn":
        return StableBaselinesSolver(env, DQN, DQNMlpPolicy)
    elif key == "dqn_cnn":
        return StableBaselinesSolver(env, DQN, DQNCnnPolicy)
    elif key == "ars":
        # args = {
        #     "horizon": 1000,
        #     "step_size": 0.04,
        #     "noise": 0.06,
        #     "n_directions": 16,
        #     "b": 16,
        #     "n_jobs": 4,
        # }
        #
        # if "Maze" in env.spec.id:
        #     args["n_jobs"] = 1
        #     args["step_size"] = 0.1
        #     args["normalize"] = False

        if "Hierarchical" in env.spec.id and "Ant" in env.spec.id:
            args = {
                "horizon": 1000,  # only used if env doesn't terminate
                "step_size": 0.3,
                "noise": 0.2,
                "n_directions": 8,
                "b": 8,
                "n_jobs": 1,
                "normalize": False,
            }
        elif "Hierarchical" in env.spec.id:
            # higher step size and noise
            args = {
                "horizon": 1000,  # only used if env doesn't terminate
                "step_size": 0.3,
                "noise": 0.2,
                "n_directions": 8,
                "b": 8,
                "n_jobs": 1,
                "normalize": False,
            }
        elif "Ant" in env.spec.id:
            args = {
                "horizon": 1000,  # only used if env doesn't terminate
                "step_size": 0.02,
                "noise": 0.03,
                "n_directions": 32,
                "b": 8,
                "n_jobs": 1,
                "normalize": True,
            }
        elif "Goal" in env.spec.id:
            # lower step size and noise
            args = {
                "horizon": 1000,  # only used if env doesn't terminate
                "step_size": 0.02,
                "noise": 0.03,
                "n_directions": 32,
                "b": 8,
                "n_jobs": 1,
                "normalize": False,
            }
        else:
            # lower step size and noise
            args = {
                "horizon": 1000,  # only used if env doesn't terminate
                "step_size": 0.04,
                "noise": 0.06,
                "n_directions": 16,
                "b": 16,
                "n_jobs": 1,
                "normalize": False,
            }

        # args = {
        #     "horizon": 1000,
        #     "step_size": 0.02,
        #     "noise": 0.03,
        #     "n_directions": 16,
        #     "b": 16,
        #     "n_jobs": 4,
        # }

        # args = {
        #     "horizon": 1000,
        #     "step_size": 0.02,
        #     "n_directions": 8,
        #     "b": 4,
        #     "noise": 0.01,
        #     "n_jobs": 8,
        # }

        if "n_jobs" in kwargs:
            args["n_jobs"] = kwargs["n_jobs"]
        return AugmentedRandomSearchSolver(env, **args)
    else:
        raise KeyError("Unknown solver key")


def get_observations_required_for_solver(key):
    if key == "lp_solver" or key == "tabular_q_learning" or key == "argmax_solver":
        return "state"
    elif key == "linear_q_learning":
        return "features"
    elif key in ("ppo", "ppo_cnn", "sac", "sac_cnn", "dqn", "dqn_linear", "ars"):
        return "raw"
    else:
        raise KeyError("Unknown solver key")
