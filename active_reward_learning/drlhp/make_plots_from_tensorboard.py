import argparse
import os
import sys
from collections import defaultdict

import gym
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import active_reward_learning
from active_reward_learning.util.helpers import get_dict_default
from active_reward_learning.util.plotting import plot_result_percentiles, set_plot_style
from active_reward_learning.util.results import FileExperimentResults

AF_COLORS = {
    "random": "#4daf4a",
    "variance": "#984ea3",
    "true reward": "#377eb8",
    "idrl": "#e41a1c",
    "idrl_m1000_p100_rollout_cand": "#e41a1c",
    "idrl_m1000_p100": "#1a81e4",
}

AF_MARKERS = {
    "random": "s",
    "variance": "^",
    "true reward": "x",
    "idrl": "o",
    "idrl_m1000_p100_rollout_cand": "o",
    "idrl_m1000_p100": "v",
}

AF_LINESTYLE = {
    "random": "-",
    "variance": "-",
    "true reward": "-",
    "idrl": "-",
    "idrl_m1000_p100_rollout_cand": "-",
    "idrl_m1000_p100": "--",
}

# AF_COLORS = defaultdict(lambda: "blue", AF_COLORS)
NEW_COLOR_COUNT = 0
NEW_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]

AF_ALPHA = {
    "idrl": 1.0,
    "idrl_m1000_p100": 0.6,
    "idrl_m1000_p100_rollout_cand": 1.0,
}
AF_ALPHA = defaultdict(lambda: 0.6, AF_ALPHA)  # 0.6 before

AF_ZORDER = {
    "idrl": 2,
    "idrl_m1000_p100": 1,
    "idrl_m1000_p100_rollout_cand": 2,
}
AF_ZORDER = defaultdict(lambda: 1, AF_ZORDER)


# returns are averaged over 1000 episodes
RANDOM_POLICY_RETURNS = {
    "HalfCheetah-Long-v3": -287.3389870083097,
    "Hopper-Long-v3": -46.1818188934309,
    "Walker2d-Long-v3": -6.772750336937592,
    "Swimmer-Long-v3": 0.5256574172271935,
    "Ant-Long-v3": -349.3281947598893,
    "Reacher-v2": -43.23157686737403,
    "InvertedPendulum-Penalty-Long-v2": -1447.4255937526677,
    "InvertedDoublePendulum-Penalty-Long-v2": -8020.161758098083,
}

EXPERT_POLICY_RETURNS = {
    "HalfCheetah-Long-v3": 14477.754,
    "Hopper-Long-v3": 3714.0933,
    "Walker2d-Long-v3": 5563.308,
    "Swimmer-Long-v3": 95.490906,
    "Ant-Long-v3": 7604.946,
    "Reacher-v2": -3.4786558,
    "InvertedPendulum-Penalty-Long-v2": 999.92804,
    "InvertedDoublePendulum-Penalty-Long-v2": 11316.3,
}


def read_tb_events(dirname):
    """Read a TensorBoard event file.
    Args:
        dirname (str): Path to Tensorboard log
    Returns:
        dict: A dictionary of containing scalar run data with keys like
            'train/loss', 'train/mae', 'val/loss', etc.
    """
    summary_iterator = EventAccumulator(dirname).Reload()
    tags = summary_iterator.Tags()["scalars"]
    out_dict = {t: {"steps": [], "values": []} for t in tags}
    for tag in tags:
        events = summary_iterator.Scalars(tag)
        for e in events:
            out_dict[tag]["steps"].append(e.step)
            out_dict[tag]["values"].append(e.value)
        out_dict[tag]["steps"] = np.array(out_dict[tag]["steps"])
        out_dict[tag]["values"] = np.array(out_dict[tag]["values"])

    return out_dict


def add_results(tb_logs_dict, env_id, af, result):
    if env_id not in tb_logs_dict:
        tb_logs_dict[env_id] = dict()
    if af not in tb_logs_dict[env_id]:
        tb_logs_dict[env_id][af] = []
    tb_logs_dict[env_id][af].append(result)
    return tb_logs_dict


def get_af_label(experiment):
    af = experiment.config["acquisition_function"]

    schedule_update_every = get_dict_default(
        experiment.config, "schedule_update_every", None
    )
    n_model_updates = get_dict_default(experiment.config, "n_model_updates", None)
    n_policy_updates = get_dict_default(experiment.config, "n_policy_updates", None)
    exploration_model_path = get_dict_default(
        experiment.config, "exploration_model_path", None
    )
    exploration_sigma = get_dict_default(experiment.config, "exploration_sigma", None)
    exploration_eps = get_dict_default(experiment.config, "exploration_eps", None)
    rollout_cand = get_dict_default(
        experiment.config, "rollout_candidate_policies_for_exploration", False
    )
    reinit = get_dict_default(
        experiment.config, "reinitialize_candidate_policies", False
    )
    af_label = af

    if schedule_update_every is not None:
        total_timesteps = get_dict_default(experiment.config, "total_timesteps", 0)
        n_model_updates = int(total_timesteps // schedule_update_every)
    if n_model_updates is not None:
        af_label += f"_m{n_model_updates}"
    if n_policy_updates is not None:
        af_label += f"_p{n_policy_updates}"
    if exploration_model_path is not None:
        af_label += f"_expl_"
    if exploration_sigma is not None:
        af_label += f"_sigma_{exploration_sigma}"
    if exploration_eps is not None:
        af_label += f"_eps_{exploration_eps}"
    if rollout_cand:
        af_label += f"_rollout_cand_"
    if reinit:
        af_label += f"_reinit"

    return af_label.strip("_")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tb_folder", type=str)
    parser.add_argument("--plot_folder", type=str, default="plots/drlhp")
    parser.add_argument("--csv_baseline", type=str, default=None)
    parser.add_argument("--baseline_label", type=str, default="Baseline")
    parser.add_argument("--mean", action="store_true")
    parser.add_argument("--stderr", action="store_true")
    parser.add_argument("--average_envs", action="store_true")
    parser.add_argument("--results_folder", type=str, default=None)
    parser.add_argument("--aspect", type=float, default=1.15)
    parser.add_argument("--n_markers", type=int, default=10)
    parser.add_argument("--marker_size", type=int, default=11)
    parser.add_argument("--no_true_reward_plot", action="store_true")
    parser.add_argument("--no_legend", action="store_true")
    parser.add_argument("--no_title", action="store_true")
    parser.add_argument("--paper_style", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.plot_folder, exist_ok=True)

    tb_logs_dict = dict()

    if args.results_folder is not None:
        for x in os.walk(args.results_folder):
            subdir = x[0]
            try:
                experiment = FileExperimentResults(subdir)
            except:
                print("Could not open", subdir)
                continue
            if experiment.info is not None:
                tb_log_path = os.path.basename(experiment.info["tb_log"])
                tb_log_path = os.path.join(args.tb_folder, tb_log_path)
                if os.path.exists(tb_log_path):
                    af_label = get_af_label(experiment)
                    env_id = experiment.config["env_id"]
                    tb_logs_dict = add_results(
                        tb_logs_dict, env_id, af_label, tb_log_path
                    )
                else:
                    print("tb_log does not exist", tb_log_path)
            else:
                print("Could not read info of", subdir)

    for x in os.walk(args.tb_folder):
        subdir = x[0]
        print(subdir)

        dir = os.path.basename(subdir)
        if dir.startswith("tb_log"):
            if args.results_folder is not None:
                continue
            else:
                dir = dir[3:]
        elif dir.startswith("sac") and not args.no_true_reward_plot:
            if args.results_folder is not None:
                env_id = dir.split("_")[1]
                if not env_id in tb_logs_dict:
                    # only load true reward runs that are in results folder as well (if given)
                    continue
        else:
            continue

        dir = dir.split("_")

        env_id = dir[1]
        if dir[0] == "sac":
            af = "true reward"
        else:
            af = dir[2]

        tb_logs_dict = add_results(tb_logs_dict, env_id, af, subdir)

    for env_id in tb_logs_dict.keys():
        print(env_id)
        for af in tb_logs_dict[env_id].keys():
            print(" " * 4, af)
            print(" " * 8, tb_logs_dict[env_id][af])

    results = dict()

    for env_id in tb_logs_dict.keys():
        print(env_id)
        results[env_id] = dict()

        env = gym.make(env_id)
        L = env.spec.max_episode_steps

        for af in tb_logs_dict[env_id].keys():
            print(" " * 4, af)
            steps_list, rew_list = [], []
            for path in set(tb_logs_dict[env_id][af]):
                print(" " * 8, path)
                data = read_tb_events(path)

                if af == "true reward":
                    if "rollout/ep_rew_mean" in data and not args.no_true_reward_plot:
                        steps = data["rollout/ep_rew_mean"]["steps"]
                        rew = data["rollout/ep_rew_mean"]["values"]
                    else:
                        continue
                else:
                    if "rollout/ep_avg_true_ret_mean" in data:
                        steps = data["rollout/ep_avg_true_ret_mean"]["steps"]
                        rew = data["rollout/ep_avg_true_ret_mean"]["values"]
                    elif "rollout/ep_avg_true_rew_mean" in data:
                        ## old runs logged average per step reward (new ones record return)
                        steps = data["rollout/ep_avg_true_rew_mean"]["steps"]
                        rew = data["rollout/ep_avg_true_rew_mean"]["values"] * L
                    else:
                        continue

                # DL: this fixes an issue where multiple tensorboard logs are written
                # to the same log
                # this should be fixed in the drlhp code now, but it is
                # still a problem in some of the results
                if np.max(steps) > 1e2:
                    idx = []
                    for i in range(1, len(steps)):
                        if steps[i] >= steps[i - 1]:
                            idx.append(i)
                        else:
                            steps_list.append(steps[idx])
                            rew_list.append(rew[idx])
                            idx = [i]
                    if len(idx) > 1:
                        steps_list.append(steps[idx])
                        rew_list.append(rew[idx])

            if len(steps_list) > 0 and len(rew_list) > 0:
                results[env_id][af] = (steps_list, rew_list)

    if args.average_envs:
        envs = results.keys()
        afs = [set(results[env_id].keys()) for env_id in results.keys()]
        all_afs = afs[0].union(*afs[1:])
        afs = afs[0].intersection(*afs[1:])
        min_returns = dict()

        new_results = {"All-Envs": dict()}
        for af in afs:
            steps_list_all, rew_list_all = [], []
            for env_id in envs:
                random_return = RANDOM_POLICY_RETURNS[env_id]
                expert_return = EXPERT_POLICY_RETURNS[env_id]
                steps_list, rew_list = results[env_id][af]
                new_rew_list = [
                    100 * (rewards - random_return) / (expert_return - random_return)
                    for rewards in rew_list
                ]

                steps_list_all += steps_list
                rew_list_all += new_rew_list
            new_results["All-Envs"][af] = (steps_list_all, rew_list_all)
        results = new_results

    if args.paper_style:
        set_plot_style()

    for env_id in results.keys():
        height, width = plt.figaspect(args.aspect)
        fig = plt.figure(figsize=(width, height))
        legend_handles = []
        legend_labels = []

        for af in results[env_id].keys():
            steps_list, rew_list = results[env_id][af]
            if af not in AF_COLORS:
                af_short = af.split("_")[0]
                afs_short = [af_.split("_")[0] for af_ in results[env_id].keys()]
                if afs_short.count(af_short) == 1 and af_short in AF_COLORS:
                    # af only used with one set of parameters
                    # (not the case during hyperparameter tuning)
                    af = af_short
                else:
                    global NEW_COLOR_COUNT
                    AF_COLORS[af] = NEW_COLORS[NEW_COLOR_COUNT % len(NEW_COLORS)]
                    NEW_COLOR_COUNT += 1

            color = get_dict_default(AF_COLORS, af, "b")
            linestyle = get_dict_default(AF_LINESTYLE, af, "-")

            if args.n_markers > 0:
                marker = get_dict_default(AF_MARKERS, af, None)
                markers_every = len(steps_list[0]) // args.n_markers
            else:
                marker = None
                markers_every = 1

            if args.mean:
                l = min(min(map(len, steps_list)), min(map(len, rew_list)))
                steps_list = [s[:l] for s in steps_list]
                rew_list = [r[:l] for r in rew_list]
                steps = steps_list[0]  # np.mean(steps_list, 0)
                rew = np.mean(rew_list, 0)
                all_steps_same = np.all(
                    [
                        np.all(steps_list[i] == steps_list[0])
                        for i in range(1, len(steps_list))
                    ]
                )
                label = f"{af}_mean_over_{len(steps_list)}"
                if args.paper_style:
                    label = label.replace("_", "\_")
                if not all_steps_same:
                    print("Warning: not all steps are the same")
                p1 = plt.plot(
                    steps,
                    rew,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markevery=markers_every,
                    markersize=args.marker_size,
                    alpha=AF_ALPHA[af],
                    zorder=AF_ZORDER[af],
                )[0]
                if args.stderr:
                    stderr = np.std(rew_list, 0) / np.sqrt(len(rew_list))
                    p2 = plt.fill_between(
                        steps,
                        rew - stderr,
                        rew + stderr,
                        color=color,
                        alpha=0.2,
                        zorder=AF_ZORDER[af],
                    )
                    legend_handles.append((p1, p2))
                else:
                    legend_handles.append(p1)
            else:
                for i, (steps, rew) in enumerate(zip(steps_list, rew_list)):
                    label = f"{af}_{i}"
                    if args.paper_style:
                        label = label.replace("_", "\_")
                    p1 = plt.plot(
                        steps,
                        rew,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markevery=markers_every,
                        markersize=args.marker_size,
                        alpha=AF_ALPHA[af],
                        zorder=AF_ZORDER[af],
                    )[0]
                    legend_handles.append(p1)
            legend_labels.append(label)

        if args.csv_baseline is not None:
            baseline_data = np.genfromtxt(args.csv_baseline, delimiter=",")
            baseline = np.max(baseline_data[:, 1])
            if args.average_envs:
                baseline /= best_return
            plt.axhline(baseline, color="black", linestyle="--")

        # plt.xlim(0, 100000)
        plt.xlabel("timestep")
        if args.average_envs:
            plt.ylabel("average score")
        else:
            plt.ylabel("return")

        if not args.no_title:
            plt.title(env_id)
        if not args.no_legend:
            plt.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )

        if args.pdf:
            plt.savefig(
                os.path.join(args.plot_folder, f"{env_id}.pdf"), bbox_inches="tight"
            )
        else:
            plt.savefig(
                os.path.join(args.plot_folder, f"{env_id}.png"), bbox_inches="tight"
            )


if __name__ == "__main__":
    main()
