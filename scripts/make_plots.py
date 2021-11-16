"""
This is a helper script to create plots from sacred results.
Parameter values are hardcoded in the script for now.
"""

import argparse
import datetime
import os
import pickle
import subprocess
import time
from collections import defaultdict
from functools import partial
from textwrap import wrap

import gym
import matplotlib.pyplot as plt
import numpy as np
from frozendict import frozendict
from matplotlib.ticker import MaxNLocator

from active_reward_learning.common.constants import (
    AF_ALPHA,
    AF_COLORS,
    AF_MARKERS,
    AF_ZORDER,
    PLOTS_PATH,
    color_cycle,
    hatch_cycle,
    marker_cycle,
)
from active_reward_learning.util.helpers import (
    get_acquisition_function_label,
    get_acquisition_function_label_clean,
)
from active_reward_learning.util.plotting import (
    plot_result_max,
    plot_result_percentiles,
    set_plot_style,
)
from active_reward_learning.util.results import FileExperimentResults


def load(results_folder, experiment_label, config_query=None):
    if experiment_label is not None:
        result = subprocess.check_output(
            f"grep '{experiment_label}' -r {results_folder} "
            "| grep config | cut -f 1 -d : | rev | cut -d / -f 2- | rev",
            shell=True,
        ).decode()
        subdirs = result.split("\n")
    else:
        subdirs = [x[0] for x in os.walk(results_folder)]
    experiments = []
    for i, subdir in enumerate(subdirs):
        print(i, subdir)
        try:
            experiment = FileExperimentResults(subdir)
        except Exception as e:
            # print(e)
            continue
        valid = True
        if config_query is not None:
            for key, value in config_query.items():
                if experiment.config[key] != value:
                    # print(f"{key}, {experiment.config[key]}, {value}")
                    valid = False
        if valid:
            experiments.append(experiment)
    return experiments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_folder", type=str, default=None)
    parser.add_argument("--remote_results_folder", type=str, default=None)
    parser.add_argument("--tmp_folder", type=str, default="/tmp")
    parser.add_argument("--experiment_label", type=str, default=None)
    parser.add_argument("--paper_style", action="store_true")
    parser.add_argument("--test_plots_only", action="store_true")
    parser.add_argument("--forbid_running", action="store_true")
    parser.add_argument("--allow_failed", action="store_true")
    parser.add_argument("--allow_interrupted", action="store_true")
    parser.add_argument("--no_title", action="store_true")
    parser.add_argument("--output_format", type=str, default="pdf")
    parser.add_argument("--xmin", type=int, default=1)
    parser.add_argument("--xmax", type=int, default=30)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--return_hline", type=float, default=None)
    parser.add_argument("--show_evaluation_return", action="store_true")
    parser.add_argument("--report_failures", action="store_true")
    parser.add_argument("--no_markers", action="store_true")
    parser.add_argument("--histograms_at", type=int, default=11)
    parser.add_argument("--mean", action="store_true")
    parser.add_argument("--stddev", action="store_true")
    parser.add_argument("--stderr", action="store_true")
    parser.add_argument("--xlabel", type=str, default=None)
    parser.add_argument("--ylabel", type=str, default=None)
    parser.add_argument("--aspect", type=float, default=1.15)
    parser.add_argument("--nolegend", action="store_true")
    parser.add_argument("--markers_every", type=int, default=1)
    parser.add_argument("--xticks_every", type=int, default=10)
    parser.add_argument("--use_af_colors_markers", action="store_true")
    parser.add_argument("--recompute_regret", action="store_true")
    parser.add_argument("--no_yaxis", action="store_true")
    parser.add_argument("--fix_axis_size", action="store_true")
    return parser.parse_args()


def get_next_style(color, hatch, marker):
    color = (color + 1) % len(color_cycle)
    hatch = (hatch + 1) % len(hatch_cycle)
    marker = (marker + 1) % len(marker_cycle)
    return color, hatch, marker


def main():
    global get_acquisition_function_label

    args = parse_args()

    if args.results_folder is None and args.remote_results_folder is None:
        raise ValueError("results_folder or remote_results_folder has to be given")

    if args.remote_results_folder is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tmp_result_folder = os.path.join(args.tmp_folder, timestamp)
        subprocess.run(
            [
                "rsync",
                "-av",
                "-e ssh",
                "--exclude",
                "*.txt",
                "--exclude",
                "*.pkl",
                args.remote_results_folder,
                tmp_result_folder,
            ]
        )
        args.results_folder = tmp_result_folder

    if args.paper_style:
        set_plot_style()

    print(f"Loading '{args.experiment_label}' from '{args.results_folder}'.")

    # list mdps
    all_experiments = load(
        args.results_folder, experiment_label=args.experiment_label, config_query=None
    )
    print("Found {} experiments.".format(len(all_experiments)))

    all_experiments_by_mdp_af = dict()
    all_acquisition_functions = set()

    mdp_dict = dict()
    mdps = []
    eval_returns = defaultdict(lambda: [])
    for ex in all_experiments:
        try:
            mdp = ex.config["mdp"]
        except:
            continue

        if "use_comparisons" in ex.config:
            mdp["use_comparisons"] = ex.config["use_comparisons"]

        mdp_str = str(mdp)
        if mdp_str not in mdp_dict:
            mdp_dict[mdp_str] = dict()
            mdps.append(mdp)

        if ex.info is not None:
            eval_return = ex.info["evaluation_policy_return"]
            eval_returns[mdp_str].append(eval_return)
        else:
            print("Warning: ex.info is none")

        af = dict(ex.config["acquisition_function"])
        if "get_evaluation_set_from" in ex.config:
            af["get_evaluation_set_from"] = ex.config["get_evaluation_set_from"]
        else:
            af["get_evaluation_set_from"] = ex.config["get_evaluation_states_from"]

        af_label = get_acquisition_function_label(af)
        print(af)
        print(af_label)
        af_frozen = frozendict(af)
        all_acquisition_functions.add(af_frozen)
        af_str = str(af)
        if af_label not in mdp_dict[mdp_str]:
            mdp_dict[mdp_str][af_label] = [0, 0]
        mdp_dict[mdp_str][af_label][0] += 1
        if ex.status == "COMPLETED":
            mdp_dict[mdp_str][af_label][1] += 1
        idx = (frozenset(mdp.items()), frozenset(af_frozen.items()))
        if idx in all_experiments_by_mdp_af:
            all_experiments_by_mdp_af[idx].append(ex)
        else:
            all_experiments_by_mdp_af[idx] = [ex]

    for mdp_str in mdp_dict.keys():
        print(mdp_str)
        for af, [count, completed] in mdp_dict[mdp_str].items():
            print(f"\t{af}   COUNT {count}   COMPLETED {completed}")
        eval_mean = np.mean(eval_returns[mdp_str])
        eval_std = np.std(eval_returns[mdp_str])
        print(f"Eval returns: {eval_mean} +- {eval_std}")

    def clean_mdp(mdp):
        new_mdp = dict(mdp)
        if new_mdp["label"].startswith("gridworld"):
            if "robots" in new_mdp:
                del new_mdp["robots"]
            if "use_feature_representation" in new_mdp:
                del new_mdp["use_feature_representation"]
        return new_mdp

    #####################################
    af_to_plot = sorted(
        all_acquisition_functions, key=lambda x: get_acquisition_function_label(x)
    )
    #####################################

    # mdps = [
    #     {
    #         "label": "minecraft",
    #         "gym_label": "MinecraftGridworld10x10WoodenFeaturesFixed-v0",
    #         "observation_noise": 0.1,
    #     },
    #     {
    #         "label": "minecraft",
    #         "gym_label": "MinecraftGridworld10x10WoodenWallsFeaturesFixed-v0",
    #         "observation_noise": 0.1,
    #     },
    # ]
    # mdps = [
    #     {
    #         "label": "gridworld_point",
    #         "discount_factor": 0.99,
    #         "episode_length": 1000,
    #         "width": 20,
    #         "height": 20,
    #         "n_objects": 20,
    #         "n_per_object": 5,
    #         "wall_probability": 0.3,
    #         "random_action_prob": 0.3,
    #         "add_optimal_policy_to_candidates": False,
    #         "observation_noise": 0.1,
    #     }
    # ]

    # comment out for more informative labels in the legend
    # get_acquisition_function_label = get_acquisition_function_label_clean

    get_acquisition_function_label = partial(
        get_acquisition_function_label, latex=args.paper_style
    )

    # METRICS TO PLOT
    metrics = [
        "cand_policy_for_mean_of_model_return",
        "cand_mse",
        "epic",
        "return",
        "regret",
        # "worst_regret",
        "cand_policy_returns_min",
        "cand_policy_returns_max",
        "cand_policy_returns_median",
        # "bayes_expected_return",
        # "hist_cand_policy_for_mean_of_model_return",
        # "cand_policy_for_mean_of_model_inferred_return",
        # # "selected_policy_return",
        # # "selected_policy_regret",
        # # "cand_log_likelihood",
        # "cand_mape",
        # "cand_total_var",
        # "cand_correct_sign_perc",
        # # "cand_log_likelihood_rff",
        # "cand_mse_rff",
        # "cand_mape_rff",
        # # "cand_total_var_rff",
        # # "n_unique_candidate_queries",
        # # "candidate_policy_mean_jaccard_dist",
        # # "cand_policy_returns_var",
        "learned_linear_weight_mse",
        "learned_linear_weight_alignment",
    ]  # + [f"learned_linear_weight_{i}" for i in range(50)]

    if args.test_plots_only:
        metrics_by_mdp_af = dict()
        for mdp in mdps:
            for af in af_to_plot:
                idx = (frozenset(mdp.items()), frozenset(af.items()))
                metrics_by_mdp_af[idx] = dict()
                for metric in metrics:
                    metrics_by_mdp_af[idx][metric] = [
                        ([1, 2, 3], [1, 2, 3]),
                        ([1, 2, 3], [1, 2, 2]),
                        ([1, 2, 3], [1.5, 2.5, 3.5]),
                    ]
    else:
        metrics_by_mdp_af = dict()
        evaluation_return_by_mdp = dict()
        for mdp in mdps:
            print("Collecting results for {}".format(mdp))

            for af in af_to_plot:
                print(f"\t{af}")
                mdp_frozen = frozenset(mdp.items())
                af_frozen = frozenset(af.items())
                idx = (mdp_frozen, af_frozen)
                if idx not in all_experiments_by_mdp_af:
                    print("Warning experiment unavailable:", idx)
                    continue

                metrics_by_mdp_af[idx] = dict()
                evaluation_return_by_mdp[mdp_frozen] = []
                experiments = all_experiments_by_mdp_af[idx]
                for ex in experiments:
                    if args.show_evaluation_return:
                        if ex.info is not None:
                            evaluation_return_by_mdp[mdp_frozen].append(
                                ex.info["evaluation_policy_return"]
                            )
                    if args.report_failures and ex.status == "FAILED":
                        print()
                        print(ex.run["fail_trace"])
                        print()

                    if (
                        ex.status == "COMPLETED"
                        or (not args.forbid_running and ex.status == "RUNNING")
                        or (args.allow_failed and ex.status == "FAILED")
                        or (args.allow_interrupted and ex.status == "INTERRUPTED")
                    ):
                        for metric in metrics:
                            if metric.startswith("worst_"):
                                metric = metric[6:]
                            if ex.metrics is not None and (
                                metric in ex.metrics
                                or (
                                    metric == "regret"
                                    and args.recompute_regret
                                    and (
                                        "return" in ex.metrics
                                        or "cand_policy_for_mean_of_model_return"
                                        in ex.metrics
                                    )
                                )
                                or (
                                    metric == "learned_linear_weight_alignment"
                                    and "learned_linear_weight_0" in ex.metrics
                                )
                            ):
                                if metric not in metrics_by_mdp_af[idx]:
                                    metrics_by_mdp_af[idx][metric] = {
                                        "steps": [],
                                        "values": [],
                                    }

                                if metric == "regret" and args.recompute_regret:
                                    if "return" in ex.metrics:
                                        steps, values = ex.get_metric("return")
                                    else:
                                        steps, values = ex.get_metric(
                                            "cand_policy_for_mean_of_model_return"
                                        )
                                    max_return1 = np.max(eval_returns[str(mdp)])
                                    max_return2 = np.max(values)
                                    max_return = max(max_return1, max_return2)
                                    values = max_return - np.array(values)
                                elif (metric == "learned_linear_weight_alignment") and (
                                    metric not in ex.metrics
                                ):
                                    # hack to get alignment even if it was not computed
                                    # before
                                    env = gym.make(mdp["gym_label"])
                                    w_true = env.reward_w
                                    w_learned = []
                                    for w_i in range(len(w_true)):
                                        steps, values = ex.get_metric(
                                            f"learned_linear_weight_{w_i}"
                                        )
                                        w_learned.append(values)
                                    w_learned = np.array(w_learned).T
                                    values = []
                                    for w_i in w_learned:
                                        alignment = np.dot(w_i, w_true)
                                        alignment /= np.linalg.norm(w_i)
                                        alignment /= np.linalg.norm(w_true)
                                        values.append(alignment)
                                    values = np.array(values)
                                else:
                                    steps, values = ex.get_metric(metric)
                                    steps, values = np.array(steps), np.array(values)

                                metrics_by_mdp_af[idx][metric]["steps"].append(steps)
                                metrics_by_mdp_af[idx][metric]["values"].append(values)
                            else:
                                print(
                                    "WARNING: metric '{}' missing for experiment".format(
                                        metric
                                    )
                                )
                    else:
                        print(
                            "WARNING: Skipped experiment with status '{}'.".format(
                                ex.status
                            )
                        )

    for mdp in mdps:
        for metric in metrics:
            if metric.startswith("worst_"):
                metric = metric[6:]
                plot_worst = True
            else:
                plot_worst = False

            if metric.startswith("hist_"):
                if args.histograms_at is None:
                    continue
                metric = metric[5:]
                plot_hist = True
            else:
                plot_hist = False

            any_result = False
            make_legend = "no"
            print("Creating plots of {} for {}".format(metric, mdp))
            fig_height, fig_width = plt.figaspect(args.aspect)
            fig = plt.figure(figsize=(fig_width, fig_height))
            color, hatch, marker = 0, 0, 0
            legend_handles, legend_labels = [], []

            n = 0
            for af in af_to_plot:
                if (
                    "observation_batch_size" in af
                    and af["observation_batch_size"] is not None
                ):
                    observation_batch_size = af["observation_batch_size"]
                else:
                    observation_batch_size = 1

                af_label = get_acquisition_function_label(af)
                mdp_frozen = frozenset(mdp.items())
                af_frozen = frozenset(af.items())
                idx = (mdp_frozen, af_frozen)
                if idx not in metrics_by_mdp_af:
                    print(f"{af_label} not in metrics_by_mdp_af")
                    continue
                n += 1
                print("label {}           n {}".format(af_label, n))

                if metric in metrics_by_mdp_af[idx] and not None in np.array(
                    metrics_by_mdp_af[idx][metric]["values"]
                ):
                    steps = np.array(metrics_by_mdp_af[idx][metric]["steps"])
                    steps *= observation_batch_size
                    values = metrics_by_mdp_af[idx][metric]["values"]
                    any_result = True
                    make_legend = "full"

                    if args.use_af_colors_markers:
                        afl = af_label.split("_")[0].split("\\")[0]
                        plot_color = AF_COLORS[afl]
                        plot_marker = AF_MARKERS[afl]
                        plot_alpha = AF_ALPHA[afl]
                        plot_zorder = AF_ZORDER[afl]
                    else:
                        plot_color = color_cycle[color]
                        plot_marker = marker_cycle[marker]
                        plot_alpha = 1
                        plot_zorder = 1

                    if plot_hist:
                        hist_values = []
                        for i in range(len(steps)):
                            for j in range(len(steps[i])):
                                if steps[i][j] == args.histograms_at:
                                    hist_values.append(values[i][j])

                        plt.hist(
                            hist_values,
                            label=af_label,
                            density=True,
                            bins=10,
                            alpha=0.8,
                        )
                        make_legend = "simple"
                    elif plot_worst:
                        legend_handles, legend_labels = plot_result_max(
                            steps,
                            values,
                            af_label,
                            plot_color,
                            hatch,
                            plot_marker,
                            legend_handles,
                            legend_labels,
                            no_markers=args.no_markers,
                            alpha=plot_alpha,
                            zorder=plot_zorder,
                        )
                    else:
                        legend_handles, legend_labels = plot_result_percentiles(
                            steps,
                            values,
                            af_label,
                            plot_color,
                            hatch,
                            plot_marker,
                            legend_handles,
                            legend_labels,
                            fix_negative=(
                                metric == "regret" or metric == "selected_policy_regret"
                            ),
                            no_markers=args.no_markers,
                            markers_every=args.markers_every,
                            plot_mean=args.mean,
                            plot_stddev=args.stddev,
                            plot_stderr=args.stderr,
                            alpha=plot_alpha,
                            zorder=plot_zorder,
                        )
                    color, hatch, marker = get_next_style(color, hatch, marker)

            if any_result:
                if args.nolegend:
                    make_legend = "no"
                if make_legend == "full":
                    # sort both labels and handles by labels
                    legend_labels, legend_handles = zip(
                        *sorted(zip(legend_labels, legend_handles), key=lambda t: t[0])
                    )

                    if not args.paper_style:
                        if metric == "regret" or "mse" in metric or "mape" in metric:
                            # uppper right
                            legend_kwargs = {
                                "loc": "center left",
                                "bbox_to_anchor": (0.5, 0.8),
                            }
                        else:
                            # lower right
                            legend_kwargs = {
                                "loc": "center left",
                                "bbox_to_anchor": (0.5, 0.2),
                            }
                    else:
                        legend_kwargs = dict()

                    leg = plt.legend(
                        legend_handles,
                        legend_labels,
                        # fontsize=5,
                        # handlelength=2,
                        # prop={"size": 20},
                        **legend_kwargs,
                    )
                elif make_legend == "simple":
                    plt.legend()

                if not args.no_title:
                    plt.title(
                        "\n".join(wrap(str(clean_mdp(mdp)), 120)),
                        fontdict={"fontsize": 6},
                    )

                if not plot_hist:
                    plt.xlim(args.xmin, args.xmax)

                    if args.ymin is not None and args.ymax is not None:
                        plt.ylim(args.ymin, args.ymax)

                if metric in ("return", "cand_policy_for_mean_of_model_return"):
                    if args.return_hline is not None:
                        plt.axhline(args.return_hline)
                    elif args.show_evaluation_return:
                        plt.axhline(np.mean(evaluation_return_by_mdp[mdp_frozen]))

                if metric.startswith("learned_linear_weight") and not metric.endswith(
                    ("mse", "alignment")
                ):
                    env = gym.make(mdp["gym_label"])
                    weight_i = int(metric.split("_")[-1])
                    w_true = env.reward_w
                    plt.axhline(w_true[weight_i])
                    plt.ylim(-2, 2)

                # if "mse" in metric or "mape" in metric:
                # plt.axhline(0, color="g", linestyle="--")

                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

                if plot_hist:
                    metric = "hist_" + metric
                elif plot_worst:
                    metric = "worst_" + metric

                if plot_hist:
                    plt.xlabel(metric[5:] + f" at t = {args.histograms_at}")
                    plt.ylabel("density")
                else:
                    if args.paper_style:
                        plt.locator_params(
                            nbins=args.xmax // args.xticks_every - 1, axis="x"
                        )
                        plt.gca().tick_params(
                            axis="both",
                            direction="out",
                            bottom=True,
                            left=True,
                            top=False,
                            right=False,
                        )
                    plt.xticks()
                    plt.yticks()
                    if args.xlabel is None:
                        plt.xlabel("number of queries")
                    else:
                        plt.xlabel(args.xlabel)
                    if args.ylabel is None:
                        plt.ylabel(metric)
                    else:
                        plt.ylabel(args.ylabel)

                if args.no_yaxis:
                    plt.gca().axes.get_yaxis().set_visible(False)

                if args.fix_axis_size:
                    ax = plt.gca()
                    ax.set_box_aspect(args.aspect)
                    # fig_size = fig.get_size_inches()
                    # ax_height, ax_width = fig.gca().get_position().size * fig_size
                    # ax_target_width = ax_height*args.aspect
                    # fig_size[1] *= ax_target_width / ax_width
                    # fig.set_size_inches(fig_size)

                plt.tight_layout()

                # plt.show()
                filename = "__".join(
                    ["{}_{}".format(k, v) for k, v in clean_mdp(mdp).items()]
                ).replace(".", "_")
                filename += "." + args.output_format

                filename = "{}_{}".format(metric, filename)
                if args.experiment_label is not None:
                    foldername = args.experiment_label
                else:
                    foldername = "plots"
                folder_path = os.path.join(PLOTS_PATH, foldername)
                os.makedirs(folder_path, exist_ok=True)

                if args.output_format == "svg":
                    plt.rcParams["svg.fonttype"] = "none"

                full_path = os.path.join(folder_path, filename)
                print("Writing to", full_path)
                plt.savefig(
                    os.path.join(folder_path, filename), format=args.output_format
                )
            else:
                print("No results")

            del fig


if __name__ == "__main__":
    main()
