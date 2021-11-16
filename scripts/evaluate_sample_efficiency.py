"""
Evaluates number of samples needed to reach some return or regret threshold.
"""

import argparse
import datetime
import os
import pickle
import subprocess
import time
from functools import partial
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
from frozendict import frozendict
from matplotlib.ticker import MaxNLocator

from active_reward_learning.common.constants import (
    PLOTS_PATH,
    color_cycle,
    hatch_cycle,
    marker_cycle,
)
from active_reward_learning.util.helpers import (
    get_acquisition_function_label,
    get_acquisition_function_label_clean,
    get_swimmer_linear_reward_true_weight,
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
    parser.add_argument("--forbid_running", action="store_true")
    parser.add_argument("--allow_failed", action="store_true")
    parser.add_argument("--allow_interrupted", action="store_true")
    parser.add_argument("--return_percentage", type=float, default=1.0)
    parser.add_argument("--return_threshold", type=float, default=None)
    parser.add_argument("--regret_threshold", type=float, default=None)
    parser.add_argument("--sample_threshold", type=int, default=None)
    parser.add_argument("--xmax", type=int, default=None)
    parser.add_argument("--pickle_out_folder", type=str, default=None)
    return parser.parse_args()


def get_next_style(color, hatch, marker):
    color = (color + 1) % len(color_cycle)
    hatch = (hatch + 1) % len(hatch_cycle)
    marker = (marker + 1) % len(marker_cycle)
    return color, hatch, marker


def get_sample_n(steps, values, threshold, lower=False, perc=1):
    if lower:
        done = np.array(values) <= threshold
    else:
        done = np.array(values) >= threshold

    # print(1, done)
    # count how many in the future are done
    done = np.cumsum(done[::-1])[::-1]
    # print(2, done)
    # compute percentage of the future
    done = done / (np.arange(len(done))[::-1] + 1)
    # print(3, done)
    # compute when percentage is above desired value
    done = done >= perc
    # print(4, done)

    try:
        done_idx = int(np.argwhere(done)[0])
    except IndexError as e:
        done_idx = len(done) - 1

    print(done_idx, steps[done_idx], steps[-1])

    return steps[done_idx]


def get_metric(ex, name, xmax):
    steps, values, run_time = ex.get_metric(name, get_runtime=True)
    steps, values, run_time = np.array(steps), np.array(values), np.array(run_time)
    if xmax is None:
        return steps, values, run_time
    mask = steps <= xmax
    return steps[mask], values[mask], run_time[mask]


def main():
    global get_acquisition_function_label

    args = parse_args()

    assert (
        args.return_threshold is not None
        or args.regret_threshold is not None
        or args.sample_threshold is not None
    )

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
                "--exclude='*.txt'",
                "--exclude='*.pkl'",
                args.remote_results_folder,
                tmp_result_folder,
            ]
        )
        args.results_folder = tmp_result_folder

    print("Loading '{}' from '{}'.".format(args.experiment_label, args.results_folder))

    # list mdps
    all_experiments = load(args.results_folder, args.experiment_label, None)
    print("Found {} experiments.".format(len(all_experiments)))

    all_experiments_by_mdp_af = dict()
    all_acquisition_functions = set()

    mdp_dict = dict()
    mdps = []
    for ex in all_experiments:
        mdp = ex.config["mdp"]
        mdp_str = str(mdp)
        if mdp_str not in mdp_dict:
            mdp_dict[mdp_str] = dict()
            mdps.append(mdp)
        af = ex.config["acquisition_function"]
        all_acquisition_functions.add(frozendict(af))
        af_str = str(af)
        af_label = get_acquisition_function_label(af)
        if af_label not in mdp_dict[mdp_str]:
            mdp_dict[mdp_str][af_label] = [0, 0]
        mdp_dict[mdp_str][af_label][0] += 1
        if ex.status == "COMPLETED":
            mdp_dict[mdp_str][af_label][1] += 1
        idx = (frozenset(mdp.items()), frozenset(af.items()))
        if idx in all_experiments_by_mdp_af:
            all_experiments_by_mdp_af[idx].append(ex)
        else:
            all_experiments_by_mdp_af[idx] = [ex]

    for mdp_str in mdp_dict.keys():
        print(mdp_str)
        for af, [count, completed] in mdp_dict[mdp_str].items():
            print("\t{}   COUNT {}   COMPLETED {}".format(af, count, completed))

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

    # remove for more informative labels in the legend
    # get_acquisition_function_label = get_acquisition_function_label_clean
    get_acquisition_function_label = partial(get_acquisition_function_label)

    metrics_by_mdp_af = dict()
    evaluation_return_by_mdp = dict()
    results = dict()
    for mdp in mdps:
        print("Collecting results for {}".format(mdp))

        for af in af_to_plot:
            if (
                "observation_batch_size" in af
                and af["observation_batch_size"] is not None
            ):
                observation_batch_size = af["observation_batch_size"]
            else:
                observation_batch_size = 1

            print("\t{}".format(af))
            af_label = get_acquisition_function_label(af)
            mdp_frozen = frozenset(mdp.items())
            af_frozen = frozenset(af.items())
            idx = (mdp_frozen, af_frozen)
            if idx not in all_experiments_by_mdp_af:
                print("Warning experiment unavailable:", idx)
                continue

            experiments = all_experiments_by_mdp_af[idx]
            n_samples_return, n_samples_regret = [], []
            return_list, regret_list = [], []
            run_time_list = []
            for ex in experiments:
                print(ex.status)
                if (
                    ex.status == "COMPLETED"
                    or (not args.forbid_running and ex.status == "RUNNING")
                    or (args.allow_failed and ex.status == "FAILED")
                    or (args.allow_interrupted and ex.status == "INTERRUPTED")
                ):
                    print(ex.config["acquisition_function"]["label"])
                    try:
                        if args.return_threshold is not None:
                            if "return" in ex.metrics:
                                steps, values, run_time = get_metric(
                                    ex, "return", args.xmax
                                )
                            else:
                                steps, values, run_time = get_metric(
                                    ex,
                                    "cand_policy_for_mean_of_model_return",
                                    args.xmax,
                                )
                            steps *= observation_batch_size
                            n_samples = get_sample_n(
                                steps,
                                values,
                                args.return_threshold,
                                lower=False,
                                perc=args.return_percentage,
                            )
                            n_samples_return.append(n_samples)
                        if args.regret_threshold is not None:
                            steps, values, run_time = get_metric(
                                ex, "regret", args.xmax
                            )
                            if values[-1] > 0:
                                print(
                                    f"Warning: Last regret at it {len(values)-1} is > 0"
                                )
                                # import pdb; pdb.set_trace()
                            # print(values)
                            steps *= observation_batch_size
                            n_samples = get_sample_n(
                                steps,
                                values,
                                args.regret_threshold,
                                lower=True,
                                perc=args.return_percentage,
                            )
                            n_samples_regret.append(n_samples)
                        if args.sample_threshold is not None:
                            steps, values, run_time = get_metric(
                                ex, "return", args.xmax
                            )
                            steps *= observation_batch_size
                            return_list.extend(values[steps == args.sample_threshold])
                            steps, values, run_time = get_metric(
                                ex, "regret", args.xmax
                            )
                            steps *= observation_batch_size
                            regret_list.extend(values[steps == args.sample_threshold])
                            run_time_list.extend(
                                run_time[steps == args.sample_threshold] / 3600.0
                            )
                    except Exception as e:
                        print(e)

            if args.return_threshold is not None:
                n_samples_return_mean = np.mean(n_samples_return)
                n_samples_return_stderr = np.std(n_samples_return) / np.sqrt(
                    len(n_samples_return)
                )
            else:
                n_samples_return_mean, n_samples_return_stderr = None, None

            if args.regret_threshold is not None:
                n_samples_regret_mean = np.mean(n_samples_regret)
                n_samples_regret_stderr = np.std(n_samples_regret) / np.sqrt(
                    len(n_samples_regret)
                )
                n_samples_regret_median = np.median(n_samples_regret)
                n_samples_regret_25 = np.percentile(n_samples_regret, 25)
                n_samples_regret_75 = np.percentile(n_samples_regret, 75)

                if args.pickle_out_folder is not None:
                    filename = f"{args.experiment_label}_{af_label}.pkl"
                    print(f"writing to {filename}")
                    with open(
                        os.path.join(args.pickle_out_folder, filename), "wb"
                    ) as f:
                        pickle.dump(n_samples_regret, f)
            else:
                n_samples_regret_mean, n_samples_regret_stderr = None, None
                n_samples_regret_median, n_samples_regret_25, n_samples_regret_75 = (
                    None,
                    None,
                    None,
                )

            if args.sample_threshold is not None:
                return_mean = np.mean(return_list)
                return_stderr = np.std(return_list) / np.sqrt(len(return_list))
                regret_mean = np.mean(regret_list)
                regret_stderr = np.std(regret_list) / np.sqrt(len(regret_list))
                run_time_mean = np.mean(run_time_list)
                run_time_stderr = np.std(run_time_list) / np.sqrt(len(run_time_list))
            else:
                return_mean, return_stderr, regret_mean, regret_stderr = (
                    None,
                    None,
                    None,
                    None,
                )
                run_time_mean, run_time_stderr = None, None

            results[af_label] = (
                n_samples_return_mean,
                n_samples_return_stderr,
                n_samples_regret_mean,
                n_samples_regret_stderr,
                n_samples_regret_median,
                n_samples_regret_25,
                n_samples_regret_75,
                return_mean,
                return_stderr,
                regret_mean,
                regret_stderr,
                run_time_mean,
                run_time_stderr,
            )

    print(
        f"Avg. # samples needed to achieve return threshold of {args.return_threshold}:"
    )
    for label, (mean, stderr, _, _, _, _, _, _, _, _, _, _, _) in results.items():
        print(f"{label}:  mean = {mean}, stderr (SEM) = {stderr}")

    print()
    print(
        f"Avg. # samples needed to achieve regret threshold of {args.regret_threshold}:"
    )
    for label, (
        _,
        _,
        mean,
        stderr,
        median,
        per25,
        per75,
        _,
        _,
        _,
        _,
        _,
        _,
    ) in results.items():
        print(
            f"{label}:  mean = {mean}, stderr (SEM) = {stderr}, median = {median}, "
            f"25th percentile {per25}, 75th percentile {per75}"
        )

    print()
    print(f"Avg. return after {args.sample_threshold} samples:")
    for label, (_, _, _, _, _, _, _, mean, stderr, _, _, _, _) in results.items():
        print(f"{label}:  mean = {mean}, stderr (SEM) = {stderr}")

    print()
    print(f"Avg. regret after {args.sample_threshold} samples:")
    for label, (_, _, _, _, _, _, _, _, _, mean, stderr, _, _) in results.items():
        print(f"{label}:  mean = {mean}, stderr (SEM) = {stderr}")

    print()
    print(f"Avg. runtime for {args.sample_threshold} samples:")
    for label, (_, _, _, _, _, _, _, _, _, _, _, mean, stderr) in results.items():
        print(f"{label}:  mean = {mean}, stderr (SEM) = {stderr}")


if __name__ == "__main__":
    main()
