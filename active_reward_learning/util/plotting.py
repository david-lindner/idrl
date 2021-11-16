# see https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
# import matplotlib
#
# matplotlib.use("Agg")

import os
from types import SimpleNamespace
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from active_reward_learning.common.policy import TabularPolicy
from active_reward_learning.envs import Gridworld, TabularMDP
from active_reward_learning.envs.reward_model_mean_wrapper import RewardModelMeanWrapper
from active_reward_learning.reward_models.query import ComparisonQueryLinear, StateQuery
from active_reward_learning.util.mujoco import evaluate_mujoco_policy_from_s0


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Source: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap


def plot_swimmer_reward_function(
    env,
    model=None,
    plot_error=False,
    policy=None,
    N=30,
    cut_1d=False,
    filename=None,
    feature_function=lambda x: x,
):
    """
    Plots a reward or value function for moving on a circle in the Swimmer environment.

    Provides three cases:
        - If model is given, plot the mean estimate of this gp model, else use the
          true reward function of the environment
        - If policy is given, estimate the value function of this policy by MC
          estimation and plot it otherwise plot the instantaneous reward
    """

    plot_value = policy is not None
    use_model = model is not None
    print("plot_value", plot_value)
    print("use_model", use_model)

    if env.spec.id != "Swimmer-Circle-v2":
        raise NotImplementedError()

    Nx, Ny = N, N
    if cut_1d:
        Nx = 3

    x = np.linspace(-10, 10, Nx)
    y = np.linspace(-10, 10, Ny)
    xx, yy = np.meshgrid(x, y)

    # vx = np.linspace(-10, 10, N)
    # vy = np.linspace(-10, 10, N)
    # x = 1
    # y = 1s
    # vxvx, vyvy = np.meshgrid(vx, vy)

    # gp_repr = [
    #     -2.69230887e-04,
    #     -1.15579502e-04,
    #     -6.19243598e-01,
    #     5.73713134e00,
    #     2.82720274e-01,
    #     -3.80467882e-01,
    # ]
    # features = feature_function(gp_repr)
    # print(model.predict(features))
    # import pdb; pdb.set_trace()

    labels = ("north ↑", "east →", "south ↓", "west ←", "zero ·", "1 west ←")
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0), (0, 0), (-0.1, 0))

    fig, ((ax1, ax2), (ax4, ax3), (ax5, ax6)) = plt.subplots(3, 2)
    axes = (ax1, ax2, ax3, ax4, ax5, ax6)
    ax6.axis("off")

    if plot_value and use_model:
        reward_model = SimpleNamespace()
        setattr(reward_model, "gp_model", model)
        env = RewardModelMeanWrapper(env, reward_model)

    for label, (vx, vy), ax in zip(labels, directions, axes):
        if use_model:
            ndim = feature_function(np.zeros(env.Ndim_repr)).shape[0]
            gp_repr_array = np.zeros((Nx * Ny, ndim))

        if plot_error or not use_model:
            true_reward = np.zeros((Ny, Nx))

        if plot_value:
            value = np.zeros((Ny, Nx))

        for i in range(Ny):
            print(i)
            for j in range(Nx):
                if plot_value:
                    q = np.array([xx[i, j], yy[i, j]])
                    v = np.array([vx, vy])
                    value[i, j] = evaluate_mujoco_policy_from_s0(env, policy, q, v, 3)
                else:
                    gp_repr = np.array(
                        [vx, vy, xx[i, j], yy[i, j]] + [0] * (env.Ndim_repr - 4)
                    )
                    # gp_repr = np.array([vxvx[i,j], vyvy[i,j], x, y] + [0] * (env.Ndim_repr - 4))

                    if use_model:
                        features = feature_function(gp_repr)
                        idx = np.ravel_multi_index((i, j), (Ny, Nx))
                        gp_repr_array[idx] = features

                    if (not use_model) or plot_error:
                        true_reward[i, j] = np.dot(env.reward_w, gp_repr)

        if use_model and not plot_value:
            reward = model.predict_multiple(gp_repr_array)[0]
            reward = np.reshape(reward, (Ny, Nx))
            if plot_error:
                reward = reward - true_reward

        if plot_value:
            vmin = -500
            vmax = None
            cmap = None
            np.save("value_function_{}.npy".format(label.split(" ")[0]), value)
            reward = value
        elif plot_error:
            vmin = -0.1
            vmax = 0.1
            cmap = "RdBu"
            # vmin, vmax = None, None
        else:
            vmin = -5
            vmax = 1
            cmap = None
            if not use_model:
                reward = true_reward

        if cut_1d:
            ax.plot(reward[:, 0])
        else:
            cmap = plt.get_cmap(cmap)
            cmap = shiftedColorMap(cmap, midpoint=1 - vmax / (vmax - vmin))
            im = ax.imshow(
                reward,
                extent=(x.min(), x.max(), y.min(), y.max()),
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            plt.colorbar(im, ax=ax)
        ax.set_title(label)

    # plt.colorbar()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def plot_query_state(env, state, ret, mean, var, plot_folder, filename):
    info = state.info

    if isinstance(env, Gridworld):
        env.reset()
        x, y = env._get_agent_pos_from_state(state.state)
        env.agent_xpos, env.agent_ypos = x, y
        rgb = env.render("rgb_array")
    else:
        # not compatible with all envs
        env.state = info["gp_repr"]
        rgb = env.render("rgb_array")

    plt.imshow(rgb, origin="upper")
    plt.title(
        "Query: mean {:.2f}  var {:.2f} (truth {:.2f})\npolicy return = {:.2f}".format(
            mean, var, state.reward, ret
        )
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, filename))


def plot_reward_value_policy(
    env: TabularMDP,
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    V: np.ndarray,
    policy: TabularPolicy,
    total_return: float = None,
    last_query: Optional[object] = None,
    plot_folder: str = None,
    filename: str = None,
) -> None:
    """
    Produces a figure to visualize the progress of a GP reward model.

    Three plots are produced (from left to right):
        (i) The true reward function compared with the current GP model,
            visualized by mean and variance.
        (ii) The value function of the last policy w.r.t the current GP reward
             model.
        (iii) The new policy (which might be greedy w.r.t. the value function
              in (ii)).

    The plots are generated by the environments `visualize_reward_estimate`,
    `visualize_value` and `visualize_policy` methods.

    Args:
    ---------
    env (TabularMDP): the MDP environment
    mu_pred (np.ndarray): 1D-array of predicted rewards from the GP
    sigma_pred (np.ndarray): 1D-array of reward variances from the GP
    V (np.ndarray): 1D-array of value function from the last policy according
                    to the current reward model
    policy (np.ndarray): 1D-array of actions for each state that encodes the
                         policy to plot
    total_return (float): Return of the latest policy (used in the plot title)
    plot_folder (str): Path to the folder to store the plots in (if not given,
                       the plot will be shown with `plt.show()`)
    filename (str): Filename if the plot should be written to a file (if not
                    given, the plot will be shown with `plt.show()`)

    Returns
    ---------
    None

    Writes
    ---------
    If `plot_folder` and `filename` are given, plots (i)-(iii) are writen to a
    file at `plot_folder/filename`.
    """
    N = len(env.rewards)
    states = range(N)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    env.visualize_reward_estimate(mu_pred, sigma_pred, ax1)
    ax1.set_title("reward")
    if V is not None:
        if last_query is not None:
            if isinstance(last_query, StateQuery):
                state = [last_query.info["state"]]
            elif isinstance(last_query, ComparisonQueryLinear):
                if isinstance(env, Gridworld):
                    state = []
                else:
                    state = [last_query.gp_repr1, last_query.gp_repr2]
            else:
                print(f"Query type {type(last_query)} can not be plotted.")
                state = []
        else:
            state = []
        env.visualize_value(V, ax2, state)
        ax2.set_title("value")
    env.visualize_policy(policy, ax3)
    ax3.set_title("policy")

    if total_return is not None:
        fig.suptitle("return = {:.2f}".format(total_return))
    if filename and plot_folder:
        plt.savefig(os.path.join(plot_folder, filename))
    else:
        plt.show()
    plt.close(fig)


def fix_inconsistent_result_lengths(steps, results):
    lens = [len(res) for res in results]
    lens_st = [len(st) for st in steps]
    assert lens == lens_st
    min_len = min(lens)
    max_len = max(lens)
    if min_len != max_len:
        print("WARNING: Inconsistent result lengths!")
        print(lens)
        median_length = int(np.median(lens))
        print("Restricting to median:", median_length)
        results = [res[:median_length] for res in results if len(res) >= median_length]
        steps = [st[:median_length] for st in steps if len(st) >= median_length]
    else:
        print("Result lengths:", min_len)
    return steps, results


def get_consistent_steps(steps):
    assert np.allclose(steps[0], steps)
    return steps[0]


def set_plot_style(family="serif"):
    import matplotlib
    import seaborn as sns

    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 3})
    sns.set_style("white")
    matplotlib.rc(
        "font",
        **{
            "family": family,
            "serif": ["Computer Modern"],
            "sans-serif": ["Latin Modern"],
        },
    )
    matplotlib.rc("text", usetex=True)


def plot_result_percentiles(
    steps,
    results,
    legend_label,
    color,
    hatch,
    marker,
    legend_handles,
    legend_labels,
    fix_negative=True,
    no_markers=False,
    markers_every=1,
    plot_mean=False,
    plot_stddev=False,
    plot_stderr=False,
    alpha=1,
    zorder=1,
):
    steps, results = fix_inconsistent_result_lengths(steps, results)
    steps = get_consistent_steps(steps)
    results_array = np.stack(results, axis=0)
    # print(results_array[:,3])
    if plot_mean:
        median = np.mean(results_array, axis=0)
    else:
        median = np.median(results_array, axis=0)

    assert not (plot_stddev and plot_stderr)

    if plot_stddev or plot_stderr:
        stddev = np.std(results_array, axis=0)
        if plot_stderr:
            stddev /= np.sqrt(results_array.shape[0])
        lower = median - stddev
        upper = median + stddev
    else:
        lower = np.percentile(results_array, 25, axis=0)
        upper = np.percentile(results_array, 75, axis=0)

    if fix_negative:
        if not np.all(results_array > -0.00001):
            print("WARNING: negative results")
            print("results_array", results_array)
        if not np.all(median > 0):
            print("WARNING: fixing negative medians")
            median = np.maximum(median, np.zeros_like(median))  ## LITTLE HACK

    if no_markers:
        marker = None

    p1 = plt.plot(
        steps,
        median,
        color=color,
        marker=marker,
        markevery=markers_every,
        markersize=11,
        alpha=alpha,
        zorder=zorder,
    )[0]
    p2 = plt.fill_between(
        steps,
        lower,
        upper,
        color=color,
        # hatch=hatch_cycle[hatch],
        alpha=0.2,
        zorder=zorder,
    )
    legend_handles.append((p1, p2))
    legend_labels.append(legend_label)
    return legend_handles, legend_labels


def plot_result_max(
    steps,
    results,
    legend_label,
    color,
    hatch,
    marker,
    legend_handles,
    legend_labels,
    no_markers=False,
    alpha=1,
    zorder=1,
):
    steps, results = fix_inconsistent_result_lengths(steps, results)
    steps = get_consistent_steps(steps)
    results_array = np.stack(results, axis=0)
    # print(results_array[:,3])
    max_vals = np.max(results_array, axis=0)
    p1 = plt.plot(
        steps, max_vals, color=color, marker=marker, alpha=alpha, zorder=zorder
    )[0]
    legend_handles.append(p1)
    legend_labels.append(legend_label)
    return legend_handles, legend_labels


def customized_box_plot(ax, steps, lowers, q1s, medians, q3s, uppers, *args, **kwargs):
    """
    Generates a customized boxplot based on the given percentile values.

    Based on:
    https://stackoverflow.com/questions/27214537/
    is-it-possible-to-draw-a-matplotlib-boxplot-given-the-percentile-values-instead
    """
    boxes = [
        {
            "label": step,
            "whislo": lower,  # Bottom whisker position
            "q1": q1,  # First quartile (25th percentile)
            "med": median,  # Median         (50th percentile)
            "q3": q3,  # Third quartile (75th percentile)
            "whishi": upper,  # Top whisker position
            "fliers": [],  # Outliers
        }
        for i, (step, lower, q1, median, q3, upper) in enumerate(
            zip(steps, lowers, q1s, medians, q3s, uppers)
        )
    ]
    ax.bxp(boxes, showfliers=False, positions=steps, *args, **kwargs)
