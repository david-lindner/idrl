import os

import gym
import numpy as np
import pytest

from active_reward_learning.envs import RewardModelMeanWrapper
from active_reward_learning.reward_models.gaussian_process_linear import (
    LinearObservationGP,
)
from active_reward_learning.reward_models.kernels import LinearKernel
from active_reward_learning.reward_models.kernels.rbf import RBFCustomDist
from active_reward_learning.util.helpers import uniformly_sample_evaluation_states

TMP_FOLDER = "/tmp"


def test_GP_1_linear():
    """
    Observing two points with y=0 should create predictions with 0 mean and variance
    assuming no noise.
    """
    kernel = RBFCustomDist(1)

    gp_linear = LinearObservationGP(kernel, [(([-1], [1]), 0), (([1], [1]), 0)], 0)
    gp_linear_no_init = LinearObservationGP(kernel, obs_var=0)
    gp_linear_no_init.observe(([-1], [1]), 0)
    gp_linear_no_init.observe(([1], [1]), 0)

    GP_labels = ("gp_linear", "gp_linear_no_init")
    GPs = (gp_linear, gp_linear_no_init)

    for label, gp in zip(GP_labels, GPs):
        mean, var = gp.predict(0)
        assert mean == 0, label

        mean, cov = gp.predict_multiple([-1, 1])
        assert np.allclose(mean, np.zeros(2), atol=1e-2), label
        assert np.allclose(cov, np.zeros((2, 2)), atol=1e-2), label


def test_GP_1_linear_fixed_prior():
    """
    Same as test before but with a fixed prior mean
    """
    kernel = RBFCustomDist(1)

    gp_linear = LinearObservationGP(
        kernel, [(([-1], [1]), 1), (([1], [1]), 1)], 0, prior_mean=1
    )
    gp_linear_no_init = LinearObservationGP(kernel, obs_var=0, prior_mean=1)
    gp_linear_no_init.observe(([-1], [1]), 1)
    gp_linear_no_init.observe(([1], [1]), 1)

    GP_labels = ("gp_linear", "gp_linear_no_init")
    GPs = (gp_linear, gp_linear_no_init)

    for label, gp in zip(GP_labels, GPs):
        mean, var = gp.predict(0)
        assert mean == 1, label

        mean, cov = gp.predict_multiple([-1, 1])
        assert np.allclose(mean, np.ones(2), atol=1e-2), label
        assert np.allclose(cov, np.zeros((2, 2)), atol=1e-2), label


def test_GP_2_linear():
    kernel = RBFCustomDist(1)

    gp_linear = LinearObservationGP(kernel, [(([1], [1]), 0)], 0)
    gp_linear_no_init = LinearObservationGP(kernel, obs_var=0)
    gp_linear_no_init.observe(([1], [1]), 0)

    GP_labels = ("gp_linear", "gp_linear_no_init")
    GPs = (gp_linear, gp_linear_no_init)

    for label, gp in zip(GP_labels, GPs):
        gp.observe(([-1], [1]), 0)
        gp.observe(([-2], [1]), 0)
        gp.observe(([-3], [1]), 0)
        gp.observe(([2], [1]), 0)
        gp.observe(([3], [1]), 0)

        mean, var = gp.predict(0.5)
        assert mean == 0, label

        mean, cov = gp.predict_multiple([-3, 2])
        assert np.allclose(mean, np.zeros(2), atol=1e-2), label
        assert np.allclose(cov, np.zeros((2, 2)), atol=1e-2), label

        mean_1, cov_1 = gp.predict_multiple(
            [-3, 2, 3, 2, 2, 3], handle_duplicates=False
        )
        mean_2, cov_2 = gp.predict_multiple([-3, 2, 3, 2, 2, 3], handle_duplicates=True)
        assert np.allclose(mean_1, mean_2, atol=1e-2), label
        assert np.allclose(cov_1, cov_2, atol=1e-2), label


def test_GP_3_linear():
    kernel = RBFCustomDist(1)

    gp_linear = LinearObservationGP(kernel, [(([1], [1]), 0)], 0)
    gp_linear_no_init = LinearObservationGP(kernel, obs_var=0)
    gp_linear_no_init.observe(([1], [1]), 0)

    GP_labels = ("gp_linear", "gp_linear_no_init")
    GPs = (gp_linear, gp_linear_no_init)

    for label, gp in zip(GP_labels, GPs):
        gp.observe(([-1, -2, -3, 2, 3], [1, 1, 1, 1, 1]), 0)
        gp.observe(([-2], [1]), 0)
        gp.observe(([-3], [1]), 0)
        gp.observe(([2], [1]), 0)
        gp.observe(([3], [1]), 0)

        mean, var = gp.predict(0.5)
        assert mean == 0, label

        mean, cov = gp.predict_multiple([-3, 2])
        assert np.allclose(mean, np.zeros(2), atol=1e-2), label
        assert np.allclose(cov, np.zeros((2, 2)), atol=1e-2), label

        mean_1, cov_1 = gp.predict_multiple([3, 2, 3, 2, 2, 3], handle_duplicates=False)
        mean_2, cov_2 = gp.predict_multiple([3, 2, 3, 2, 2, 3], handle_duplicates=True)
        assert np.allclose(mean_1, mean_2, atol=1e-2), label
        assert np.allclose(cov_1, cov_2, atol=1e-2), label


def test_GP_4_linear():
    kernel = RBFCustomDist(1)
    gp_linear = LinearObservationGP(kernel, obs_var=0)

    GP_labels = ("gp_linear",)
    GPs = (gp_linear,)

    for label, gp in zip(GP_labels, GPs):
        gp.observe(([37], [1]), 0)
        gp.observe(([0], [1]), 0)
        gp.observe(([1], [1]), 0)
        gp.observe(([2], [1]), 0)
        gp.observe(([3], [1]), 0)
        gp.observe(([4], [1]), 0)
        gp.observe(([5], [1]), 0)

        mean, var = gp.predict(0)
        assert mean == 0, label


def test_GP_5_linear():
    """
    Observe 1 at 0, 1 at 1 and 4 for the sum of 0, 1, 2. Then 2 has to have value 2!
    """
    kernel = RBFCustomDist(1)
    gp_linear = LinearObservationGP(kernel, obs_var=0)

    GP_labels = ("gp_linear",)
    GPs = (gp_linear,)

    for label, gp in zip(GP_labels, GPs):
        gp.observe(([0], [1]), 1)
        gp.observe(([1], [1]), 1)
        gp.observe(([0, 1, 2], [1, 1, 1]), 4)

        mean, var = gp.predict(2)
        assert np.isclose(mean, 2), label


def test_GP_linear_cholesky_update():
    kernel = RBFCustomDist(1)
    np.random.seed(1)

    gp_linear = LinearObservationGP(kernel, obs_var=0.1)
    GP_labels = ("gp_linear",)
    GPs = (gp_linear,)

    for label, gp in zip(GP_labels, GPs):
        for _ in range(20):
            x = 5 * np.random.random()
            gp.observe(([x], [1]), np.random.random())
            cholesky = gp._get_kernel_cholesky()
            np.testing.assert_allclose(gp.L, cholesky, atol=1e-4, rtol=1e-3)
    for label, gp in zip(GP_labels, GPs):
        for _ in range(20):
            x, y = 5 * np.random.random(), 5 * np.random.random()
            gp.observe(([x, y], [1, -1]), x - y + np.random.random())
            cholesky = gp._get_kernel_cholesky()
            np.testing.assert_allclose(gp.L, cholesky, atol=1e-4, rtol=1e-3)


def test_observed_value_influences_mean_but_not_variance_linear():
    kernel = RBFCustomDist(1)
    gp_linear = LinearObservationGP(kernel, [(([1], [1]), 0)], 0)
    GP_labels = "gp_linear"
    GPs = (gp_linear,)

    for label, gp in zip(GP_labels, GPs):
        mean1, cov1 = gp.make_temporary_observation_and_predict(
            ([0], [1]), 1, [(-1, 1), (1, 1)]
        )
        mean2, cov2 = gp.make_temporary_observation_and_predict(
            ([0], [1]), 0, [(-1, 1), (1, 1)]
        )
        mean3, cov3 = gp.make_temporary_observation_and_predict(
            ([0], [1]), -1, [(-1, 1), (1, 1)]
        )
        assert np.all(cov1 == cov2), label
        assert np.all(cov2 == cov3), label
        assert np.any(mean1 != mean2), label
        assert np.any(mean2 != mean3), label
        assert np.any(mean1 != mean3), label


def test_noiseless_linear_kernel():
    np.random.seed(138219)
    min_val, max_val = -1, 1

    try:
        import mujoco_py

        mujoco_available = True
    except ModuleNotFoundError:
        mujoco_available = False

    if mujoco_available:
        for obs_noise in (10 ** (-10),):
            for env_gym_label in (
                "PointMaze1D-v2",
                "PointMaze1D-v2",
                "SwimmerMaze1D-v2",
                "SwimmerMaze1D-v2",
                "HighwayDriving-FixedReward-v0",
            ):
                print(env_gym_label)
                env = gym.make(env_gym_label)
                env.reset()
                _, _, _, info = env.step(env.action_space.sample())
                d = env.Ndim_repr
                n = d + 1  # 1 sample more than necessary for numerical reasons

                w_true = env.reward_w

                x1 = min_val + (max_val - min_val) * np.random.sample((n, d))
                x2 = np.array(info["gp_repr"]) + np.random.normal(0, 0.1, (n, d))
                print(w_true)

                for x in (x1, x2):
                    assert np.linalg.matrix_rank(x) == d

                    y = [np.dot(env.reward_w, x_) for x_ in x]
                    x, y = np.array(x), np.array(y)

                    if "Maze1D" in env_gym_label:
                        variances = 3
                    else:
                        variances = 1

                    kernel2 = LinearKernel(d, variances=variances * np.ones(d))
                    gp_model2 = LinearObservationGP(
                        kernel2, [(([x], [1]), y) for x, y in zip(x, y)], obs_noise
                    )
                    gp_model2.obs_noise = obs_noise

                    w_gp2 = gp_model2.linear_predictive_mean
                    np.testing.assert_allclose(w_true, w_gp2, atol=1e-1)


def test_linear_approx_indicators_do_not_affect_each_other():
    np.random.seed(3)

    ndim = 50
    nsamples = 20

    x_train = [np.eye(ndim)[np.random.choice(ndim)] for _ in range(nsamples)]
    w = np.random.random(ndim)
    y_train = [np.dot(w, xx) for xx in x_train]

    x_train_add = []
    for i in range(2, ndim):
        x_add = np.zeros(ndim)
        x_add[i] = 1
        x_train_add.append(x_add)

    x1 = np.zeros(ndim)
    x1[0] = 1
    x2 = np.zeros(ndim)
    x2[1] = 1
    x_test = [x1, x2]

    obs_noise = 1e-1

    kernel_lin = LinearKernel(ndim)
    gp_model_lin_1 = LinearObservationGP(
        kernel_lin,
        [(([x], [1]), y) for x, y in zip(x_train, y_train)],
        obs_noise,
    )

    gp_model_lin_2 = LinearObservationGP(
        kernel_lin,
        [(([x], [1]), y) for x, y in zip(x_train + x_train_add, y_train)],
        obs_noise,
    )

    mean_lin_1, cov_lin_1 = gp_model_lin_1.predict_multiple(x_test)
    mean_lin_2, cov_lin_2 = gp_model_lin_2.predict_multiple(x_test)
    np.testing.assert_allclose(mean_lin_1, mean_lin_2)
    np.testing.assert_allclose(cov_lin_1, cov_lin_2)
