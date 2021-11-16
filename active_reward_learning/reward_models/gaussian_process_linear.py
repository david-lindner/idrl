import logging
import pickle

import numpy as np

from active_reward_learning.reward_models.kernels import LinearKernel
from active_reward_learning.util.helpers import multivariate_normal_sample

logger = logging.getLogger("GP")
logger.setLevel(logging.WARNING)

EPS = np.finfo(np.float32).eps


def draw_random_rewards_from_GP(N_states, kernel, reward_seed=None):
    if reward_seed is not None:
        reset_seed = np.random.randint(0, 1000000)
        np.random.seed(reward_seed)
    gp = LinearObservationGP(kernel, None, 0)
    observations = np.array([[x] for x in range(N_states)]), np.array([[1]] * N_states)
    reward = gp.sample_y_from_prior(observations)
    if reward_seed is not None:
        np.random.seed(reset_seed)
    return reward


def get_rbf_kernel():
    lengthscale = 1
    variance = 1

    def kernel(x, y):
        r = (x - y) ** 2 / lengthscale
        k = variance ** 2 * np.exp(-0.5 * r ** 2)
        return k


class LinearObservationGP:
    """
    Implements a Gaussian Process model to be used for reward modelling.

    Parameter
    -------------
    kernel: kernel function to use for the GP
    X (list): list of x values observed so far
    Y (list): list of y values observed so far
    N_obs (int): number of observations so far
    ndim (int): number of dimensions of the input space
    """

    def __init__(
        self, kernel, observations=None, obs_var=0, dtype=np.float32, prior_mean=0
    ):
        """
        Create a GP object.

        Args
        ----------
        kernel: kernel function to use for the GP
        observations (iterable): set of obsevations to initialize the GP with
        obs_var (float): variance of the observations for the GP
        """
        if obs_var < 1e-10:
            print("Warning: obs_var too small, setting it to 1e-10")
            obs_var = 1e-10

        self.ndim = kernel.input_dim
        self.kernel = kernel
        self.obs_var = obs_var
        self.dtype = dtype
        self.prior_mean = prior_mean

        self.X_list = []
        self.A_list = []
        self.Y_list = []
        self.N_obs = 0
        self.L = None

        self.X_matrix_list = []
        self.X_matrix = np.array(self.X_matrix_list)
        self._A_inv = None

        if observations is not None:
            for ((x, a), y) in observations:
                self.observe((x, a), y)

        x_test = np.ones(self.ndim)
        assert np.isscalar(self._linear_kernel([x_test], [x_test], [1], [1]))

    def _get_kernel_cholesky(self, obs_noise=None):
        if obs_noise is None:
            obs_noise = self.obs_var
        K_train = self._get_training_kernel_matrix()
        K_train += obs_noise * np.identity(K_train.shape[0])
        cholesky = np.linalg.cholesky(K_train)
        return cholesky

    def _update_cholesky(self, x, a, obs_noise=None):
        if obs_noise is None:
            obs_noise = self.obs_var
        L = self.L
        k_new = self._get_k_train_new([x], [a])
        k_new_new = self._linear_kernel(x, x, a, a) + obs_noise

        L12 = np.linalg.solve(L, k_new)
        L22 = np.sqrt(k_new_new - np.matmul(L12.T, L12))
        L_new = np.block([[L, np.zeros_like(L12)], [L12.T, L22]])

        if np.any(np.isnan(L_new)):
            print(
                L22, k_new_new, np.matmul(L12.T, L12), k_new_new - np.matmul(L12.T, L12)
            )
            raise Exception(
                "Invalid cholesky decomposition. Probably, the kernel "
                "matrix is not positive definite."
            )
        self.L = L_new

    def _linear_kernel(self, x1, x2, a1, a2):
        val = 0
        for x1i, a1i in zip(x1, a1):
            for x2j, a2j in zip(x2, a2):
                val += a1i * a2j * self.kernel.k_func(x1i, x2j)
        return val

    def _get_k_train_new(self, x_new, a_new):
        return self._get_kernel_matrix(self.X_list, x_new, self.A_list, a_new)

    def _get_training_kernel_matrix(self):
        return self._get_kernel_matrix(
            self.X_list, self.X_list, self.A_list, self.A_list
        )

    def _get_kernel_matrix(self, Xa, Xb, Aa, Ab):
        return np.array(
            [
                [
                    self._linear_kernel(
                        x1,
                        x2,
                        a1,
                        a2,
                    )
                    for x2, a2 in zip(Xb, Ab)
                ]
                for x1, a1 in zip(Xa, Aa)
            ]
        )

    def observe(self, obs, y, obs_noise=None):
        """
        Add a single observation to the GP model.

        Args:
        -------------
        x (np.array): x observations
        a (np.array): linear combination
        y (float): y observation
        """
        x, a = obs
        x = np.array(x)
        a = np.array(a)
        y = self.dtype(y - self.prior_mean)

        # fix shapes
        assert np.ndim(a) == 1
        n = a.shape[0]
        if np.ndim(x) == 1:
            if self.ndim > 1:
                # single point
                assert x.shape == (self.ndim,)
                assert np.ndim(a) == 1
                x = np.expand_dims(x, 0)
            else:
                # multiple 1d points
                assert self.ndim == 1
                assert len(x) == len(a)
                x = np.expand_dims(x, 1)
        assert x.shape == (n, self.ndim)

        if self.L is not None:
            # update cholesky incrementally if already exists
            self._update_cholesky(x, a, obs_noise=obs_noise)

        self.X_list.append(x)
        self.A_list.append(a)
        self.Y_list.append(np.asscalar(y))
        x_comb = np.sum(x * np.repeat(np.expand_dims(a, 1), self.ndim, axis=1), axis=0)
        self.X_matrix_list.append(x_comb)
        self.X_matrix = np.array(self.X_matrix_list)
        self._A_inv = None
        self.N_obs += 1

        if self.L is None:
            # compute cholesky from scratch if it does not exist
            self.L = self._get_kernel_cholesky(obs_noise=obs_noise)

    def predict_multiple(
        self, x_new, linear_combination=None, handle_duplicates=True, **kwargs
    ):
        """
        Get multiple GP predictions including covariance matrix.

        Args:
        ---------
        x_new (list): list (or array) of observations to predict y for
        linear_combination (list)

        Returns
        ---------
        mean (np.ndarray): mean vector of the predictions
        cov (np.ndarray): covarinace matrix of the GP predictions
        """
        x_new = np.array(x_new)

        if linear_combination is None:
            # predicting single points instead of linear combination
            x_new = np.reshape(x_new, (-1, 1, self.ndim))
            a_new = np.ones((x_new.shape[0], 1))
        else:
            a_new = np.array(linear_combination)

        assert np.ndim(x_new) == 3
        assert x_new.shape[2] == self.ndim

        K_new = self._get_kernel_matrix(x_new, x_new, a_new, a_new)
        if self.N_obs == 0:
            # use prior when no points have been observed
            mean = np.zeros(len(x_new))
            cov = K_new
        elif self.N_obs == 1:
            # handle case with a single observation without cholesky
            y = np.array(self.Y_list)
            k_new = self._get_k_train_new(x_new, a_new)
            K_train = np.matmul(self.L.T, self.L)
            K_train_inv = np.linalg.inv(K_train)
            k_new_train_inv = np.matmul(k_new.T, K_train_inv)
            mean = np.matmul(k_new_train_inv, y)
            cov = K_new - np.matmul(k_new_train_inv, k_new)
            cov += self.obs_var * np.identity(cov.shape[0])
        else:
            y = np.array(self.Y_list)
            L = self.L
            A = np.linalg.solve(L, y)
            A = np.linalg.solve(L.T, A)
            k_new = self._get_k_train_new(x_new, a_new)
            mean = np.matmul(k_new.T, A)
            v = np.linalg.solve(L, k_new)
            cov = K_new - np.matmul(v.T, v)
            cov += self.obs_var * np.identity(cov.shape[0])
        return mean + self.prior_mean, cov

    def predict(
        self,
        x,
    ):
        """
        Get a GP prediction for a single x value.

        Args:
        -----------
        x (np.ndarray): x observation to predict y for
        random_fourier_features (bool): use random fourier features for prediction

        Returns
        ----------
        mean (float): mean prediction of the GP
        var (float): variance from the GP prediction
        """
        mean, cov = self.predict_multiple(x)
        return mean[0], cov[0, 0]

    def sample_y_from_prior(self, observations):
        """
        Get a sample from the GP prior N(0, K).

        Args:
        ---------
        X (np.ndarray): X values to sample for

        Returns:
        ---------
        y (np.ndarray): y values sampled from GP
        """
        X, A = observations
        K = self._get_kernel_matrix(X, X, A, A)
        y = np.random.multivariate_normal(np.zeros(len(X)), K)
        return y + self.prior_mean

    def prior_likelihood(self, observations, Y):
        """
        Return the likelihood of data under the GP prior N(0, K).

        Args:
        ---------
        observations (np.ndarray): features of input
        Y (np.ndarray): values of input

        Returns:
        ---------
        likelihood (float): prior likelihood of X
        """
        d = self.ndim
        X, A = observations
        Y -= self.prior_mean
        K = self._get_kernel_matrix(X, X, A, A)
        K += self.obs_var * np.identity(K.shape[0])
        print("np.linalg.det(K)", np.linalg.det(K))
        likelihood = (
            (2 * np.pi) ** (-d / 2)
            * np.linalg.det(K) ** (-0.5)
            * np.exp(-0.5 * np.matmul(Y.T, np.matmul(np.linalg.inv(K), Y)))
        )
        return likelihood

    def sample_y_from_posterior(self, x, n_samples):
        """
        Get a sample from the GP posterior.

        Args:
        ---------
        x (np.ndarray): x values to sample for
        n_samples (int): number of samples to draw

        Returns:
        ---------
        y (np.ndarray): y values sampled from GP, will have shape (len(x), n_samples)
        """
        # sample without duplicates, then set all duplicates to same value to avoid
        # singularities in the cholesky decomposition in `multivariate_normal_sample`
        x_unique = np.unique(x, axis=0)

        mean_uniq, cov_uniq = self.predict_multiple(x_unique)

        samples_uniq = multivariate_normal_sample(
            mean_uniq, cov_uniq, n_samples=n_samples
        )

        samples = []
        for i in range(x.shape[0]):
            for j in range(x_unique.shape[0]):
                if np.all(x[i] == x_unique[j]):
                    samples.append(samples_uniq[j])
                    break

        samples = np.array(samples)
        assert samples.shape == (x.shape[0], n_samples)
        return samples + self.prior_mean

    def make_temporary_observation_and_predict(
        self, obs, y, obs_pred, predictive_mean=False
    ):
        old_L = np.copy(self.L) if self.L is not None else None
        # temp = deepcopy(self)
        self.observe(obs, y)
        if predictive_mean:
            mean, cov = self.linear_predictive_mean, self.linear_predictive_cov
        else:
            mean, cov = self.predict_multiple(obs_pred)
        self.L = old_L
        del self.X_list[-1]
        del self.A_list[-1]
        del self.Y_list[-1]
        del self.X_matrix_list[-1]
        self.X_matrix = np.array(self.X_matrix_list)
        self._A_inv = None
        self.N_obs -= 1
        return mean, cov

    @property
    def A_inv(self):
        if self._A_inv is None:
            A = (1 / self.kernel.variances) * np.identity(self.ndim)
            A += (1 / self.obs_var) * np.matmul(self.X_matrix.T, self.X_matrix)
            conditioning_number = np.linalg.cond(A)
            if conditioning_number > 1e7:
                print("Warning: high conditioning_number:", conditioning_number)
            self._A_inv = np.linalg.solve(A.T.dot(A), A.T)
        return self._A_inv

    @property
    def linear_predictive_mean(self):
        assert isinstance(self.kernel, LinearKernel)
        if self.N_obs > 0:
            Y_data = np.array(self.Y_list)
            mu = (1 / self.obs_var) * np.matmul(
                self.A_inv, np.matmul(self.X_matrix.T, Y_data)
            )
        else:
            mu = np.zeros(self.ndim)
        return mu

    @property
    def linear_predictive_cov(self):
        assert isinstance(self.kernel, LinearKernel)
        if self.N_obs > 0:
            cov = self.A_inv
        else:
            cov = np.diag(self.kernel.variances)
        return cov

    def predictive_log_likelihood(self, X, Y):
        print("NotImplemented: predictive_log_likelihood")
        return 0

    def optimize_parameters(self):
        print("NotImplemented: optimize_parameters")

    def save(self, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "rb") as f:
            gp = pickle.load(f)
        return gp
