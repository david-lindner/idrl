import numpy as np


class LinearKernel:
    def __init__(self, input_dim, variances=None):
        self.input_dim = input_dim
        if variances is None:
            variances = np.ones(input_dim)
        self.variances = variances

    def k_func(self, x: np.ndarray, y: np.ndarray) -> float:
        assert x.shape == (self.input_dim,)
        assert y.shape == (self.input_dim,)
        k = np.matmul(x.T, np.matmul(np.diag(self.variances), y))
        return np.float(k)
