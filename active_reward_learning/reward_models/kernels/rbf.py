from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class RBFCustomDist:
    """
    Radial basis function (RBF) kernel that allows to specify a custom distance.

    Attributes:
    -------------
    variance: RBF variance (sigma)
    lengthscale: RBF lengthscale (l)
    distance: Distance function that takes two points and returns a float
    k_cache: used to cache the covariances that have already been calculated
    """

    def __init__(
        self,
        input_dim: int,
        variance: float = 1,
        lengthscale: float = 1,
        distance: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.sum(
            np.sqrt(np.square(x - y).sum())
        ),
    ):
        self.input_dim = input_dim
        self.variance = variance
        self.lengthscale = lengthscale
        self.distance = distance
        self.k_cache: Dict[Tuple[str, str], float] = {}

    def k_func(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Return the kernel function evaluated between two points `x` and `y`.
        This is determined using the RBF function and the custom `distance`
        function.

        Importantly, this function implement caching to avoid recomputing the
        same value multiple times.
        """
        key = (x.tostring(), y.tostring())
        if key in self.k_cache:
            return self.k_cache[key]
        else:
            r = self.distance(x, y) / self.lengthscale
            k = self.variance ** 2 * np.exp(-0.5 * r ** 2)
            k = np.asscalar(k)
            self.k_cache[key] = k
            return k
