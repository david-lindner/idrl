import os

import numpy as np
from scipy.stats import norm

from active_reward_learning.util.helpers import get_array_from_scalar
from active_reward_learning.util.video import save_video


def get_info_dict(info):
    return tuple(info)


class QueryBase:
    def __init__(self):
        self.info = None

    def _asdict(self):
        raise NotImplementedError()


class LinearQuery(QueryBase):
    def __init__(self, gp_repr_list, linear_combination, reward, info):
        self.gp_repr_list = get_array_from_scalar(gp_repr_list)
        self.linear_combination = get_array_from_scalar(linear_combination)
        self.reward_ = reward
        self.info = info

    @property
    def reward(self):
        return self.reward_

    def __hash__(self):
        l = (
            tuple(
                [
                    gp_repr.tostring()
                    if isinstance(gp_repr, np.ndarray)
                    else str(gp_repr)
                    for gp_repr in self.gp_repr_list
                ]
            ),
            tuple(
                [
                    a.tostring() if isinstance(a, np.ndarray) else str(a)
                    for a in self.linear_combination
                ]
            ),
            self.reward,
            get_info_dict(self.info),
        )
        return hash(l)

    def _asdict(self):
        return {
            "gp_repr_list": [list(gp_repr) for gp_repr in self.gp_repr_list],
            "reward": float(self.reward),
            "info": get_info_dict(self.info),
        }

    def __eq__(self, other):
        return self.__hash__() == hash(other)

    def to_file(self, path, filename="query.txt"):
        file_path = os.path.join(path, filename)
        s = (
            str(self.gp_repr_list)
            + "\n\n"
            + str(self.linear_combination)
            + "n\n"
            + str(self.reward_)
        )
        with open(file_path, "w"):
            print(s)
        return file_path


class PointQuery(LinearQuery):
    def __init__(self, gp_repr, reward, info):
        self.gp_repr = get_array_from_scalar(gp_repr)
        super().__init__([gp_repr], [1], reward, info)

    def __hash__(self):
        l = (
            self.gp_repr.tostring(),
            self.reward,
            get_info_dict(self.info),
        )
        return hash(l)

    def _asdict(self):
        return {
            "gp_repr": self.gp_repr.tolist(),
            "reward": float(self.reward),
            "info": get_info_dict(self.info),
        }

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class TrajectoryQuery(LinearQuery):
    def __init__(self, gp_repr_list, reward_list, info, rgb_arrays=None):
        reward = sum(reward_list)
        self.reward_list = reward_list
        self.rgb_arrays = rgb_arrays
        super().__init__(gp_repr_list, [1] * len(gp_repr_list), reward, info)

    def to_file(self, path, filename="query.gif"):
        if self.rgb_arrays is None:
            filename += ".txt"
            return super().to_file(path, filename=filename)
        else:
            file_path = os.path.join(path, filename)
            save_video(self.rgb_arrays, file_path, fps=5)
            return file_path


class ComparisonQueryLinear(LinearQuery):
    def __init__(
        self, gp_repr1, gp_repr2, reward1, reward2, info, response="bernoulli", **kwargs
    ):
        gp_repr_list = [gp_repr1, gp_repr2]
        linear_combination = [1, -1]
        super().__init__(gp_repr_list, linear_combination, 0, info)
        self.gp_repr1 = gp_repr_list[0]
        self.gp_repr2 = gp_repr_list[1]
        self.reward1 = reward1
        self.reward2 = reward2

        self.response = response
        if response == "difference":
            self.p = None
        elif response == "bernoulli":
            assert 0 <= reward1 <= 1, reward1
            assert 0 <= reward2 <= 1, reward2
            self.p = (1 + reward1 - reward2) / 2
        elif response == "deterministic":
            self.p = 1 if reward1 > reward2 else 0
        elif response == "probit":
            if "sigma" in kwargs:
                self.sigma = kwargs["sigma"]
            else:
                self.sigma = 1
            self.p = norm.cdf((reward1 - reward2) / (np.sqrt(2) * self.sigma))

    @property
    def reward(self):
        if self.p is None:
            return self.reward1 - self.reward2
        else:
            return np.random.choice([-1, 1], p=[1 - self.p, self.p])

    def __hash__(self):
        l = (
            self.gp_repr1.tostring(),
            self.gp_repr2.tostring(),
            self.reward1,
            self.reward2,
            get_info_dict(self.info),
            self.response,
        )
        return hash(l)


class StateQuery(PointQuery):
    def __init__(self, state, gp_repr, reward, info, obs=None):
        info["state"] = state
        info["obs"] = obs
        super().__init__(gp_repr, reward, info)


class StateComparisonQueryLinear(ComparisonQueryLinear):
    def __init__(
        self, gp_repr1, gp_repr2, reward1, reward2, info, response="bernoulli"
    ):
        super().__init__(gp_repr1, gp_repr2, reward1, reward2, info, response=response)
