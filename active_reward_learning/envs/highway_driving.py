"""
Based on https://github.com/dsadigh/driving-preferences
and https://github.com/Stanford-ILIAD/easy-active-learning/
"""
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from gym import spaces
from gym.utils import seeding
from matplotlib.image import AxesImage, BboxImage
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.ndimage import rotate, zoom

from active_reward_learning.common.constants import (
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_UP,
)
from active_reward_learning.common.policy import FixedPolicy

IMG_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "img", "highway")

GRASS = np.tile(plt.imread(os.path.join(IMG_FOLDER, "grass.png")), (5, 5, 1))

CAR = {
    color: zoom(
        np.array(
            plt.imread(os.path.join(IMG_FOLDER, "car-{}.png".format(color))) * 255.0,
            dtype=np.uint8,  # zoom requires uint8 format
        ),
        [0.3, 0.3, 1.0],
    )
    for color in ["gray", "orange", "purple", "red", "white", "yellow"]
}

COLOR_AGENT = "orange"
COLOR_ROBOT = "white"

CAR_AGENT = CAR[COLOR_AGENT]
CAR_ROBOT = CAR[COLOR_ROBOT]
CAR_SCALE = 0.15 / max(list(CAR.values())[0].shape[:2])

LANE_SCALE = 10.0
LANE_COLOR = (0.4, 0.4, 0.4)  # 'gray'
LANE_BCOLOR = "white"

STEPS = 100


def set_image(
    obj: AxesImage,
    data: np.ndarray,
    scale: float = CAR_SCALE,
    x: List[float] = [0.0, 0.0, 0.0, 0.0],
):
    ox = x[0]
    oy = x[1]
    angle = x[2]
    img = rotate(data, np.rad2deg(angle))
    h, w = img.shape[0], img.shape[1]
    obj.set_data(img)
    obj.set_extent(
        [
            ox - scale * w * 0.5,
            ox + scale * w * 0.5,
            oy - scale * h * 0.5,
            oy + scale * h * 0.5,
        ]
    )


class Car:
    def __init__(
        self,
        initial_state: Union[List[float], np.ndarray],
        actions: List[Union[Tuple[float, float], np.ndarray]],
    ):
        self.initial_state = initial_state
        self.state = self.initial_state
        self.actions = actions
        self.action_i = 0

    def reset(self):
        self.state = self.initial_state
        self.action_i = 0

    def update(
        self, update_fct: Callable[[List[float], float, float], List[float]]
    ) -> None:
        u1, u2 = self.actions[self.action_i % len(self.actions)]
        self.state = update_fct(self.state, u1, u2)
        self.action_i += 1

    def gaussian(
        self, x: List[float], height: float = 0.07, width: float = 0.03
    ) -> float:
        car_pos = np.asarray([self.state[0], self.state[1]])
        car_theta = self.state[2]
        car_heading = (np.cos(car_theta), np.sin(car_theta))
        pos = np.asarray([x[0], x[1]])
        d = car_pos - pos
        dh = np.dot(d, car_heading)
        dw = np.cross(d, car_heading)
        return np.exp(-0.5 * ((dh / height) ** 2 + (dw / width) ** 2))


class Lane:
    def __init__(
        self,
        start_pos: Union[List[float], np.ndarray],
        end_pos: Union[List[float], np.ndarray],
        width: float,
    ):
        self.start_pos = np.asarray(start_pos)
        self.end_pos = np.asarray(end_pos)
        self.width = width
        d = self.end_pos - self.start_pos
        self.dir = d / np.linalg.norm(d)
        self.perp = np.asarray([-self.dir[1], self.dir[0]])

    def gaussian(self, state: List[float], sigma: float = 0.5) -> float:
        pos = np.asarray([state[0], state[1]])
        dist_perp = np.dot(pos - self.start_pos, self.perp)
        return np.exp(-0.5 * (dist_perp / (sigma * self.width / 2.0)) ** 2)

    def direction(self, x: List[float]):
        return np.cos(x[2]) * self.dir[0] + np.sin(x[2]) * self.dir[1]

    def shifted(self, m: int) -> "Lane":
        return Lane(
            self.start_pos + self.perp * self.width * m,
            self.end_pos + self.perp * self.width * m,
            self.width,
        )


class HighwayDriving(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 1}

    def __init__(
        self,
        cars: List[Car],
        reward_weights: Optional[np.ndarray] = None,
    ):
        self.initial_state = [0, -0.3, np.pi / 2, 0.4]
        self.state = self.initial_state

        self.episode_length = 50
        self.dt = 0.1

        self.friction = 1
        self.vmax = 1
        self.xlim = (-1, 1)
        # self.ylim = (-0.2, 0.8)
        self.ylim = (-0.4, 2.5)
        print("state0", self.state)

        lane = Lane([0.0, -1.0], [0.0, 1.0], 0.17)
        road = Lane([0.0, -1.0], [0.0, 1.0], 0.17 * 3)
        self.lanes = [lane.shifted(0), lane.shifted(-1), lane.shifted(1)]
        self.fences = [lane.shifted(2), lane.shifted(-2)]
        self.roads = [road]
        self.cars = cars

        n_features = len(self.get_features())
        if reward_weights is not None:
            assert reward_weights.shape == (n_features,)
            self.reward_w = np.array(reward_weights)
        else:
            self.reward_w = np.random.normal(size=n_features)

        self.reward_w[-1] = 0
        self.reward_w /= np.linalg.norm(self.reward_w)
        self.reward_w /= 2
        self.reward_w[-1] = 0.5

        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = spaces.Box(
            np.array([-np.inf, -np.inf, 0, -np.inf]),
            np.array([np.inf, np.inf, 2 * np.pi, np.inf]),
            dtype=np.float32,
        )

        self.Ndim_repr = n_features

        self.time = 0
        self.history: List[
            Tuple[List[float], List[Tuple[float, float, float, float]]]
        ] = []
        self._update_history()

        self.seed()

    def _update_history(self) -> None:
        self.history.append((np.array(self.state), self._get_car_states()))

    def _get_car_states(self) -> List[np.ndarray]:
        return [np.array(car.state) for car in self.cars]

    def _update_state(self, state: List[float], u1: float, u2: float) -> List[float]:
        x, y, theta, v = state
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = v * u1
        dv = u2 - self.friction * v
        new_v = max(min(v + dv * self.dt, self.vmax), -self.vmax)
        return [x + dx * self.dt, y + dy * self.dt, theta + dtheta * self.dt, new_v]

    def _get_reward_for_state(self, state: Optional[List[float]] = None) -> float:
        if state is None:
            state = self.state
        return np.dot(self.reward_w, self.get_features(state))

    def seed(self, seed: Optional[float] = None) -> List[float]:
        self.np_random, seed = seeding.np_random(seed)
        assert seed is not None
        return [seed]

    def step(
        self, action: Tuple[float, float]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = np.array(action)
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        u1, u2 = action

        self.state = self._update_state(self.state, u1, u2)
        for car in self.cars:
            car.update(self._update_state)

        self.time += 1
        done = bool(self.time >= self.episode_length)
        reward = self._get_reward_for_state()
        # print("state", self.state)
        # print("features", self.get_features())
        # print("reward", reward)
        info = {"gp_repr": self.get_features(), "x": self.state[0], "y": self.state[1]}
        self._update_history()
        return np.array(self.state + [self.time]), reward, done, info

    def reset(self) -> np.ndarray:
        self.state = self.initial_state
        self.time = 0
        for car in self.cars:
            car.reset()
        self.history = []
        self._update_history()
        return np.array(self.state + [self.time])

    def get_features(self, state: Optional[List[float]] = None) -> np.ndarray:
        if state is None:
            state = self.state

        x, y, theta, v = state

        # staying in lane (higher is better)
        staying_in_lane = (
            np.exp(
                -30
                * np.min(
                    [np.square(x - 0.17), np.square(x), np.square(x + 0.17)], axis=0
                )
            )
            / 0.15343634
        )

        # keeping speed (lower is better)
        keeping_speed = np.square(v - 1) / 0.42202643

        # heading (higher is better)
        heading = np.sin(theta) / 0.06112367

        car_x, car_y, car_theta, car_v = self.cars[0].state
        # collision avoidance (lower is better)
        collision_avoidance = (
            np.exp(-(7 * np.square(x - car_x) + 3 * np.square(y - car_y))) / 0.15258019
        )

        return (
            np.array(
                [
                    staying_in_lane,
                    keeping_speed,
                    heading,
                    collision_avoidance,
                    self.episode_length,
                ]
            )
            / self.episode_length
        )

    def _get_trajectory_features_from_flat_policy(self, policy, p=False):
        a_dim = self.action_space.shape[0]
        n_policy_steps = len(policy) // a_dim
        n_repeat = self.episode_length // n_policy_steps

        self.reset()
        features = np.zeros_like(self.get_features())
        for i in range(self.episode_length):
            if i % n_repeat == 0:
                action_i = a_dim * (i // n_repeat)
                action = (policy[action_i], policy[action_i + 1])
            s, _, done, _ = self.step(action)
            assert (i < self.episode_length - 1) or done
            if p:
                print("action", action)
                print("state", s)
            features += self.get_features()
        return features

    def get_optimal_policy(self, w=None, restarts=10, n_action_repeat=10):
        a_dim = self.action_space.shape[0]
        # for numerical stability add eps
        eps = 1e-5
        a_low = list(self.action_space.low + eps)
        a_high = list(self.action_space.high - eps)
        n_policy_steps = self.episode_length // n_action_repeat

        if w is None:
            w = self.reward_w

        def func(policy):
            features = self._get_trajectory_features_from_flat_policy(policy)
            return -np.array(features).dot(w)

        opt_val = np.inf
        bounds = list(zip(a_low, a_high)) * n_policy_steps

        for i in range(restarts):
            print(i, end=" ", flush=True)
            x0 = np.random.uniform(
                low=a_low * n_policy_steps,
                high=a_high * n_policy_steps,
                size=(n_policy_steps * a_dim,),
            )
            temp_res = opt.fmin_l_bfgs_b(func, x0=x0, bounds=bounds, approx_grad=True)
            if temp_res[1] < opt_val:
                optimal_policy = temp_res[0]
                opt_val = temp_res[1]

        # f = self._get_trajectory_features_from_flat_policy(optimal_policy, p=True)
        # print("return 1", np.dot(self.reward_w, f))

        policy_repeat = []
        for i in range(n_policy_steps):
            policy_repeat.extend([optimal_policy[2 * i : 2 * i + 2]] * n_action_repeat)
        return FixedPolicy(np.array(policy_repeat))

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode not in ("human", "rgb_array", "human_static"):
            raise NotImplementedError("render mode {} not supported".format(mode))
        fig = plt.figure(figsize=(7, 7))

        ax = plt.gca()
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.set_aspect("equal")

        grass = BboxImage(ax.bbox, interpolation="bicubic", zorder=-1000)
        grass.set_data(GRASS)
        ax.add_artist(grass)

        for lane in self.lanes:
            path = Path(
                [
                    lane.start_pos
                    - LANE_SCALE * lane.dir
                    - lane.perp * lane.width * 0.5,
                    lane.start_pos
                    - LANE_SCALE * lane.dir
                    + lane.perp * lane.width * 0.5,
                    lane.end_pos + LANE_SCALE * lane.dir + lane.perp * lane.width * 0.5,
                    lane.end_pos + LANE_SCALE * lane.dir - lane.perp * lane.width * 0.5,
                    lane.start_pos
                    - LANE_SCALE * lane.dir
                    - lane.perp * lane.width * 0.5,
                ],
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
            )
            ax.add_artist(
                PathPatch(
                    path,
                    facecolor=LANE_COLOR,
                    lw=0.5,
                    edgecolor=LANE_BCOLOR,
                    zorder=-100,
                )
            )

        for car in self.cars:
            img = AxesImage(ax, interpolation="bicubic", zorder=20)
            set_image(img, CAR_ROBOT, x=car.state)
            ax.add_artist(img)

        human = AxesImage(ax, interpolation=None, zorder=100)
        set_image(human, CAR_AGENT, x=self.state)
        ax.add_artist(human)

        plt.axis("off")
        plt.tight_layout()
        if mode != "human_static":
            fig.canvas.draw()
            rgb = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            rgb = rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            del fig
            if mode == "rgb_array":
                return rgb
            elif mode == "human":
                plt.imshow(rgb, origin="upper")
                plt.axis("off")
                plt.tight_layout()
                plt.pause(0.05)
                plt.clf()
        return None

    def get_keys_to_action(self):
        return {
            (): np.array((0, 0)),
            (KEY_UP,): np.array((0, 0.15)),
            (KEY_DOWN,): np.array((0, -0.15)),
            (KEY_LEFT,): np.array((1, 0)),
            (KEY_RIGHT,): np.array((-1, 0)),
            (KEY_UP, KEY_LEFT): np.array((1, 0.15)),
            (KEY_UP, KEY_RIGHT): np.array((-1, 0.15)),
            (KEY_DOWN, KEY_LEFT): np.array((1, -0.15)),
            (KEY_DOWN, KEY_RIGHT): np.array((-1, -0.15)),
        }  # control with arrow keys

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def plot_history(self):
        x_player = []
        y_player = []
        N_cars = len(self.cars)
        x_cars = [[]] * N_cars
        y_cars = [[]] * N_cars
        for h in self.history:
            x_player.append(h[0][0])
            y_player.append(h[0][1])
            for car_state in h[1]:
                for i in range(N_cars):
                    x_cars[i].append(h[1][i][0])
                    y_cars[i].append(h[1][i][1])
        print("x_player", x_player)
        print("y_player", y_player)
        print("x_cars", x_cars)
        print("y_cars", y_cars)
        self.reset()
        rgb = self.render(mode="human_static")
        plt.axis("off")
        plt.tight_layout()
        plt.plot(
            x_player,
            y_player,
            zorder=10,
            linestyle="-",
            color=COLOR_AGENT,
            linewidth=2.5,
            marker="o",
            markersize=8,
            markevery=5,
        )
        for i in range(N_cars):
            plt.plot(
                x_cars[i],
                y_cars[i],
                zorder=10,
                linestyle="-",
                color=COLOR_ROBOT,
                linewidth=2.5,
                marker="o",
                markersize=8,
                markevery=5,
            )

    def sample_features_rewards(self, n_samples):
        min_val = -1
        max_val = 1
        samples = min_val + (max_val - min_val) * np.random.sample(
            (n_samples, self.Ndim_repr)
        )
        samples[:, -1] = 0
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        samples[:, -1] = 1
        return samples, np.matmul(samples, self.reward_w.T)


def get_highway(random_reward=False):
    # driving straight on left lane
    # car = Car([-0.13, 0.0, np.pi / 2.0, 0.1], [(0, 0.05)])
    # driving straight on right lane
    # car = Car([0.13, 0.0, np.pi / 2.0, 0.1], [(0, 0.05)])
    # car driving from left to middle lane
    # car = Car(
    #     [-0.13, 0.0, np.pi / 2.0, 0.4],
    #     [(-1, 0.4), (-1, 0), (-1, 0), (-1, 0), (-1, 0)],
    # )
    # car driving from right to middle lane
    car = Car(
        [0.17, 0.0, np.pi / 2.0, 0.41],
        [(0, 0.41)] * 10
        + [(1, 0.41)] * 10
        + [(-1, 0.41)] * 10
        + [(0, 0.52)] * 10
        + [(0, 0.52)] * 10
        # [(0, 0.41), (1, 0.41), (-1, 0.41), (0, 0.52), (0, 0.52)],
    )
    cars = [car]

    if random_reward:
        reward_weights = np.random.multivariate_normal(np.zeros(4), np.identity(4))
    else:
        reward_weights = np.array([0.2, -0.7, 0.4, -0.4])

    reward_weights /= np.linalg.norm(reward_weights)
    reward_weights = np.array(list(reward_weights) + [1])
    return HighwayDriving(cars, reward_weights)


if __name__ == "__main__":
    import time

    from gym.utils.play import play

    from active_reward_learning.solvers import get_standard_solver

    env = gym.make("HighwayDriving-FixedReward-v0")
    # print(env.sample_features_rewards(10))

    # play(env, fps=30)
    # policy = env.get_optimal_policy(restarts=1)
    solver = get_standard_solver(env, "lbfgsb_solver")
    policy = solver.solve()

    s = env.reset()
    done = False
    r = 0
    while not done:
        a = policy.get_action(s)
        s, reward, done, info = env.step(a)
        print("action", a)
        print("state", s)
        print("features", env.get_features())
        r += reward
        # env.render("human")
        # time.sleep(0.5)

    print("return 2", r)
    env.plot_history()
    plt.show()
