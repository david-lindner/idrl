import time
from typing import Callable, Dict, Optional

from stable_baselines.common.vec_env import DummyVecEnv

from active_reward_learning.common import StableBaselinesPolicy
from active_reward_learning.envs import TabularMDP

from .base import BaseSolver


class StableBaselinesSolver(BaseSolver):
    def __init__(
        self, env, stable_baselines_model, stable_baselines_policy, tensorboard_log=None
    ):
        assert not isinstance(env, TabularMDP) or env.observation_type == "raw"
        self.stable_baselines_model = stable_baselines_model
        self.stable_baselines_policy = stable_baselines_policy
        self.tensorboard_log = tensorboard_log
        super().__init__(env)
        self.set_env(env)
        self.policy: StableBaselinesPolicy

    def solve(
        self,
        n_episodes: int = 100,
        logging_callback: Optional[Callable[[Dict, Dict], None]] = None,
    ) -> StableBaselinesPolicy:
        t = time.time()
        self.model.learn(n_episodes, callback=logging_callback)
        self.update_policy()
        self.solve_time += time.time() - t
        return self.policy

    def load(self, filename):
        self.model = self.stable_baselines_model.load(filename)
        self.policy = StableBaselinesPolicy(self.model)

    def save(self, filename):
        self.model.save(filename)

    def update_policy(self):
        self.policy = StableBaselinesPolicy(self.model)

    def set_env(self, env):
        self.model = self.stable_baselines_model(
            self.stable_baselines_policy,
            DummyVecEnv([lambda: env]),
            verbose=1,
            tensorboard_log=self.tensorboard_log,
        )
        self.update_policy()
        super().set_env(env)
