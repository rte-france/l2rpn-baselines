from typing import Any
from torch_geometric.data import HeteroData
import grid2op
from grid2op.Reward import LinesCapacityReward
from grid2op.Chronics import MultifolderWithCache
from lightsim2grid import LightSimBackend
from grid2op.gym_compat import GymActionSpace
from ray.tune.registry import register_env
from ray.rllib.utils.spaces.repeated import Repeated
from gymnasium import Env
from gymnasium import spaces
import numpy as np
from gymnasium import spaces
from collections import defaultdict
import torch
import networkx as nx
from collections import OrderedDict


class TestEnv(Env):
    def __init__(self) -> None:
        
        super().__init__()
        self.bound = 10
        self.observation_space = spaces.Box(
            low=-2*self.bound,
            high=2*self.bound,
            shape=(3,),
        )
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
        )

    def observe(self):
        obs = np.concatenate(
            [
                self.curr_state-self.target_state,
                self.target_state,
                self.curr_state,
            ]
        )
        return obs

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.target_state = np.random.uniform(low=-self.bound, high=self.bound, size=(1,))
        self.curr_state = np.zeros_like(self.target_state)
        self.n_steps = 0
        return self.observe()

    def step(self, action: Any):
        self.curr_state += action
        self.curr_state = np.clip(self.curr_state, -self.bound, self.bound)
        distance_to_target = np.abs(self.curr_state - self.target_state)
        reward = -float(distance_to_target)
        self.n_steps += 1
        return self.observe(), reward, self.n_steps >= 100, {}


def env_creator(env_config):
    return TestEnv()

register_env("test_env", env_creator)
