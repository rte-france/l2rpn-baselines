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
import matplotlib.pyplot as plt
from gymnasium import spaces
import io
import numpy as np
from PIL import Image
from gymnasium import spaces
from collections import defaultdict
import torch
import networkx as nx
from collections import OrderedDict


class TestEnv(Env):
    def __init__(self) -> None:
        super().__init__()
        self.bound = 10
        self.n_dim = 2
        self.n_agents = 1
        self.observation_space = spaces.Box(
            low=-3*self.bound,
            high=3*self.bound,
            shape=(3, self.n_dim,),
        )
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_dim,),
        )

    def observe(self):
        obs = np.stack([
            self.curr_state - self.target_state,
            self.target_state,
            self.curr_state,
        ])
        assert self.observation_space.contains(obs)
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.target_state = np.random.uniform(low=-self.bound, high=self.bound, size=(self.n_dim,)).astype(np.float32)
        self.curr_state = np.zeros_like(self.target_state).astype(np.float32)
        self.n_steps = 0
        return self.observe(), {}

    def step(self, action: Any):
        initial_distance = np.linalg.norm(self.curr_state - self.target_state)
        self.curr_state += action
        self.curr_state = np.clip(self.curr_state, -2*self.bound, 2*self.bound)
        new_distance = np.linalg.norm(self.curr_state - self.target_state)
        reward = initial_distance - new_distance
        # reward = min(2*reward,reward)
        self.n_steps += 1
        return self.observe(), reward, self.n_steps >= 100, False, {}


    def render(self, mode='human'):
        fig, ax = plt.subplots(tight_layout=True)
        ax.set_xlim(-3 * self.bound, 3 * self.bound)
        ax.set_ylim(-3 * self.bound, 3 * self.bound)

        # Draw target state
        ax.scatter(self.target_state[0], self.target_state[1], c='red', label='Target')

        # Draw current state
        ax.scatter(self.curr_state[0], self.curr_state[1], c='blue', label='Agent')

        ax.legend()

        if mode == 'human':
            plt.show()
        elif mode == 'rgb_array':
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_arr = np.array(Image.open(buf))
            plt.close(fig)
            return img_arr


def env_creator(env_config):
    return TestEnv()

register_env("test_env", env_creator)
