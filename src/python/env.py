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
    def __init__(self, env_name) -> None:
        super().__init__()
        self.env_name = env_name
        self.env = grid2op.make(
            env_name,
            reward_class=LinesCapacityReward,
            backend=LightSimBackend(),
            experimental_read_from_local_dir=True,
        )
        self.n_gen = self.env.n_gen
        self.n_agents = 1
        self.observation_space = spaces.Box(
            low=-np.repeat(self.env.observation_space.gen_pmax[:,np.newaxis], 3,axis=1),
            high=np.repeat(self.env.observation_space.gen_pmax[:,np.newaxis], 3,axis=1),
            shape=(self.n_gen, 3,),  # Adjust shape for 3D
        )
        self.action_space = spaces.Box(
            low=-self.env.observation_space.gen_max_ramp_down,
            high=self.env.observation_space.gen_max_ramp_up,
            shape=(self.n_gen,),
        )

    def observe(self):
        obs = np.stack([
            self.curr_state - self.target_state,
            self.target_state,
            self.curr_state,
        ], axis=1)
        assert self.observation_space.contains(obs)
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        np.random.seed(seed)
        self.target_state = np.random.uniform(low=self.env.observation_space.gen_pmin, high=self.env.observation_space.gen_pmax, size=(self.n_gen,)).astype(np.float32)
        self.target_state[self.env.observation_space.gen_max_ramp_up == 0] = 0
        self.curr_state = np.zeros_like(self.target_state).astype(np.float32)
        self.n_steps = 0
        return self.observe(), {}

    def step(self, action: Any):
        initial_distance = np.linalg.norm(self.curr_state - self.target_state)
        self.curr_state += action
        self.curr_state = np.clip(self.curr_state, self.env.observation_space.gen_pmin, self.env.observation_space.gen_pmax)
        new_distance = np.linalg.norm(self.curr_state - self.target_state)
        reward = initial_distance - new_distance
        self.n_steps += 1
        return self.observe(), reward, self.n_steps >= 100, False, {}

    def render(self, mode='human'):
        fig, axs = plt.subplots(self.n_gen, 1, figsize=(10, self.n_gen * 2), tight_layout=True)
        for i, ax in enumerate(axs):
            ax.set_xlim(self.env.observation_space.gen_pmin.min()-10, self.env.observation_space.gen_pmax.max()+10)
            ax.scatter(self.target_state[i], 0.5, c='red', label=f'Gen {i} Target')
            ax.scatter(self.curr_state[i], 0.5, c='blue', label=f'Gen {i} Agent')
            ax.legend()
            ax.yaxis.set_visible(False)
        
        if mode == 'human':
            plt.show()
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_arr = np.array(Image.open(buf))
            plt.close(fig)
            return img_arr


def env_creator(env_config):
    return TestEnv(env_config["env_name"])

register_env("test_env", env_creator)
