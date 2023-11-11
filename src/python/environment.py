from typing import Dict
from typing import Any
from torch_geometric.data import HeteroData
from torch_geometric.data import Batch
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
from utils import (
    to_pyg_graph,
    edge_data_fields,
    node_data_fields,
    node_observation_space,
)
from gymnasium import register
from grid2op.Environment import Environment


class TestEnv(Env):
    def __init__(self, env_name: str = "l2rpn_case14_sandbox") -> None:
        super().__init__()
        self.env_name = env_name
        self.env = grid2op.make(
            env_name,
            reward_class=LinesCapacityReward,
            backend=LightSimBackend(),
            experimental_read_from_local_dir=True,
        )
        self.n_gen = self.env.n_gen

        # Observation space normalization factors
        self.gen_pmax = self.env.observation_space.gen_pmax
        self.gen_pmin = self.env.observation_space.gen_pmin
        assert np.all(self.gen_pmax >= self.gen_pmin) and np.all(self.gen_pmin >= 0)  # type: ignore

        # Observation space observation
        self.observation_space: ObservationSpace = ObservationSpace(self.env)
        self.elements_graph = self.env.reset().get_elements_graph()

        # Action space
        self.action_space = spaces.Dict()
        self.action_space["redispatch"] = spaces.Box(
            low=-1, high=1, shape=(self.n_gen,), dtype=np.float32
        )

        # Action space normalization factor
        self.action_norm_factor = np.maximum(
            self.env.observation_space.gen_max_ramp_up,  # type: ignore
            -self.env.observation_space.gen_max_ramp_down,  # type: ignore
        )

    def normalize_obs(self, obs):
        obs_diff_range = self.gen_pmax - self.gen_pmin  # type: ignore
        obs[:, 0] = obs[:, 0] / obs_diff_range
        obs[:, 1] = (obs[:, 1] - self.gen_pmin) / obs_diff_range
        obs[:, 2] = (obs[:, 2] - self.gen_pmin) / obs_diff_range
        return obs

    def denormalize_action(self, action):
        action = action * self.action_norm_factor
        # action["redispatch"] = action["redispatch"] * self.action_norm_factor
        return action

    def observe(self):
        obs = np.stack(
            [
                self.curr_state - self.target_state,
                self.target_state,
                self.curr_state,
            ],
            axis=1,
        )
        obs = self.normalize_obs(obs)
        obs_original = self.observation_space.grid2op_to_pyg(self.elements_graph)
        obs_original["gen"].x = torch.tensor(obs)
        # obs = self.observation_space.pyg_to_dict(obs_original)
        # if not self.observation_space.contains(obs):
        #     raise Exception("Invalid observation")
        return obs_original

    def set_target_state(self):
        self.target_state = np.random.uniform(  # type: ignore
            low=self.env.observation_space.gen_pmin,  # type: ignore
            high=self.env.observation_space.gen_pmax,  # type: ignore
            size=(self.n_gen,),
        ).astype(np.float32)
        self.target_state[self.env.observation_space.gen_max_ramp_up == 0] = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        np.random.seed(seed)
        self.set_target_state()
        self.curr_state = np.zeros_like(self.target_state).astype(np.float32)
        self.n_steps = 0
        return self.observe(), {}

    def step(self, action):
        action = self.denormalize_action(action)
        initial_distance = np.linalg.norm(self.curr_state - self.target_state)
        self.curr_state += action
        self.curr_state = np.clip(
            self.curr_state,
            self.env.observation_space.gen_pmin,
            self.env.observation_space.gen_pmax,
        )
        new_distance = np.linalg.norm(self.curr_state - self.target_state)
        reward = initial_distance - new_distance
        self.n_steps += 1
        # if self.n_steps in [50]:
        #     self.set_target_state()
        done = self.n_steps >= 100
        return self.observe(), reward, done, False, {}

    def render(self, mode="human"):
        fig, axs = plt.subplots(
            self.n_gen, 1, figsize=(10, self.n_gen * 2), tight_layout=True
        )
        for i, ax in enumerate(axs):
            ax.set_xlim(
                self.env.observation_space.gen_pmin.min(),  # type: ignore
                self.env.observation_space.gen_pmax.max(),  # type: ignore
            )
            ax.scatter(self.target_state[i], 0.5, c="red", label=f"Gen {i} Target")
            ax.scatter(self.curr_state[i], 0.5, c="blue", label=f"Gen {i} Agent")
            ax.legend()
            ax.yaxis.set_visible(False)

        if mode == "human":
            plt.show()
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_arr = np.array(Image.open(buf))
            plt.close(fig)
            return img_arr


class ObservationSpace(spaces.Dict):
    def __init__(self, env: Environment):
        graph = self.grid2op_to_pyg(env.reset().get_elements_graph())

        dic = OrderedDict()
        dic["node_features"] = spaces.Dict()
        for node_type, _ in graph.node_items():
            dic["node_features"][node_type] = node_observation_space[node_type](
                len(graph[node_type].x)
            )
        self.n_gen = env.n_gen

        # Add edges
        dic["edge_list"] = spaces.Dict()
        for edge_type, _ in graph.edge_items():
            num_node_type_source = len(graph[edge_type[0]].x)
            num_node_type_target = len(graph[edge_type[2]].x)
            dic["edge_list"][edge_type] = Repeated(  # type: ignore
                spaces.MultiDiscrete([num_node_type_source, num_node_type_target]),
                max_len=num_node_type_source * num_node_type_target,
            )

        spaces.Dict.__init__(self, dic)

    def pyg_to_dict(self, pyg_graph: HeteroData) -> Dict[str, Dict[str, np.ndarray]]:
        result = OrderedDict()
        result["node_features"] = OrderedDict()
        for node_type, info in pyg_graph.node_items():
            result["node_features"][node_type] = info.x.numpy().astype(np.float32)

        result["edge_list"] = OrderedDict()
        for edge_type, info in pyg_graph.edge_items():
            result["edge_list"][edge_type] = info.edge_index.numpy().T

        return result

    def grid2op_to_pyg(self, elements_graph: nx.DiGraph) -> HeteroData:
        # Initialize HeteroData
        graph = HeteroData()
        id_map = defaultdict(lambda: defaultdict(int))
        nodes = defaultdict(list)
        edge_types = defaultdict(list)
        edge_features = defaultdict(list)

        # Node processing
        for new_id, (old_id, features) in enumerate(elements_graph.nodes(data=True)):
            node_type = features["type"]
            id_map[node_type][old_id] = len(id_map[node_type])
            nodes[node_type].append(
                torch.tensor(
                    [features.get(field, 0) for field in node_data_fields[node_type]]
                )
            )

        # Populate HeteroData nodes
        for key, vals in nodes.items():
            graph[key].x = torch.stack(vals)

        # Initialize dictionaries to hold edge features
        edge_features = defaultdict(list)

        # Edge processing
        for src, dst, attr in elements_graph.edges(data=True):
            src_type, dst_type = (
                elements_graph.nodes[src]["type"],
                elements_graph.nodes[dst]["type"],
            )
            edge_type = attr["type"]

            if edge_type not in edge_data_fields:
                raise Exception(f"Edge type {edge_type} not supported")

            edge_types[(src_type, edge_type, dst_type)].append(
                (id_map[src_type][src], id_map[dst_type][dst])
            )
            edge_features[edge_type].append(
                torch.tensor(
                    [attr.get(field, 0) for field in edge_data_fields[edge_type]]
                )
            )

        # Populate HeteroData edges and edge features
        for key, vals in edge_types.items():
            graph[key].edge_index = (
                torch.tensor(vals, dtype=torch.long).t().contiguous()
            )
            if len(edge_data_fields[key[1]]) > 0:
                graph[key].edge_attr = torch.stack(edge_features[key[1]])

        return graph

    def dict_to_pyg(self, graph_dict: OrderedDict):
        batch_size = graph_dict["node_features"]["gen"].shape[0]
        graphs = []
        for i in range(batch_size):
            pyg_graph = HeteroData()

            # Convert node features back to tensor
            for node_type, features in graph_dict["node_features"].items():
                pyg_graph[node_type].x = features[i]
            graphs.append(pyg_graph)

        # Convert edge index arrays back to tensors
        for edge_type, edge_indices in graph_dict["edge_list"].items():
            edges = edge_indices.unbatch_all()
            for i in range(batch_size):
                graphs[i][edge_type].edge_index = edges[i]

        batched_graph = Batch.from_data_list(graphs)  # type: ignore
        return batched_graph


def env_creator(env_config: dict[str, Any]) -> TestEnv:
    return TestEnv(env_config["env_name"])


register("test_env", TestEnv)

register_env("test_env", env_creator)
