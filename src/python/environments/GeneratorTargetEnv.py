from typing import Any
from torch_geometric.data import HeteroData
import grid2op
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
from ray.rllib.utils.spaces.repeated import Repeated
from gymnasium import Env
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image
from gymnasium import spaces
from collections import defaultdict
import torch
import networkx as nx
from collections import OrderedDict
from utils import (
    edge_data_fields,
    node_data_fields,
)
from grid2op.Environment import Environment
from torch_geometric.transforms import ToUndirected, AddSelfLoops

MIN_POWER_VALUE = 0
MAX_POWER_VALUE = 100


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
        self.gen_pmax = torch.tensor(self.env.observation_space.gen_pmax)
        self.gen_pmin = torch.tensor(self.env.observation_space.gen_pmin)
        assert torch.all(self.gen_pmax >= self.gen_pmin) and torch.all(
            self.gen_pmin >= 0
        )  # type: ignore

        # Observation space observation
        self.observation_space: ObservationSpace = ObservationSpace(self.env)
        self.elements_graph = self.env.reset().get_elements_graph()
        self.elements_graph_pyg = self.observation_space.grid2op_to_pyg(
            self.elements_graph
        )

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

    def denormalize_action(self, action):
        action = action * self.action_norm_factor
        # action["redispatch"] = action["redispatch"] * self.action_norm_factor
        return action

    def get_neighbors(self, gen_id):
        # This function returns the indices of all neighbors of the given generator node
        neighbors = []
        for edge in self.elements_graph_pyg["rev_gen_to_bus"].edge_index.T:
            if edge[1] == gen_id:
                neighbors.append(edge[0].item())
        return neighbors

    def set_observations(self, obs: HeteroData):
        obs["gen"].x = torch.tensor(
            np.stack(
                [
                    self.curr_state - self.target_state,
                    self.target_state,
                    self.curr_state,
                ],
                axis=1,
            )
        )
        obs["bus"].x = torch.tensor(
            np.stack(
                [
                    self.load_states["bus"],
                ],
                axis=1,
            )
        )
        return obs

    def set_target_state(self):
        self.target_state = torch.tensor(np.zeros(self.n_gen, dtype=np.float32))
        # Compute the mean state of neighbors for each generator
        for gen_id in range(self.n_gen):
            neighbors = self.get_neighbors(gen_id)
            if neighbors:
                # Assuming all nodes have a unified indexing in self.load_states
                neighbor_states = torch.tensor(
                    [self.load_states["bus"][node_id] for node_id in neighbors]
                )
                self.target_state[gen_id] = torch.mean(neighbor_states)
            else:
                # If a generator has no neighbors, use 0 as the target state
                self.target_state[gen_id] = 0

    def observe(self):
        obs = self.observation_space.grid2op_to_pyg(self.elements_graph)
        obs = self.set_observations(obs)
        return obs

    def set_loads_sates(self):
        self.load_states = {}
        for node_type in self.elements_graph_pyg.node_types:
            self.load_states[node_type] = torch.tensor(
                np.random.uniform(
                    low=MIN_POWER_VALUE,
                    high=MAX_POWER_VALUE,
                    size=(len(self.elements_graph_pyg[node_type].x),),
                ).astype(np.float32)
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self.set_loads_sates()
        self.set_target_state()
        self.curr_state = torch.zeros_like(self.target_state)
        self.n_steps = 0
        return self.observe(), {}

    def compute_distance(self):
        return torch.abs(self.curr_state - self.target_state).sum()

    def step(self, action):
        action = self.denormalize_action(action)
        initial_distance = self.compute_distance()
        self.curr_state += action
        self.curr_state = torch.clip(
            self.curr_state,
            self.gen_pmin,
            self.gen_pmax,
        )
        new_distance = self.compute_distance()
        reward = initial_distance - new_distance
        self.n_steps += 1
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


node_observation_space = OrderedDict(
    {
        "substation": lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 1), dtype=np.float32
        ),
        "bus": lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 1), dtype=np.float32
        ),
        "load": lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 1), dtype=np.float32
        ),
        "gen": lambda n_lements: spaces.Box(
            low=-1, high=1, shape=(n_lements, 3), dtype=np.float32
        ),
        "line": lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 2), dtype=np.float32
        ),
        "shunt": lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 1), dtype=np.float32
        ),
    }
)


class ObservationSpace(spaces.Dict):
    def __init__(self, env: Environment):
        self.add_self_loops = AddSelfLoops()
        self.to_undirected = ToUndirected()
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

        graph = self.to_undirected(graph)

        for node_type in graph.node_types:
            graph[
                (node_type, f"self loops {node_type}", node_type)
            ].edge_index = torch.empty(
                (2, 0), dtype=torch.int64, device=graph[node_type].x.device
            )

        graph = self.add_self_loops(graph)

        return graph
