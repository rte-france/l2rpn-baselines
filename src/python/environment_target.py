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


class Grid2OpEnv(Env):
    def __init__(self, env_name) -> None:
        super().__init__()
        self.env_name = env_name
        self.env = grid2op.make(
            env_name,
            reward_class=LinesCapacityReward,
            backend=LightSimBackend(),
            experimental_read_from_local_dir=True,
        )
        obs = self.env.reset()
        elements_graph = obs.get_elements_graph()
        pyg_graph = to_pyg_graph(elements_graph)

        self.action_space = GymActionSpace(self.env, action_attr_to_keep=["redispatch"])
        self.observation_space = ObservationSpace(pyg_graph, self.env)
        self.n_steps = 0

    def convert_observation_space(self, obs):
        elements_graph = obs.get_elements_graph()
        pyg_graph = to_pyg_graph(elements_graph)
        result = {}
        result["node_features"] = {}
        for node_type, info in pyg_graph.node_items():
            if node_type != "gen":
                continue
            result["node_features"][node_type] = info.x.numpy().astype(np.float32)
        # result["edge_list"] = {}
        # for edge_type, info in pyg_graph.edge_items():
        #     result["edge_list"][edge_type] = info.edge_index.T.numpy()

        # TEST CASE
        result["node_features"]["gen"] = np.concatenate(
            [result["node_features"]["gen"], self.gen_curr, self.gen_targets], axis=1
        ).astype(np.float32)
        # TEST CASE

        assert self.observation_space.contains(result)
        return result

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs = self.env.reset()
        self.n_steps = 0

        ### TEST CASE
        self.gen_targets = np.random.uniform(
            low=-10, high=10, size=(self.action_space["redispatch"].shape[0], 1)
        )
        self.gen_targets = self.gen_targets * self.action_space[
            "redispatch"
        ].high.reshape(-1, 1)
        self.gen_curr = np.zeros_like(self.gen_targets)
        ### TEST CASE

        obs = self.convert_observation_space(obs)

        if not self.observation_space.contains(obs):
            raise Exception("Invalid observation")
        return obs, {}

    def step(self, action: Any):
        # TEST CASE
        self.gen_curr += action["redispatch"].reshape(self.gen_curr.shape)
        reward = -np.sum(action["redispatch"] < 0)
        info = {}
        obs = self.env.current_obs
        # TEST CASE
        self.n_steps += 1
        a = self.action_space.from_gym(action)
        # obs, reward, done, info = self.env.step(a)
        obs = self.convert_observation_space(obs)
        truncated = False
        if not self.observation_space.contains(obs):
            raise Exception("Invalid observation")
        return obs, reward, self.n_steps > 100, truncated, info

    def render(self, mode: str = "human") -> Any:
        return self.env.render(mode=mode)


class ObservationSpace(spaces.Dict):
    def __init__(self, graph, env):
        # do as you please here
        dic = OrderedDict()
        dic["node_features"] = spaces.Dict()

        # Add nodes
        for node_type, info in graph.node_items():
            if node_type != "gen":
                continue
            dic["node_features"][node_type] = spaces.Box(
                low=np.inf,
                high=np.inf,
                shape=info.x.shape,
            )

        # TEST CASE
        dic["node_features"]["gen"] = spaces.Box(
            low=np.inf,
            high=np.inf,
            shape=(graph["gen"].x.shape[0], graph["gen"].x.shape[1] + 2),
        )
        # TEST CASE

        # Add edges
        # dic["edge_list"] = spaces.Dict()
        # for edge_type, _ in graph.edge_items():
        #     num_node_type_source = len(graph[edge_type[0]].x)
        #     num_node_type_target = len(graph[edge_type[2]].x)
        #     dic["edge_list"][edge_type] = Repeated(
        #         spaces.MultiDiscrete([num_node_type_source, num_node_type_target]),
        #         max_len=num_node_type_source * num_node_type_target,
        #     )

        spaces.Dict.__init__(self, dic)

    def to_gym(self, obs):
        # this is this very same function that you need to implement
        # it should have this exact name, take only one observation (grid2op) as input
        # and return a gym object that belong to your space "AGymSpace"
        r = obs.get_elements_graph()
        # Initialize HeteroData
        graph = HeteroData()
        id_map = defaultdict(lambda: defaultdict(int))
        nodes = defaultdict(list)
        edge_types = defaultdict(list)
        edge_features = defaultdict(list)

        # Node processing
        for new_id, (old_id, features) in enumerate(r.nodes(data=True)):
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
        for src, dst, attr in r.edges(data=True):
            src_type, dst_type = r.nodes[src]["type"], r.nodes[dst]["type"]
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


def env_creator(env_config):
    return Grid2OpEnv(env_config["env_name"])


# env = grid2op.make(
#         "l2rpn_case14_sandbox",
#         reward_class=LinesCapacityReward,
#         backend=LightSimBackend(),
# )
# obs = env.reset()
# r = obs.get_elements_graph()


node_data_fields = {
    "substation": [
        "id"
    ],  # All fields: {'id': 1, 'type': 'substation', 'name': 'sub_1', 'cooldown': 0}
    "bus": [
        "v",
        "theta",
        "id",
    ],  # All fields: {'id': 2, 'global_id': 2, 'local_id': 1, 'type': 'bus', 'connected': True, 'v': 142.1, 'theta': -4.190119}
    "load": ["id"],  # {'id': 8, 'type': 'load', 'name': 'load_11_8', 'connected': True}
    "gen": [
        "id",
        "target_dispatch",
        "actual_dispath",
        "gen_p_before_curtal",
        "curtalment_mw",
        "curtailement",
        "curtailment_limit",
        "gen_margin_up",
        "gen_margin_down",
    ],  # {'id': 3, 'type': 'gen', 'name': 'gen_5_3', 'connected': True, 'target_dispatch': 0.0, 'actual_dispatch': 0.0, 'gen_p_before_curtail': 0.0, 'curtailment_mw': 0.0, 'curtailment': 0.0, 'curtailment_limit': 1.0, 'gen_margin_up': 0.0, 'gen_margin_down': 0.0}
    "line": [
        "id",
        "rho",
    ],  # {'id': 1, 'type': 'line', 'name': '0_4_1', 'rho': 0.36042336, 'connected': True, 'timestep_overflow': 0, 'time_before_cooldown_line': 0, 'time_next_maintenance': -1, 'duration_next_maintenance': 0}
    "shunt": [
        "id"
    ],  # {'id': 0, 'type': 'shunt', 'name': 'shunt_8_0', 'connected': True}
}
edge_data_fields = {
    "bus_to_substation": {},
    "load_to_bus": {
        "id",
        "p",
        "q",
        "v",
        "theta",
    },  # {'id': 0, 'type': 'load_to_bus', 'p': 21.9, 'q': 15.4, 'v': 142.1, 'theta': -1.4930121}
    "gen_to_bus": {
        "id",
        "p",
        "q",
        "v",
        "theta",
    },  # {'id': 0, 'type': 'gen_to_bus', 'p': -81.4, 'q': -19.496038, 'v': 142.1, 'theta': -1.4930121}
    "line_to_bus": {
        "id",
        "p",
        "q",
        "v",
        "theta",
        "a",
    },  # {'id': 0, 'type': 'line_to_bus', 'p': 42.346096, 'q': -16.060501, 'v': 142.1, 'a': 184.01027, 'side': 'or', 'theta': 0.0}
    "shunt_to_bus": {
        "id",
        "p",
        "v",
        "q",
    },  # {'id': 0, 'type': 'shunt_to_bus', 'p': -6.938894e-16, 'q': -21.208096, 'v': 21.13022}
}


def to_pyg_graph(graph_nx: nx.DiGraph) -> HeteroData:
    graph = HeteroData()
    id_map = defaultdict(lambda: defaultdict(int))
    nodes = defaultdict(list)
    edge_types = defaultdict(list)
    edge_features = defaultdict(list)

    # Node processing
    for new_id, (old_id, features) in enumerate(graph_nx.nodes(data=True)):
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
    for src, dst, attr in graph_nx.edges(data=True):
        src_type, dst_type = graph_nx.nodes[src]["type"], graph_nx.nodes[dst]["type"]
        edge_type = attr["type"]

        if edge_type not in edge_data_fields:
            raise Exception(f"Edge type {edge_type} not supported")

        edge_types[(src_type, edge_type, dst_type)].append(
            (id_map[src_type][src], id_map[dst_type][dst])
        )
        edge_features[edge_type].append(
            torch.tensor([attr.get(field, 0) for field in edge_data_fields[edge_type]])
        )

    # Populate HeteroData edges and edge features
    for key, vals in edge_types.items():
        graph[key].edge_index = torch.tensor(vals, dtype=torch.long).t().contiguous()
        if len(edge_data_fields[key[1]]) > 0:
            graph[key].edge_attr = torch.stack(edge_features[key[1]])

    return graph


register_env("grid2op_env", env_creator)
