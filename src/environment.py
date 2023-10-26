from typing import Any
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
import grid2op
from grid2op.Reward import LinesCapacityReward
from grid2op.Chronics import MultifolderWithCache
from lightsim2grid import LightSimBackend
from grid2op.gym_compat import GymEnv
from grid2op.gym_compat import BoxGymActSpace
from ray.tune.registry import register_env
from gymnasium import Env
from gymnasium import spaces
import numpy as np


class Grid2OpEnv(Env):
    def __init__(self, env_name) -> None:
        super().__init__()
        self.env_name = env_name
        env = grid2op.make(
            env_name,
            reward_class=LinesCapacityReward,
            backend=LightSimBackend(),
        )
        self.gym_env = GymEnv(env,action_attr_to_keep=["redispatch"])
        # for i in self.gym_env.action_space:
        #     if isinstance(self.gym_env.action_space[i], spaces.MultiBinary):
        #         self.gym_env.action_space[i] = spaces.MultiDiscrete(
        #             np.ones(self.gym_env.action_space[i].n) * 2
        #         )

        self.action_space = self.gym_env.action_space
        self.observation_space = self.gym_env.observation_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        return self.gym_env.reset()
    
    def step(self, action: Any):
        return self.gym_env.step(action)
    
    def render(self, mode: str = "human") -> Any:
        return self.gym_env.render()

def env_creator(env_config):
    return Grid2OpEnv(env_config["env_name"])

# env = grid2op.make(
#         "l2rpn_case14_sandbox",
#         reward_class=LinesCapacityReward,
#         backend=LightSimBackend(),
# )
# obs = env.reset()
# r = obs.get_elements_graph()

# # Initialize HeteroData
# node_data_fields = {
#     "substation": ['id'], # All fields: {'id': 1, 'type': 'substation', 'name': 'sub_1', 'cooldown': 0}
#     "bus": ['v','theta','id'], # All fields: {'id': 2, 'global_id': 2, 'local_id': 1, 'type': 'bus', 'connected': True, 'v': 142.1, 'theta': -4.190119}
#     "load": ['id'], #{'id': 8, 'type': 'load', 'name': 'load_11_8', 'connected': True}
#     "gen":['id','target_dispatch','actual_dispath','gen_p_before_curtal','curtalment_mw','curtailement','curtailment_limit','gen_margin_up','gen_margin_down'], # {'id': 3, 'type': 'gen', 'name': 'gen_5_3', 'connected': True, 'target_dispatch': 0.0, 'actual_dispatch': 0.0, 'gen_p_before_curtail': 0.0, 'curtailment_mw': 0.0, 'curtailment': 0.0, 'curtailment_limit': 1.0, 'gen_margin_up': 0.0, 'gen_margin_down': 0.0}
#     "line": ['id', 'rho'], #{'id': 1, 'type': 'line', 'name': '0_4_1', 'rho': 0.36042336, 'connected': True, 'timestep_overflow': 0, 'time_before_cooldown_line': 0, 'time_next_maintenance': -1, 'duration_next_maintenance': 0}
#     "shunt": ['id'], #{'id': 0, 'type': 'shunt', 'name': 'shunt_8_0', 'connected': True}
# }
# edge_data_fields = {
#     "bus_to_substation": {},
#     "load_to_bus": {'id','p','q','v','theta'}, # {'id': 0, 'type': 'load_to_bus', 'p': 21.9, 'q': 15.4, 'v': 142.1, 'theta': -1.4930121}
#     "gen_to_bus": {'id','p','q','v','theta'}, # {'id': 0, 'type': 'gen_to_bus', 'p': -81.4, 'q': -19.496038, 'v': 142.1, 'theta': -1.4930121}
#     "line_to_bus": {'id','p','q','v','theta', 'a'}, # {'id': 0, 'type': 'line_to_bus', 'p': 42.346096, 'q': -16.060501, 'v': 142.1, 'a': 184.01027, 'side': 'or', 'theta': 0.0}
#     "shunt_to_bus": {'id','p','v','q'} # {'id': 0, 'type': 'shunt_to_bus', 'p': -6.938894e-16, 'q': -21.208096, 'v': 21.13022}
# }
# graph = HeteroData()
# id_map = defaultdict(lambda: defaultdict(int))
# nodes = defaultdict(list)
# edge_types = defaultdict(list)
# edge_features = defaultdict(list)

# # Node processing
# for new_id, (old_id, features) in enumerate(r.nodes(data=True)):
#     node_type = features['type']
#     id_map[node_type][old_id] = len(id_map[node_type])
#     nodes[node_type].append(torch.tensor([features.get(field, 0) for field in node_data_fields[node_type]]))

# # Populate HeteroData nodes
# for key, vals in nodes.items():
#     graph[key].x = torch.stack(vals)

# # Initialize dictionaries to hold edge features
# edge_features = defaultdict(list)

# # Edge processing
# for src, dst, attr in r.edges(data=True):
#     src_type, dst_type = r.nodes[src]['type'], r.nodes[dst]['type']
#     edge_type = attr["type"]

#     if edge_type not in edge_data_fields:
#         raise Exception(f"Edge type {edge_type} not supported")

#     edge_types[(src_type, edge_type, dst_type)].append((id_map[src_type][src], id_map[dst_type][dst]))
#     edge_features[edge_type].append(torch.tensor([attr.get(field, 0) for field in edge_data_fields[edge_type]]))

# # Populate HeteroData edges and edge features
# for key, vals in edge_types.items():
#     graph[key].edge_index = torch.tensor(vals, dtype=torch.long).t().contiguous()
#     if len(edge_data_fields[key[1]]) > 0:
#         graph[key].edge_attr = torch.stack(edge_features[key[1]])


register_env("grid2op_env", env_creator)
