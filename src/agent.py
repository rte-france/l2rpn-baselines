from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from torch_geometric.nn import RGATConv
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import numpy as np
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.models.modelv2 import restore_original_dimensions
from typing import Mapping, Any
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import from_networkx
from collections import defaultdict
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.models import ModelCatalog


class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.policy = nn.ModuleDict()
        for key in self.action_space.keys():
            # One dim for mean, one dim for std
            self.policy[key] = nn.Sequential(
                nn.Linear(2777, 2*self.action_space["redispatch"].shape[0]),
                nn.Sigmoid(),
            )
        self.value_model = nn.Sequential(
            nn.Linear(2777, 1),
        )


    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"]
        self.val = self.value_model(obs).flatten()
        logits = {}
        for key in self.action_space.keys():
            logits[key] = self.policy[key](obs)
        flattened_logits = torch.cat([logits[key] for key in self.action_space.keys()], dim=1)
        return flattened_logits, []
    
    def value_function(self):
        return self.val

ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)


if __name__ == "__main__":
    from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
    import environment

    ray.init(local_mode=True, num_cpus=1, num_gpus=0)
    config = PPOConfig()
    config = config.environment(
        env="grid2op_env", env_config={"env_name": "l2rpn_case14_sandbox"}
    )
    env = environment.Grid2OpEnv("l2rpn_case14_sandbox")
    config.rl_module( _enable_rl_module_api=False)
    config = config.training(
        _enable_learner_api=False,
        model={"custom_model": "my_torch_model"},
    )

    # Set num workers to 1
    alg = config.build()
    alg.train()
