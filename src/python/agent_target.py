from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from torch import nn

from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import RGCNConv
from ray.rllib.models import ModelCatalog
from ray import tune, train


class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.policy = nn.ModuleDict()
        self.node_embedder = nn.ModuleDict()
        self.embedding_size = 16
        for node_type in self.obs_space.original_space["node_features"]:
            self.node_embedder[node_type] = nn.Sequential(
                nn.Linear(
                    self.obs_space.original_space["node_features"][node_type].shape[1],
                    self.embedding_size,
                ),
                nn.ReLU(),
            )
        self.conv1 = RGCNConv(
            [self.embedding_size]
            * self.obs_space.original_space["node_features"]["gen"].shape[0],
            self.embedding_size,
            4,
        )
        self.action_model = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, 2),
        )
        self.param = nn.Parameter(torch.tensor([-1.0, 1.0]))
        self.value_model = nn.Sequential(
            nn.Linear(
                self.embedding_size
                * self.obs_space.original_space["node_features"]["gen"].shape[0],
                1,
            ),
        )

    def forward(self, input_dict, state, seq_lens):
        batch_size = len(input_dict)
        graphs = [HeteroData() for _ in range(batch_size)]
        for i in range(batch_size):
            # for edge_type in input_dict["obs"]["edge_list"]:
            #     if input_dict["obs"]["edge_list"][edge_type].lengths[i] == 0:
            #         continue
            #     graphs[i][edge_type] = input_dict["obs"]["edge_list"][edge_type][i]

            for node_type in input_dict["obs"]["node_features"]:
                graphs[i][node_type].x = self.node_embedder[node_type](
                    input_dict["obs"]["node_features"][node_type][i]
                )

        batch = Batch.from_data_list(graphs)
        # for src, edge_type, dst in batch.edge_types:
        #     x = self.conv1((batch[src].x,batch[dst].x), batch.edge_index, batch.edge_type)

        action = {}
        gen_embeds = torch.stack([i["gen"].x for i in batch.to_data_list()])
        action["redispatch"] = self.param.repeat(
            (gen_embeds.shape[0], gen_embeds.shape[1], 1)
        )  # self.action_model(gen_embeds).squeeze()
        if len(action["redispatch"].shape) == 2:  # batch size 1
            action["redispatch"] = action["redispatch"].unsqueeze(0)
        action["redispatch"][:, :, 0] = torch.sigmoid(
            action["redispatch"][:, :, 0]
        ) * torch.tensor(self.action_space["redispatch"].high)
        action["redispatch"][:, :, 1] = torch.sigmoid(
            action["redispatch"][:, :, 1]
        ) * torch.tensor(self.action_space["redispatch"].high / 2)
        self.val = self.value_model(gen_embeds.flatten(1)).reshape(-1)
        flattened_actions = torch.cat([action[t] for t in self.action_space])
        flattened_actions = flattened_actions.reshape(batch_size, -1)
        return flattened_actions, []

    def value_function(self):
        return self.val


ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)


if __name__ == "__main__":
    import environment

    context = ray.init(local_mode=True)
    print(context.dashboard_url)

    env = environment.Grid2OpEnv("l2rpn_case14_sandbox")
    ray.rllib.utils.check_env(env)
    config = PPOConfig()
    config = config.environment(
        env="grid2op_env", env_config={"env_name": "l2rpn_case14_sandbox"}
    )
    env = environment.Grid2OpEnv("l2rpn_case14_sandbox")
    config.rl_module(_enable_rl_module_api=False)
    config = config.training(
        _enable_learner_api=False,
        model={"custom_model": "my_torch_model"},
    )

    trainer = config.build()
    trainer.train()

    tuner = tune.Tuner(
        "PPO",
        run_config=train.RunConfig(
            stop={"training_iteration": 100000},
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=1000, checkpoint_at_end=True
            ),
        ),
        param_space=config,
    )

    tuner.fit()
    print("Done")
