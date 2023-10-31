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
from tqdm import tqdm
from ray import tune, train


import torch.nn.functional as F
class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.actor = nn.Linear(obs_space.shape[0], action_space.shape[0] * 2)
        self.critic = nn.Linear(obs_space.shape[0], 1)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        action_logits = self.actor(obs)
        mean, log_std = torch.chunk(action_logits, 2, dim=1)
        
        mean = torch.tanh(mean)
        std = F.softplus(log_std)
        action = torch.cat([mean, std], dim=1)

        self.val = self.critic(obs)

        return action, []

    def value_function(self):
        return self.val.flatten()



ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)


if __name__ == "__main__":
    import env_0
    from PPO import PPO
    env = env_0.TestEnv()
    agent = CustomTorchModel(env.observation_space, env.action_space, 1, {}, "test")
    optimzer = PPO(agent,
                action_dim=env.action_space.shape[0],
                state_dim=env.observation_space.shape[0],
                lr_actor=0.0003,
                lr_critic=0.001,
                gamma=1.0,
                K_epochs=80,
                eps_clip=0.2,
                has_continuous_action_space=True,
                )
    for i in tqdm(range(1000)):
        optimzer.update()


    # import env_0

    # context = ray.init(local_mode=True)
    # print(context.dashboard_url)

    # env = env_0.TestEnv()
    # ray.rllib.utils.check_env(env)
    # config = PPOConfig()
    # config = config.environment(
    #     env="test_env"
    # )
    # config.rl_module(_enable_rl_module_api=False)
    # config = config.training(
    #     lr=0.0001,
    #     _enable_learner_api=False,
    #     model={"custom_model": "my_torch_model"},
    # )

    # trainer = config.build()
    # trainer.train()

    # tuner = tune.Tuner(
    #     "PPO",
    #     run_config=train.RunConfig(
    #         stop={"training_iteration": 100000},
    #         checkpoint_config=train.CheckpointConfig(
    #             checkpoint_frequency=1000, checkpoint_at_end=True
    #         ),
    #     ),
    #     param_space=config,
    # )

    # tuner.fit()
    # print("Done")
