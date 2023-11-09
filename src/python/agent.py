from pyexpat import model
from threading import local
from IPython import embed
from torch import mode, nn
from ray.rllib.algorithms.ppo import PPOConfig
import imageio
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from ray.rllib.models import ModelCatalog
from tqdm import tqdm
from ray import tune, train
from ray.rllib.models.torch.misc import normc_initializer
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb


import torch.nn.functional as F


class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.n_dim = action_space["redispatch"].shape[0]  # type: ignore
        self.embed_dim = 64

        # Create a list of actor models, one for each agent
        obs_space = obs_space.original_space  # type: ignore
        self.actors = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                obs_space["node_features"]["gen"].shape[0]
                * obs_space["node_features"]["gen"].shape[1],
                self.embed_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 2 * self.n_dim),
        )
        self.special_init(self.actors)

        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                obs_space["node_features"]["gen"].shape[0]
                * obs_space["node_features"]["gen"].shape[1],
                self.embed_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )
        self.special_init(self.critic)

    def special_init(self, module):
        is_last_linear_layer = True
        for m in reversed(list(module.modules())):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                if is_last_linear_layer:
                    normc_initializer(0.01)(m.weight)
                    is_last_linear_layer = False
                else:
                    normc_initializer(1.0)(m.weight)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["node_features"]["gen"]  # type: ignore

        # Apply distinct actor policy for each agent
        action = self.actors(obs)  # .reshape(-1, self.n_dim, 2)

        # mean, log_std = torch.chunk(action, 2, dim=2)

        # std = F.softplus(log_std)
        # action = torch.cat([mean, std], dim=2)

        self.val = self.critic(obs).reshape(-1)

        flattened_action = action.flatten(start_dim=1)
        return flattened_action, []

    def value_function(self):
        return self.val.flatten()


ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)

if __name__ == "__main__":
    import environment

    context = ray.init()
    print(context.dashboard_url)

    env = environment.TestEnv(env_name="l2rpn_case14_sandbox")
    ray.rllib.utils.check_env(env)
    config: PPOConfig = PPOConfig()
    config = config.framework("torch")  # type: ignore
    config = config.environment(  # type: ignore
        env="test_env",
        env_config={"env_name": "l2rpn_case14_sandbox"},
        # normalize_actions=True,
    )
    # config = config.rollouts( # type: ignore
    #     observation_filter="MeanStdFilter",
    # )
    config.rl_module(_enable_rl_module_api=False)
    config = config.training(
        _enable_learner_api=False,
        model={"custom_model": "my_torch_model"},
        # model={"fcnet_hiddens": [64, 64]},
        gamma=0.99,
        vf_clip_param=100,
    )

    # config = config.exploration(
    #     explore=True,
    #     exploration_config={
    #         "type": "StochasticSampling",
    #     },
    # )
    config = config.evaluation(  # type: ignore
        evaluation_interval=10,
        evaluation_num_episodes=10,
    )
    config = config.resources(num_gpus=1).rollouts(num_rollout_workers=4)  # type: ignore

    trainer = config.build()
    trainer.train()

    tuner = tune.Tuner(
        "PPO",
        run_config=train.RunConfig(
            stop={"training_iteration": 100},
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
            callbacks=[WandbLoggerCallback(project="grid2op")],
        ),
        param_space=config,  # type: ignore
    )

    tuner.fit()
    print("Done")
