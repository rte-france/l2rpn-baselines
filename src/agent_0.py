from torch import nn
from ray.rllib.algorithms.ppo import PPOConfig
import imageio
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from ray.rllib.models import ModelCatalog
from tqdm import tqdm
from ray import tune, train
from ray.rllib.models.torch.misc import normc_initializer


import torch.nn.functional as F
class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.n_dim = action_space.shape[0]
        
        # Create a list of actor models, one for each agent
        self.actors = nn.Sequential(
                            nn.Linear(obs_space.shape[0]*obs_space.shape[1], 64),
                            nn.ReLU(),
                            nn.Linear(64, 64),
                            nn.ReLU(),
                            nn.Linear(64, 2*self.n_dim),
                        )
        # Set bias to 0
        for param in self.actors.parameters():
            nn.init.zeros_(param)
        normc_initializer(1.0)(self.actors[0].weight)
        normc_initializer(1.0)(self.actors[2].weight)
        normc_initializer(0.01)(self.actors[4].weight)
        
        self.critic = nn.Sequential(
                            nn.Linear(obs_space.shape[0]*obs_space.shape[1], 64),
                            nn.ReLU(),
                            nn.Linear(64, 64),
                            nn.ReLU(),
                            nn.Linear(64, 1),
                        )
        # Set bias to 0
        for param in self.critic.parameters():
            nn.init.zeros_(param)
        normc_initializer(1.0)(self.critic[0].weight)
        normc_initializer(1.0)(self.critic[2].weight)
        normc_initializer(0.01)(self.critic[4].weight)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"]

        # Apply distinct actor policy for each agent
        action = self.actors(obs)# .reshape(-1, self.n_dim, 2)
        
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
    import env_0

    context = ray.init()
    print(context.dashboard_url)

    env = env_0.TestEnv()
    ray.rllib.utils.check_env(env)
    config:PPOConfig = PPOConfig()
    config = config.framework("torch") # type: ignore
    config = config.environment( # type: ignore
        env="test_env"
    )
    config.rl_module(_enable_rl_module_api=False)
    config = config.training(
        _enable_learner_api=False,
        model={"custom_model": "my_torch_model"},
        gamma=0.95,
        vf_clip_param=100,
    )
    # config = config.exploration(
    #     explore=True,
    #     exploration_config={
    #         "type": "StochasticSampling",
    #     }
    # )
    config = config.evaluation( # type: ignore
        evaluation_interval=10,
        evaluation_num_episodes=10,
    )
    config = config.resources(num_gpus=1).rollouts(num_rollout_workers=4) # type: ignore

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
        ),
        param_space=config,
    )

    tuner.fit()
    # Create a movie of the agent's performance
    env = env_0.TestEnv()
    env.reset()
    frames = []
    for i in tqdm(range(100)):
        action = trainer.compute_single_action(env.observe())
        obs, reward, done, terminated, info = env.step(action)
        frames.append(env.render(mode="rgb_array"))
        if done:
            break
    imageio.mimsave("movie.gif", frames)
    print("Done")
