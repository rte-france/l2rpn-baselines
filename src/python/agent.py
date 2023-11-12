from pyexpat import model
from threading import local
from IPython import embed
from numpy import size
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
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import FastRGCNConv

import torch.nn.functional as F

from environment import ObservationSpace


class GraphNet(nn.Module):
    def __init__(self, obs_space, action_space, embed_dim, out_dim):
        super().__init__()
        self.n_dim = action_space["redispatch"].shape[0]  # type: ignore
        self.embed_dim = embed_dim
        self.obs_space = obs_space

        self.embeder = nn.Linear(
            obs_space["node_features"]["gen"].shape[1],
            self.embed_dim,
        )
        self.conv1 = FastRGCNConv(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            num_relations=1,
            aggr="mean",
        )
        # self.layer2 = nn.Linear(
        #     self.embed_dim// 6,
        #     self.embed_dim// 6,
        # )
        self.conv2 = FastRGCNConv(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            num_relations=1,
            aggr="mean",
        )
        self.act = nn.ReLU()
        self.final_layer = nn.Linear(self.embed_dim, out_dim)

    def forward(self, input: HeteroData):
        # obs = input_dict["obs"]["node_features"]["gen"]  # type: ignore
        # graph = self.original_space.dict_to_pyg(input_dict["obs"]) # type: ignore
        # Apply distinct actor policy for each agent
        # Add self loops from "gen" to "gen"
        obs = input["gen"].x
        # obs = input["gen"].x.view((input.num_graphs,-1))
        x = self.embeder(obs)
        # x = self.act(x)
        # x = x.view((input.num_graphs, -1))
        # x = self.layer2(x)
        # x = self.act(x)
        x = self.act(self.conv1(x, input[("gen", "self loops", "gen")].edge_index, edge_type=torch.zeros((input[("gen", "self loops", "gen")].edge_index.shape[1],), dtype=torch.int64, device=input["gen"].x.device)))
        x = self.act(self.conv2(x, input[("gen", "self loops", "gen")].edge_index, edge_type=torch.zeros((input[("gen", "self loops", "gen")].edge_index.shape[1],), dtype=torch.int64, device=input["gen"].x.device)))
        x = self.final_layer(x)
        return x


class ActorCritic(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.n_dim = action_space["redispatch"].shape[0]  # type: ignore
        self.embed_dim = 16

        # Create a list of actor models, one for each agent
        self.original_space = obs_space
        self.actor = GraphNet(obs_space, action_space, self.embed_dim, 2)
        self.special_init(self.actor)

        self.critic = nn.Sequential(
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
                    normc_initializer(0.001)(m.weight)
                    is_last_linear_layer = False
                else:
                    normc_initializer(1.0)(m.weight)

    def forward(self, input: HeteroData, state, seq_lens):
        input[("gen", "self loops", "gen")].edge_index = torch.empty(
            (2, 0), dtype=torch.int64, device=input["gen"].x.device
        )
        input[("gen", "self loops", "gen")].edge_index = add_self_loops(
            input[("gen", "self loops", "gen")].edge_index,
            num_nodes=input["gen"].x.shape[0],
        )[0]
        action = self.actor(input)  # .reshape(-1, self.n_dim, 2)
        mean = action[:, 0]
        log_std = action[:, 1]
        mean = mean.reshape(input.num_graphs, -1)
        log_std = log_std.reshape(input.num_graphs, -1)
        # action = action.reshape(input.num_graphs, -1)
        # mean, log_std = torch.chunk(action.reshape(input.num_graphs, -1), 2, dim=1)

        std = F.softplus(log_std)

        self.val = self.critic(input["gen"].x.view((input.num_graphs, -1))).reshape(-1)

        # flattened_action = action.flatten(start_dim=1)
        return mean, std, []

    def value_function(self):
        return self.val.flatten()

    def act_eval(self, state):
        # if len(state.shape) == 2:
        #     state = state.unsqueeze(0)
        action_mean, action_std, _ = self.forward(state, None, None)
        return action_mean

    def act(self, state):
        # state = state.unsqueeze(0)
        action_mean, action_std, _ = self.forward(state, None, None)
        # action_mean, action_std = torch.chunk(flattened_action, 2, dim=0)
        cov_mat = torch.diag_embed(action_std)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.value_function()

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        # action_means = []
        # action_stds = []
        # state_vals = []
        # for i in range(len(state)):
        #     action_mean, action_var, _ = self.forward(state[i], None, None)
        #     action_means.append(action_mean)
        #     action_stds.append(action_var)
        #     state_vals.append(self.value_function())

        # action_mean = torch.cat(action_means)
        # action_var = torch.cat(action_stds)
        # state_values = torch.cat(state_vals)

        # # action_var = self.action_var.expand_as(action_mean)
        # cov_mat = torch.diag_embed(action_var)
        # dist = MultivariateNormal(action_mean, cov_mat)

        # # For Single Action Environments.
        # # if self.action_dim == 1:
        # #     action = action.reshape(-1, self.action_dim)
        # action_logprobs = dist.log_prob(action)
        # dist_entropy = dist.entropy()

        # state_values = self.value_function()
        # return action_logprobs, state_values, dist_entropy
        action_mean2, action_var2, _ = self.forward(state, None, None)

        # action_var = self.action_var.expand_as(action_mean)
        cov_mat2 = torch.diag_embed(action_var2)
        dist2 = MultivariateNormal(action_mean2, cov_mat2)

        # For Single Action Environments.
        # if self.action_dim == 1:
        #     action = action.reshape(-1, self.action_dim)
        action_logprobs2 = dist2.log_prob(action)
        dist_entropy2 = dist2.entropy()
        state_values2 = self.value_function()
        # print(action_logprobs2.mean(), state_values2.mean(), dist_entropy2.mean())
        return action_logprobs2, state_values2, dist_entropy2


ModelCatalog.register_custom_model("my_torch_model", ActorCritic)

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
        sgd_minibatch_size=1,
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

    print("Starting test training")
    # trainer = config.build()
    # trainer.train()
    print("Starting training")
    tuner = tune.Tuner(
        "PPO",
        run_config=train.RunConfig(
            stop={"training_iteration": 100},
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
            callbacks=[WandbLoggerCallback(project="grid2op")],
            verbose=3,
        ),
        param_space=config,  # type: ignore
    )

    tuner.fit()
    print("Done")
