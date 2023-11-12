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
        self.conv2 = FastRGCNConv(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            num_relations=1,
            aggr="mean",
        )
        self.act = nn.ReLU()
        self.final_layer = nn.Linear(self.embed_dim, out_dim)

    def forward(self, input: HeteroData) -> torch.Tensor:
        obs = input["gen"].x
        x = self.embeder(obs)
        x = self.act(
            self.conv1(
                x,
                input[("gen", "self loops", "gen")].edge_index,
                edge_type=torch.zeros(
                    (input[("gen", "self loops", "gen")].edge_index.shape[1],),
                    dtype=torch.int64,
                    device=input["gen"].x.device,
                ),
            )
        )
        x = self.act(
            self.conv2(
                x,
                input[("gen", "self loops", "gen")].edge_index,
                edge_type=torch.zeros(
                    (input[("gen", "self loops", "gen")].edge_index.shape[1],),
                    dtype=torch.int64,
                    device=input["gen"].x.device,
                ),
            )
        )
        x = self.final_layer(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        nn.Module.__init__(self)
        self.n_dim = action_space["redispatch"].shape[0]  # type: ignore
        self.embed_dim = 16

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

        return mean, std, []

    def value_function(self):
        return self.val.flatten()

    def act_eval(self, state):
        action_mean, action_std, _ = self.forward(state, None, None)
        return action_mean

    def act(self, state):
        action_mean, action_std, _ = self.forward(state, None, None)
        # action_mean, action_std = torch.chunk(flattened_action, 2, dim=0)
        cov_mat = torch.diag_embed(action_std)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.value_function()

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_mean, action_var, _ = self.forward(state, None, None)

        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.value_function()
        return action_logprobs, state_values, dist_entropy
