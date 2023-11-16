from typing import Tuple
from torch import nn
import torch
from ray.rllib.models.torch.misc import normc_initializer
from torch.distributions import MultivariateNormal
from torch_geometric.data import HeteroData
from torch_geometric.nn import EdgeConv
import torch.nn.functional as F


class GraphNet(nn.Module):
    def __init__(self, obs_space, action_space, embed_dim, out_dim):
        super().__init__()
        self.n_dim = action_space["redispatch"].shape[0]  # type: ignore
        self.embed_dim = embed_dim
        self.obs_space = obs_space

        self.node_embeder = nn.ModuleDict()
        for node_type in obs_space["node_features"]:
            self.node_embeder[node_type] = nn.Linear(
                obs_space["node_features"][node_type].shape[1], embed_dim
            )
        self.conv1 = EdgeConv(
            nn=nn.Linear(2 * self.embed_dim, self.embed_dim),
            aggr="mean",
        )
        self.act = nn.ReLU()
        self.final_layer = nn.Linear(2 * self.embed_dim, out_dim)
        self.val_layer = nn.Linear(2 * 6 * self.embed_dim, 1)
        normc_initializer(0.001)(self.final_layer.weight)
        normc_initializer(0.001)(self.val_layer.weight)

    def forward(
        self, input: HeteroData
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for node_type in self.obs_space["node_features"]:
            input[node_type].x = self.node_embeder[node_type](
                input[node_type].x.float()
            )
        input_homogeneous = input.to_homogeneous()
        skip_connection = input_homogeneous.x[input_homogeneous.node_type == 3]
        input_homogeneous.x = self.act(  # type: ignore
            self.conv1(
                input_homogeneous.x,
                input_homogeneous.edge_index,
            ),
        )
        embeddings = input_homogeneous.x[input_homogeneous.node_type == 3]
        embeddings = torch.cat([embeddings, skip_connection], dim=1)
        value_vector = self.val_layer(embeddings.reshape(input.num_graphs, -1))
        action = self.final_layer(embeddings)
        action_mean = action[:, 0]
        action_std = action[:, 1]
        action_std = F.softplus(action_std)
        return action_mean, action_std, value_vector


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        nn.Module.__init__(self)
        self.n_dim = action_space["redispatch"].shape[0]  # type: ignore
        self.embed_dim = 16

        self.original_space = obs_space
        self.actor = GraphNet(obs_space, action_space, self.embed_dim, 2)

    def forward(self, input: HeteroData, state, seq_lens):
        action_mean, action_std, self.value_vector = self.actor(input.clone())
        return action_mean, action_std

    def value_function(self):
        return self.value_vector.flatten()

    def act_eval(self, state):
        action_mean, action_std = self.forward(state, None, None)
        return action_mean

    def act(self, state):
        action_mean, action_std = self.forward(state, None, None)
        cov_mat = torch.diag_embed(action_std)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.value_function()

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_mean, action_var = self.forward(state, None, None)

        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.value_function()
        return action_logprobs, state_values, dist_entropy


def special_init(module):
    is_last_linear_layer = True
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)
            if is_last_linear_layer:
                normc_initializer(0.001)(m.weight)
                is_last_linear_layer = False
            else:
                normc_initializer(1.0)(m.weight)
