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

        gen_embeddings = input_homogeneous.x[input_homogeneous.node_type == 3]
        gen_embeddings = torch.cat([gen_embeddings, skip_connection], dim=1)
        value = gen_embeddings.reshape(input.num_graphs, -1)
        value = self.val_layer(value)
        action = self.final_layer(gen_embeddings)
        action_mean = action[:, 0].reshape(input.num_graphs, -1)
        action_std = F.softplus(action[:, 1]).reshape(input.num_graphs, -1)
        return action_mean, action_std, value

    def get_backbone_params(self):
        return list(self.node_embeder.parameters()) + list(self.conv1.parameters())

    def get_actor_params(self):
        return self.final_layer.parameters()

    def get_critic_params(self):
        return self.val_layer.parameters()


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        nn.Module.__init__(self)
        self.n_dim = action_space["redispatch"].shape[0]  # type: ignore
        self.embed_dim = 16

        self.original_space = obs_space
        self.agent = GraphNet(obs_space, action_space, self.embed_dim, 2)

    def forward(self, input: HeteroData):
        mean, std, self.val = self.agent(input.clone())
        return mean, std

    def value_function(self):
        return self.val.flatten()

    def act_eval(self, state):
        action_mean, action_std = self.forward(state)
        return action_mean

    def act(self, state):
        action_mean, action_std = self.forward(state)
        cov_mat = torch.diag_embed(action_std)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.value_function()

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_mean, action_var = self.forward(state)

        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.value_function()
        return action_logprobs, state_values, dist_entropy
