from torch import nn
import torch
from ray.rllib.models.torch.misc import normc_initializer
from torch.distributions import MultivariateNormal
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import FastRGCNConv, SAGEConv, GCNConv, EdgeConv
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
        self.conv1 = SAGEConv(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            aggr="mean",
        )
        # self.conv2 = SAGEConv(
        #     in_channels=self.embed_dim,
        #     out_channels=self.embed_dim,
        #     aggr="mean",
        # )
        self.act = nn.ReLU()
        self.final_layer = nn.Linear(2 * self.embed_dim, out_dim)
        self.val_layer = nn.Linear(2 * 6 * self.embed_dim, 1)
        normc_initializer(0.001)(self.final_layer.weight)
        normc_initializer(0.001)(self.val_layer.weight)

    def forward(self, input: HeteroData) -> torch.Tensor:
        for node_type in self.obs_space["node_features"]:
            input[node_type].x = self.node_embeder[node_type](
                input[node_type].x.float()
            )
        input_homogeneous = input.to_homogeneous()
        skip_connection = input_homogeneous.x[input_homogeneous.node_type == 3]
        input_homogeneous.x = self.act(
            self.conv1(
                input_homogeneous.x,
                input_homogeneous.edge_index,
            ),
        )

        # input_homogeneous.x = self.act(
        #     self.conv2(
        #         input_homogeneous.x,
        #         input_homogeneous.edge_index,
        #     ),
        # )
        result = input_homogeneous.x[input_homogeneous.node_type == 3]
        result = torch.cat([result, skip_connection], dim=1)
        value = result.reshape(input.num_graphs, -1)
        value = self.val_layer(value)
        result = self.final_layer(result)
        return result, value


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        nn.Module.__init__(self)
        self.n_dim = action_space["redispatch"].shape[0]  # type: ignore
        self.embed_dim = 16

        self.original_space = obs_space
        self.actor = GraphNet(obs_space, action_space, self.embed_dim, 2)
        # self.special_init(self.actor)

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
        special_init(self.critic)

    def forward(self, input: HeteroData, state, seq_lens):
        # self.val = self.critic(input["gen"].x.view((input.num_graphs, -1))).reshape(-1)
        action, self.val = self.actor(input.clone())  # .reshape(-1, self.n_dim, 2)
        mean = action[:, 0]
        log_std = action[:, 1]
        mean = mean.reshape(input.num_graphs, -1)
        log_std = log_std.reshape(input.num_graphs, -1)
        # action = action.reshape(input.num_graphs, -1)
        # mean, log_std = torch.chunk(action.reshape(input.num_graphs, -1), 2, dim=1)

        std = F.softplus(log_std)

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
