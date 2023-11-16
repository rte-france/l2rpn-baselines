import torch
import torch.nn as nn
from agent import ActorCritic
import wandb
from torch_geometric.data import Batch

################################## set device ##################################
print(
    "============================================================================================"
)
# set device to cpu or cuda
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print(
    "============================================================================================"
)


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPO:
    def __init__(
        self,
        observation_space,
        action_space,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        entropy_loss_weight=0.001,
    ):
        self.has_continuous_action_space = has_continuous_action_space

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_loss_weight = entropy_loss_weight

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(observation_space, action_space).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.policy.agent.get_backbone_params(),
                    "lr": max(lr_actor, lr_critic),
                },
                {"params": self.policy.agent.get_actor_params(), "lr": lr_actor},
                {"params": self.policy.agent.get_critic_params(), "lr": lr_critic},
            ]
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.1
        )

        self.policy_old = ActorCritic(
            observation_space,
            action_space,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action_eval(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = state.to(device)
                state.num_graphs = 1
                action = self.policy_old.act_eval(state)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = state.to(device)
                state.num_graphs = 1
                action = self.policy_old.act_eval(state)

            return action.item()

    def select_action(self, state):
        state.num_graphs = 1
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = state.to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state.clone())
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = Batch.from_data_list(self.buffer.states)
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        running_loss = 0
        running_policy_loss = 0
        running_value_loss = 0
        running_entropy_loss = 0
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - self.entropy_loss_weight * dist_entropy
            )
            running_loss += loss.mean().item()
            running_policy_loss += -torch.min(surr1, surr2).mean().item()
            running_value_loss += self.MseLoss(state_values, rewards).mean().item()
            running_entropy_loss += dist_entropy.mean().item()
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        wandb.log(
            {
                "loss": running_loss / self.K_epochs,
                "policy_loss": running_policy_loss / self.K_epochs,
                "value_loss": running_value_loss / self.K_epochs,
                "entropy_loss": running_entropy_loss / self.K_epochs,
            }
        )
        wandb.log(
            {
                "value_loss_scaled": running_value_loss / self.K_epochs * 0.5,
                "entropy_loss_scaled": running_entropy_loss
                / self.K_epochs
                * self.entropy_loss_weight,
            }
        )
        self.scheduler.step()
        wandb.log(
            {
                "optimizer learning rate actor": self.optimizer.param_groups[0]["lr"],
                "optimizer learning rate critic": self.optimizer.param_groups[1]["lr"],
            }
        )
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
