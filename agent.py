import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BranchingQNetwork
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Agent(nn.Module):

    def __init__(self, obs, ac, config):

        super().__init__()
        self.q = BranchingQNetwork(obs, ac, config.bins)
        self.target = BranchingQNetwork(obs, ac, config.bins)
        self.target.load_state_dict(self.q.state_dict())
        self.target_net_update_freq = config.target_net_update_freq
        self.update_counter = 0

    def get_action(self, state):
        with torch.no_grad():
            out = self.q(state).squeeze(0)
            action = torch.argmax(out, dim=1)
        return action.numpy()

    def update_policy(self, adam, memory, config):

        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.sample(config.batch_size)

        states = torch.tensor(b_states).float()
        actions = torch.tensor(b_actions).long().reshape(states.shape[0], -1, 1)
        rewards = torch.tensor(b_rewards).float().reshape(-1, 1)
        next_states = torch.tensor(b_next_states).float()
        masks = torch.tensor(b_masks).float().reshape(-1, 1)

        current_q_values = self.q(states).gather(2, actions).squeeze(-1)

        with torch.no_grad():
            argmax = torch.argmax(self.q(next_states), dim=2)
            max_next_q_values = self.target(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)
            max_next_q_values = max_next_q_values.mean(1, keepdim=True)

        expected_q_values = rewards + max_next_q_values * 0.99 * masks  # Bellmann
        loss = F.mse_loss(expected_q_values, current_q_values)

        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters():
            p.grad.data.clamp_(-1., 1.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0:
            self.update_counter = 0
            self.target.load_state_dict(self.q.state_dict())
