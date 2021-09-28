import torch 
import torch.nn as nn 


class BranchingQNetwork(nn.Module):

    def __init__(self, observations, action_dimensions, n):

        super().__init__()

        self.action_dimensions = action_dimensions
        self.n = n

        self.model = nn.Sequential(nn.Linear(observations, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(128, n) for i in range(action_dimensions)])

    def forward(self, state):

        out = self.model(state)
        value = self.value_head(out)  # state value
        advantages = torch.stack([advantage(out) for advantage in self.adv_heads], dim=1)
        q_val = value.unsqueeze(2) + advantages - advantages.mean(2, keepdim=True)
        return q_val
