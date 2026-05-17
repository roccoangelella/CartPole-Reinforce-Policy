"""Obs space is a two arrays matrix, first is for minimum values and second is for the maximum ones.
- 0th is Cart position
- 1st is cart velocity (-inf to +inf)
- 2nd is Pole angle (roughly +-24°)
- 3rd is Pole's angular velocity (-inf to +inf)

Act space instead is either go one unit right or one unit left

Willing to apply a Reinforce Policy we use a simple MLP that, given the state, outputs the best action to minimize loss. We'll define loss later.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size) -> None:
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.MLP(x)

    def act(self, state, device):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)  #we convert state's numpy array to torch tensor, ensuring they're floats and one dimensional.
        probs = self.forward(state)
        m = Categorical(probs)  #we take softmax output and prepare the policy
        action = m.sample()  #we choose the action by sampling from the softmax output distribution
        return action.item(), m.log_prob(action)  #pick the action decided by policy and return it among its ln
