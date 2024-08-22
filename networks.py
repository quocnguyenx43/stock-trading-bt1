import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, n_obs_dim: int, n_action_dim: int, n_hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_obs_dim, n_hidden_dim)
        self.fc2 = nn.Linear(n_hidden_dim, n_hidden_dim)
        self.fc3 = nn.Linear(n_hidden_dim, n_action_dim)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.tanh(self.fc3(x))
    

class Critic(nn.Module):
    def __init__(self, n_obs_dim: int, n_action_dim: int, n_hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_obs_dim + n_action_dim, n_hidden_dim)
        self.fc2 = nn.Linear(n_hidden_dim, n_hidden_dim)
        self.fc3 = nn.Linear(n_hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
