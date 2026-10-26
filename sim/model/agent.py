import math

import torch
from torch import nn
from torch.distributions import Normal


def layer_init(layer: nn.Linear, gain: float = math.sqrt(2)) -> nn.Linear:
    """Orthogonal weights, zero bias (the standard PPO init)."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def mlp(obs_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        layer_init(nn.Linear(obs_dim, hidden_dim)),
        nn.ReLU(),
        layer_init(nn.Linear(hidden_dim, hidden_dim)),
        nn.ReLU(),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.body = mlp(obs_dim, hidden_dim)
        # Small gain: the untrained policy outputs actions near 0
        # (drive straight at half speed) instead of random full lock
        self.mean = layer_init(nn.Linear(hidden_dim, action_dim), gain=0.01)
        # std ~0.6 to start: wide enough to explore, without most samples
        # landing outside [-1, 1] and getting clamped
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean action in [-1, 1]. This is what runs on the car (no sampling)."""
        return torch.tanh(self.mean(self.body(obs)))

    def get_dist(self, obs: torch.Tensor) -> Normal:
        return Normal(self.forward(obs), self.log_std.exp(), validate_args=False)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.body = mlp(obs_dim, hidden_dim)
        self.value = layer_init(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(self.body(obs)).squeeze(-1)
