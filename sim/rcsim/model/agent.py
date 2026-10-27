import math

import torch
from torch import nn
from torch.distributions import Normal


def layer_init(layer, gain: float = math.sqrt(2)):
    """Orthogonal weights, zero bias (the standard PPO init)."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class PilotNetActor(nn.Module):
    """The policy: camera image in, [steer, throttle] out. The only network that
    goes on the car.

    Same layers as src/training/pilotnet.py (NVIDIA PilotNet), so it exports to
    ONNX and runs in the existing C++ runtime the same way. Differences, for RL:
    - the 2 outputs go through tanh, giving the mean action in [-1, 1]
    - a learned log_std adds exploration noise during training
    - PPO-style init, with a small output layer so it starts near action 0
    """

    def __init__(self, pixel_shape: tuple[int, int, int], action_dim: int = 2):
        super().__init__()
        c, h, w = pixel_shape
        self.conv_layers = nn.Sequential(
            layer_init(nn.Conv2d(c, 24, kernel_size=5, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(24, 36, kernel_size=5, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(36, 48, kernel_size=5, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(48, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
        )
        with torch.no_grad():
            conv_out = self.conv_layers(torch.zeros(1, c, h, w)).flatten(1).shape[1]
        self.fc_layers = nn.Sequential(
            layer_init(nn.Linear(conv_out, 1164)),
            nn.ReLU(),
            layer_init(nn.Linear(1164, 100)),
            nn.ReLU(),
            layer_init(nn.Linear(100, 50)),
            nn.ReLU(),
            layer_init(nn.Linear(50, 10)),
            nn.ReLU(),
        )
        self.output = layer_init(nn.Linear(10, action_dim), gain=0.01)
        # std ~0.6 to start: explores without most samples landing outside [-1, 1]
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: (N, 3, 84, 160) uint8 RGB. Returns the mean action in [-1, 1],
        which is what runs on the car (no sampling)."""
        x = pixels.float().div_(255)  # same scaling as ToTensor() and the ONNX /255
        x = x.contiguous(memory_format=torch.channels_last)  # faster convs
        # bf16 for the CNN: about 2x faster on RTX 30xx; the output goes back to
        # float32 so the log-prob math in PPO stays precise
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = self.conv_layers(x).flatten(1)
            x = self.output(self.fc_layers(x))
        return torch.tanh(x.float())

    def get_dist(self, pixels: torch.Tensor) -> Normal:
        return Normal(self.forward(pixels), self.log_std.exp(), validate_args=False)


class Critic(nn.Module):
    """Value network on sim-only state (asymmetric actor-critic). Training only:
    it never goes on the car, so it can use information the car doesn't have."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.value = layer_init(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(self.body(obs)).squeeze(-1)
