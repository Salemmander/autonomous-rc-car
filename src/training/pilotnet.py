"""
PilotNet model for behavioral cloning.
Based on NVIDIA's "End to End Learning for Self-Driving Cars" (2016).

Input images are cropped (top 30% removed) and resized to 66x200 in the
data loader before being passed to this model. Converted to YUV color space.
"""

import torch.nn as nn


class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(1152, 1164),
            nn.ReLU(),
            nn.Linear(1164, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
        )
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        # Normalize
        x = x / 255.0

        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        steering_angle = self.output(x)
        return steering_angle
