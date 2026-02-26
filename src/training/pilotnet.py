"""
PilotNet model for behavioral cloning.
Based on NVIDIA's "End to End Learning for Self-Driving Cars" (2016).

Input images are cropped (top 30% removed) in the data loader before being
passed to this model. Converted to YCbCr color space.
"""

import torch
import torch.nn as nn
# import torchvision.transforms.


class PilotNet(nn.Module):
    def __init__(self, input_height: int, input_width: int):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
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

        # get conv_layers output size
        dummy = torch.zeros(1, 3, self.input_height, self.input_width)
        conv_out = self.conv_layers(dummy)
        conv_out_size = conv_out.flatten(1).size(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 1164),
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
        # No normalize step here as it happens in the transform with ToTensor()

        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        steering_angle = self.output(x)
        return steering_angle


if __name__ == "__main__":
    # TODO: Wire up training for PilotNet
    # 1. Build transform pipeline (torchvision.transforms.Compose):
    #    - Crop top 30% of image (remove ceiling/sky) via Lambda
    #    - Convert RGB to YCbCr via Lambda
    #    - ToTensor() (scales to [0,1])
    # 2. Create PilotNet() and Trainer with hardcoded hyperparams
    # 3. Call trainer.train()
    # Run with: uv run python -m src.training.pilotnet
    pass
