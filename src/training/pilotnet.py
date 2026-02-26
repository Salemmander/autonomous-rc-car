"""
PilotNet model for behavioral cloning.
Based on NVIDIA's "End to End Learning for Self-Driving Cars" (2016).

Input images are cropped (top 30% removed) in the data loader before being
passed to this model. Converted to YCbCr color space.
"""

from src.training.train import Trainer
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Lambda


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
    MAX_EPOCHS = 200
    PATIENCE = 10
    BATCH_SIZE = 64
    LR = 0.001
    VAL_SPLIT = 0.2

    transform = Compose(
        [
            Lambda(
                lambda img: img.crop(
                    (0, int(img.size[1] * 0.3), img.size[0], img.size[1])
                )
            ),
            Lambda(lambda img: img.convert("YCbCr")),
            ToTensor(),
        ]
    )

    orig_height, orig_width = 120, 160

    model = PilotNet(int(orig_height * 0.7), orig_width)
    trainer = Trainer(
        model, "data/", transform, MAX_EPOCHS, PATIENCE, BATCH_SIZE, LR, VAL_SPLIT
    )
    trainer.train()
