"""Export PilotNet + preprocess to TorchScript for LibTorch C++ inference.

Input:  NCHW RGB float in [0, 255] (camera frame, e.g. 1x3x720x1280)
Output: (N, 2) = [steering, throttle]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.pilotnet import INPUT_SIZE, PilotNet

MODEL_PATH = "models/pilotnet_2026-04-22_16-39-13.pth"
TS_PATH = "models/pilotnet_ts.pt"

NET_HEIGHT = 84
NET_WIDTH = 160
CAM_HEIGHT = 720
CAM_WIDTH = 1280


def rgb_to_ycbcr_255(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.cat((y, cb, cr), dim=1) / 255.0


class PilotNetPreprocess(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=INPUT_SIZE,
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        top = int(INPUT_SIZE[0] * 0.3)
        x = x[:, :, top:, :]
        return rgb_to_ycbcr_255(x)


class PilotNetRuntime(nn.Module):
    def __init__(self, model: PilotNet):
        super().__init__()
        self.preprocess = PilotNetPreprocess()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.model.conv_layers(x)
        x = x.flatten(1)
        x = self.model.fc_layers(x)
        return self.model.output(x)


def main() -> None:
    model = PilotNet(NET_HEIGHT, NET_WIDTH)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    runtime = PilotNetRuntime(model)
    runtime.eval()

    dummy = torch.zeros(1, 3, CAM_HEIGHT, CAM_WIDTH)
    with torch.no_grad():
        traced = torch.jit.trace(runtime, dummy)
        traced.save(TS_PATH)

    print(f"wrote {TS_PATH}")
    print(f"input:  (1, 3, {CAM_HEIGHT}, {CAM_WIDTH}) RGB float [0,255]")
    print("output: (1, 2) = [steering, throttle]")


if __name__ == "__main__":
    main()
