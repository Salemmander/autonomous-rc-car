"""Export PilotNet + preprocess to ONNX for C++ inference.

Input:  NCHW RGB float in [0, 255] (camera frame, e.g. 1x3x720x1280)
Output: (N, 2) = [steering, throttle]

Preprocess (torch approx of PilotNet.transform):
  resize 120x160 -> crop top 30% -> RGB->YCbCr -> /255
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.pilotnet import INPUT_SIZE, PilotNet

MODEL_PATH = "models/pilotnet_2026-04-22_16-39-13.pth"
ONNX_PATH = "models/pilotnet.onnx"

NET_HEIGHT = 84
NET_WIDTH = 160
CAM_HEIGHT = 720
CAM_WIDTH = 1280


def rgb_to_ycbcr_255(x: torch.Tensor) -> torch.Tensor:
    """PIL-style RGB->YCbCr for NCHW tensors in 0..255, then scale to 0..1."""
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


class PilotNetOnnx(nn.Module):
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

    export_model = PilotNetOnnx(model)
    export_model.eval()

    dummy = torch.zeros(1, 3, CAM_HEIGHT, CAM_WIDTH)

    torch.onnx.export(
        export_model,
        dummy,
        ONNX_PATH,
        input_names=["rgb_nchw_0_255"],
        output_names=["steer_throttle"],
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {ONNX_PATH}")
    print(f"input:  (1, 3, {CAM_HEIGHT}, {CAM_WIDTH}) RGB float [0,255]")
    print("output: (1, 2) = [steering, throttle]")


if __name__ == "__main__":
    main()
