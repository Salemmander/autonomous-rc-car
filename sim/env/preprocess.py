"""Camera preprocessing. The Pi must do exactly the same, or the policy sees
different images on the car than in training.

160x120 RGB camera frame -> crop the top 30% (mostly sky/background) -> 160x84 RGB.
Frames are kept as uint8 (0..255) to save GPU memory in the rollout buffer; the
actor divides by 255.

Note: the supervised PilotNet (src/training, scripts/export_onnx.py) uses YCbCr.
This RL policy uses plain RGB, so its ONNX export (later) must crop only, with no
color conversion.
"""

import torch

CROP_TOP_FRAC = 0.3


def crop_top(height: int) -> int:
    """Rows removed from the top: int(120 * 0.3) = 36, so 84 rows remain."""
    return int(height * CROP_TOP_FRAC)


def _crop_to_uint8(rgb_hwc: torch.Tensor) -> torch.Tensor:
    """(N, H, W, 3) float RGB in [0, 1] -> (N, 3, H - crop, W) uint8 RGB."""
    x = rgb_hwc[:, crop_top(rgb_hwc.shape[1]) :, :, :] * 255.0
    return x.round().clamp(0, 255).to(torch.uint8).permute(0, 3, 1, 2)


# Compiled into one fused GPU kernel instead of several separate passes over
# all 1024 frames
_crop_to_uint8_fused = torch.compile(_crop_to_uint8)


def preprocess(rgb_hwc: torch.Tensor, out: torch.Tensor) -> None:
    """rgb_hwc: (N, H, W, 3) float in [0, 1], the renderer's own memory layout.
    out: (N, 3, H - crop_top(H), W) uint8 RGB, written in place.
    """
    out.copy_(_crop_to_uint8_fused(rgb_hwc))
