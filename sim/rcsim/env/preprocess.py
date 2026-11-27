"""Camera preprocessing. The Pi must do exactly the same, or the policy sees
different images on the car than in training.

160x120 RGB camera frame -> full height (CROP_TOP_FRAC = 0.0 so stop signs on
poles stay visible above the horizon). Frames are kept as uint8 (0..255) to save
GPU memory in the rollout buffer; the actor divides by 255.

The env does this in a Warp kernel (rcsim/env/kernels.py postprocess), which also adds
the training-only image randomization. `preprocess` below is the plain version,
what the Pi does; with the default look the kernel's output matches it exactly.

Note: the supervised PilotNet (src/training, scripts/export_onnx.py) uses YCbCr
and a 30% top crop. This RL policy uses full-frame plain RGB.
"""

import torch

CROP_TOP_FRAC = 0.0


def crop_top(height: int) -> int:
    """Rows removed from the top (0 with full-frame stop-sign view)."""
    return int(height * CROP_TOP_FRAC)


def preprocess(rgb_hwc: torch.Tensor, out: torch.Tensor) -> None:
    """rgb_hwc: (N, H, W, 3) float in [0, 1], the renderer's own memory layout.
    out: (N, 3, H - crop_top(H), W) uint8 RGB, written in place.
    """
    x = rgb_hwc[:, crop_top(rgb_hwc.shape[1]) :, :, :] * 255.0
    out.copy_(x.round().clamp(0, 255).to(torch.uint8).permute(0, 3, 1, 2))
