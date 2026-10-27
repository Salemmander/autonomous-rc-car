"""Preview what the policy sees: a grid of car camera images, each with a random
look from [visuals] (floor, tape, camera, lighting, exposure, noise), after preprocessing.

Run from sim/:
    uv run python show_views.py                       # 24 random looks -> views.png
    uv run python show_views.py --default             # the default look (eval.py, view.py)
    uv run python show_views.py --set visuals.noise=0.1
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import warp as wp
from PIL import Image

from assets.track_gen import generate_track
from env.car_env import CarEnv
from experiment import load_experiment

wp.config.log_level = wp.LOG_WARNING
SCALE = 2  # each 160x84 image shown at 2x


def main():
    parser = argparse.ArgumentParser(description="Preview the policy's camera images")
    parser.add_argument("experiment", nargs="?", type=Path)
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--default", action="store_true", help="the default look")
    parser.add_argument("--out", type=Path, default=Path("views.png"))
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    exp = load_experiment(args.experiment, args.set)
    tracks = [generate_track(exp.seed, i, exp.tracks) for i in range(args.count)]
    env = CarEnv(
        num_worlds=args.count,
        seed=args.seed,
        render=True,
        tracks=tracks,
        randomize=exp.randomize,
        visuals=exp.visuals,
        nominal=args.default,
    )
    env.reset()
    # Drive a few steps straight-ish so the car is moving and not all at the start
    for _ in range(5):
        env.step(torch.tensor([[0.0, 0.0]], device="cuda").expand(args.count, 2))
    assert env.pixels_t is not None
    imgs = env.pixels_t.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3) uint8

    cols = 6
    rows = -(-len(imgs) // cols)
    h, w = imgs.shape[1:3]
    sheet = np.zeros((rows * (h + 4), cols * (w + 4), 3), np.uint8)
    for k, img in enumerate(imgs):
        r, c = divmod(k, cols)
        sheet[r * (h + 4) : r * (h + 4) + h, c * (w + 4) : c * (w + 4) + w] = img
    out = Image.fromarray(sheet).resize(
        (sheet.shape[1] * SCALE, sheet.shape[0] * SCALE), Image.Resampling.NEAREST
    )
    out.save(args.out)
    print(f"saved {args.out}")
    if not args.no_open:
        out.show()


if __name__ == "__main__":
    main()
