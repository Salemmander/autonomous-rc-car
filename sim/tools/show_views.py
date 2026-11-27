"""Preview what the policy sees: a grid of car camera images, each with a random
look from [visuals] (floor, tape, camera, lighting, exposure, noise), after preprocessing.

Run from sim/:
    uv run python tools/show_views.py                       # 24 random looks -> out/views.png
    uv run python tools/show_views.py --signs               # place each car approaching a stop sign
    uv run python tools/show_views.py --default             # the default look (eval.py, view.py)
    uv run python tools/show_views.py --set visuals.noise=0.1
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import warp as wp
from PIL import Image

from rcsim.assets.track_gen import generate_track
from rcsim.env.car_env import CarEnv
from rcsim.experiment import load_experiment
from rcsim.paths import OUT_DIR

wp.config.log_level = wp.LOG_WARNING
SCALE = 2  # each 160x120 image shown at 2x


def main():
    parser = argparse.ArgumentParser(description="Preview the policy's camera images")
    parser.add_argument("experiment", nargs="?", type=Path)
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--default", action="store_true", help="the default look")
    parser.add_argument(
        "--signs",
        action="store_true",
        help="place each car 0.55-1.3 m ahead of a stop sign so signs are in frame",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR / "views.png")
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
        signs=exp.signs,
        nominal=args.default,
    )
    env.reset()
    if args.signs:
        # Signs always spawn >= 2.2 m ahead of reset, so move each car to 0.55-1.3 m
        # before its first stop sign to preview how signs look across random lighting
        rng = np.random.default_rng(args.seed)
        qpos = env.device_data.qpos.numpy()
        track_ids = env.worlds.track_id.numpy()
        sign_s = env.worlds.sign_s.numpy()
        num_signs = env.worlds.num_signs.numpy()
        for i in range(args.count):
            if num_signs[i] == 0:
                continue
            tr = tracks[int(track_ids[i])]
            dist = float(rng.uniform(0.55, 1.3))
            s_car = (float(sign_s[i, 0]) - dist) % tr.length
            ci = int(np.argmin(np.abs(tr.s - s_car)))
            p = tr.centerline[ci]
            t = tr.tangents[ci]
            qpos[i, :5] = [p[0], p[1], math.atan2(t[1], t[0]), 0.0, 0.0]
        env.device_data.qpos.assign(qpos)
        env._render()
    else:
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"saved {args.out}")
    if not args.no_open:
        out.show()


if __name__ == "__main__":
    main()
