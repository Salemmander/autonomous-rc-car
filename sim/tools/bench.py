"""Measure env throughput: world-steps per second, with and without rendering.

Run from sim/: uv run python tools/bench.py [--num-worlds 1024] [--render] [--oval]
"""

import argparse
import time

import torch

from rcsim.assets.track_gen import generate_tracks
from rcsim.env.car_env import CarEnv
from rcsim.experiment import load_experiment

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-worlds", type=int, help="default: env.num_worlds from baseline.toml"
)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--render", action="store_true")
parser.add_argument(
    "--oval", action="store_true", help="the oval instead of random tracks"
)
args = parser.parse_args()

exp = load_experiment()
num_worlds = args.num_worlds or exp.env.num_worlds
tracks = None if args.oval else generate_tracks(exp.seed, exp.tracks)
env = CarEnv(
    num_worlds=num_worlds,
    render=args.render,
    tracks=tracks,
    randomize=exp.randomize,
    visuals=exp.visuals,
    signs=exp.signs,
)
env.reset()
actions = torch.rand(num_worlds, env.action_dim, device="cuda") * 2 - 1

# Warm up (kernel compile, first graph launch), then time
for _ in range(10):
    env.step(actions)
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(args.steps):
    env.step(actions)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

per_step_ms = elapsed / args.steps * 1000
sps = num_worlds * args.steps / elapsed
print(
    f"{num_worlds} worlds, render={args.render}: "
    f"{per_step_ms:.3f} ms/step, {sps:,.0f} world-steps/s"
)
