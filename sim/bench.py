"""Measure env throughput: world-steps per second, with and without rendering.

Run from sim/: uv run python bench.py --num-worlds 4096 [--render]
"""

import argparse
import time

import torch

from env.car_env import CarEnv

parser = argparse.ArgumentParser()
parser.add_argument("--num-worlds", type=int, default=4096)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--render", action="store_true")
args = parser.parse_args()

env = CarEnv(num_worlds=args.num_worlds, render=args.render)
env.reset()
actions = torch.rand(args.num_worlds, env.action_dim, device="cuda") * 2 - 1

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
sps = args.num_worlds * args.steps / elapsed
print(
    f"{args.num_worlds} worlds, render={args.render}: "
    f"{per_step_ms:.3f} ms/step, {sps:,.0f} world-steps/s"
)
