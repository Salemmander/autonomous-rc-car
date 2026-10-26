"""Score a trained policy on tracks it never trained on.

Drives every test track at once (one world each), with the mean action and no
exploration noise, like on the car. Each gets one episode (episode_steps, 30 s
by default) from a fixed start, so the numbers are repeatable and comparable
between runs. Crashes are listed by test track, to watch with view.py --test N.

Run from sim/:
    uv run python eval.py                    # latest run, best checkpoint, 256 test tracks
    uv run python eval.py runs/<run> --tracks 1024
    uv run python eval.py --train            # its training tracks instead (for comparison)
    uv run python eval.py --randomize        # a random car and look per track, like training
    uv run python eval.py --last             # actor.pt / --checkpoint actor_iter0100.pt
"""

import argparse

import torch
import warp as wp

from assets.track_gen import generate_track
from checkpoints import add_args, resolve
from env.car_env import CarEnv
from experiment import load_experiment
from model.agent import PilotNetActor

wp.config.log_level = wp.LOG_WARNING

parser = argparse.ArgumentParser(description="Score a policy on held-out tracks")
add_args(parser)
parser.add_argument("--tracks", type=int, default=256, help="how many tracks")
parser.add_argument(
    "--train", action="store_true", help="use the run's training tracks instead"
)
parser.add_argument(
    "--randomize",
    action="store_true",
    help="randomize the car and look per track ([randomize] and [visuals] ranges) "
    "instead of the car.toml car and the default look",
)
args = parser.parse_args()

run, actor_file = resolve(args)
cfg = load_experiment(run / "config.toml")
kind = "training" if args.train else "test"
tracks = [
    generate_track(cfg.seed, i, cfg.tracks, test=not args.train)
    for i in range(args.tracks)
]

env = CarEnv(
    num_worlds=len(tracks),
    seed=cfg.seed,
    render=True,
    tracks=tracks,
    reward=cfg.reward,
    episode_steps=cfg.env.episode_steps,
    randomize=cfg.randomize,
    visuals=cfg.visuals,
    evaluate=True,  # world i drives track i from its fixed start
    nominal=not args.randomize,
)
assert env.pixel_shape is not None
actor = PilotNetActor(env.pixel_shape, env.action_dim).cuda()
actor.load_state_dict(torch.load(actor_file, map_location="cuda"))
actor.eval()

n = len(tracks)
alive = torch.ones(
    n, device="cuda"
)  # 0 after the first crash; later episodes don't count
crash_step = torch.full((n,), -1, device="cuda")
laps = torch.zeros(n, device="cuda")
speed = torch.zeros(n, device="cuda")
offset = torch.zeros(n, device="cuda")
steps_alive = torch.zeros(n, device="cuda")

out = env.reset()
pixels = out.pixels
with torch.no_grad():
    for step in range(cfg.env.episode_steps):
        assert pixels is not None
        out = env.step(actor(pixels))  # actor() is the mean action
        pixels = out.pixels
        crashed = out.done * (1 - out.truncated)

        laps += env.laps_t * alive
        # obs of a world that just ended is already its next start, so skip it
        counted = alive * (1 - out.done)
        speed += out.obs[:, 3] * counted
        offset += out.obs[:, 0].abs() * counted
        steps_alive += counted

        crash_step = torch.where((crashed * alive) > 0, step, crash_step)
        alive *= 1 - crashed

seconds = cfg.env.episode_steps * env.car.dt
crash_step = crash_step.cpu()
finished = int((crash_step < 0).sum())
minutes = (steps_alive.sum() * env.car.dt / 60).clamp(min=1e-9)

print(f"\n{run.name} / {actor_file.name}")
car = "randomized car and look" if args.randomize else "car.toml car, default look"
print(f"{n} {kind} tracks, {car}, one {seconds:.0f} s episode each, mean action (no noise)\n")
print(f"  finished without crashing  {finished}/{n} ({100 * finished / n:.1f}%)")
print(f"  laps/min                   {(laps.sum() / minutes).item():.2f}")
print(
    f"  speed                      {(speed.sum() / steps_alive.sum()).item() * env.car.max_speed:.2f} m/s"
)
print(
    f"  |offset|                   {(offset.sum() / steps_alive.sum()).item():.3f}  (1 = on the tape)"
)

crashes = [(int(s), i) for i, s in enumerate(crash_step.tolist()) if s >= 0]
if crashes:
    flag = "--track" if args.train else "--test"
    listed = ", ".join(
        f"{i} ({s * env.car.dt:.1f} s)" for s, i in sorted(crashes, key=lambda c: c[1])
    )
    print(f"\n  crashed on {kind} tracks (time of crash): {listed}")
    first = min(crashes, key=lambda c: c[1])[1]
    print(f"  watch one: uv run python view.py {run} {flag} {first}")
