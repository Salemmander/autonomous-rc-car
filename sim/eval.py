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

from rcsim.assets.track_gen import generate_track
from rcsim.checkpoints import add_args, resolve
from rcsim.env.car_env import CarEnv
from rcsim.env.kernels import CMD_STEER
from rcsim.experiment import load_experiment
from rcsim.model.agent import PilotNetActor, RecurrentActor

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
    signs=cfg.signs,
    evaluate=True,  # world i drives track i from its fixed start
    nominal=not args.randomize,
)
assert env.pixel_shape is not None
state = torch.load(actor_file, map_location="cuda")
if "gru.weight_ih" in state:
    hidden_dim = state["gru.bias_hh"].shape[0] // 3
    actor = RecurrentActor(env.pixel_shape, env.action_dim, hidden_dim).cuda()
else:
    actor = PilotNetActor(env.pixel_shape, env.action_dim).cuda()
actor.load_state_dict(state)
actor.eval()

n = len(tracks)
alive = torch.ones(
    n, device="cuda"
)  # 0 after the first failure (off-track or ran sign); later episodes don't count
fail_step = torch.full((n,), -1, device="cuda")
fail_ran_sign = torch.zeros(n, dtype=torch.bool, device="cuda")
laps = torch.zeros(n, device="cuda")
stops = torch.zeros(n, device="cuda")
ran_signs = torch.zeros(n, device="cuda")
speed = torch.zeros(n, device="cuda")
offset = torch.zeros(n, device="cuda")
steps_alive = torch.zeros(n, device="cuda")

out = env.reset()
obs, pixels = out.obs, out.pixels
h = (
    torch.zeros(n, actor.hidden_dim, device="cuda")
    if isinstance(actor, RecurrentActor)
    else None
)
with torch.no_grad():
    for step in range(cfg.env.episode_steps):
        assert pixels is not None
        if isinstance(actor, RecurrentActor):
            assert h is not None
            prev_cmd = obs[:, CMD_STEER : CMD_STEER + 2]
            action, h = actor.step(pixels, prev_cmd, h)
        else:
            action = actor(pixels)  # actor() is the mean action
        out = env.step(action)
        obs, pixels = out.obs, out.pixels
        if h is not None:
            h = h * (1 - out.done).unsqueeze(-1)
        failed = out.done * (1 - out.truncated)

        laps += env.laps_t * alive
        stops += env.signs_stopped_t * alive
        ran_signs += env.signs_ran_t * alive
        # obs of a world that just ended is already its next start, so skip it
        counted = alive * (1 - out.done)
        speed += out.obs[:, 3] * counted
        offset += out.obs[:, 0].abs() * counted
        steps_alive += counted

        just_failed = (failed * alive) > 0
        fail_step = torch.where(just_failed, step, fail_step)
        fail_ran_sign = torch.where(just_failed, env.signs_ran_t > 0, fail_ran_sign)
        alive *= 1 - failed

seconds = cfg.env.episode_steps * env.car.dt
fail_step = fail_step.cpu()
fail_ran_sign = fail_ran_sign.cpu()
finished = int((fail_step < 0).sum())
n_ran_fail = int(((fail_step >= 0) & fail_ran_sign).sum())
n_off_track = int(((fail_step >= 0) & ~fail_ran_sign).sum())
minutes = (steps_alive.sum() * env.car.dt / 60).clamp(min=1e-9)

print(f"\n{run.name} / {actor_file.name}")
car = "randomized car and look" if args.randomize else "car.toml car, default look"
print(
    f"{n} {kind} tracks, {car}, one {seconds:.0f} s episode each, mean action (no noise)\n"
)
print(
    f"  finished cleanly           {finished}/{n} ({100 * finished / n:.1f}%)  "
    f"[off-track {n_off_track}, ran sign {n_ran_fail}]"
)
print(f"  laps/min                   {(laps.sum() / minutes).item():.2f}")
print(
    f"  speed                      {(speed.sum() / steps_alive.sum()).item() * env.car.max_speed:.2f} m/s"
)
print(
    f"  |offset|                   {(offset.sum() / steps_alive.sum()).item():.3f}  (1 = on the tape)"
)
total_stops = int(stops.sum().item())
total_ran = int(ran_signs.sum().item())
total_signs = total_stops + total_ran
if total_signs > 0:
    print(
        f"  stop signs obeyed          {total_stops}/{total_signs} "
        f"({100 * total_stops / total_signs:.1f}%)  [ran {total_ran}]"
    )

failures = [
    (int(s), i, bool(r))
    for i, (s, r) in enumerate(
        zip(fail_step.tolist(), fail_ran_sign.tolist(), strict=True)
    )
    if s >= 0
]
if failures:
    flag = "--track" if args.train else "--test"
    listed = ", ".join(
        f"{i} ({s * env.car.dt:.1f} s, {'ran sign' if r else 'off-track'})"
        for s, i, r in sorted(failures, key=lambda c: c[1])
    )
    print(f"\n  failed on {kind} tracks: {listed}")
    first = min(failures, key=lambda c: c[1])[1]
    print(f"  watch one: uv run python view.py {run} {flag} {first}")
