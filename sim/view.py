"""Watch a trained policy drive in the MuJoCo viewer.

The car runs in the same GPU env used for training (1 world), with the run's
reward settings, so observations, rewards, and resets match training exactly.
Each step its pose is copied into a CPU MuJoCo scene for display.

Drives one track: by default test track 0, from a set the policy never trained on.
Each episode starts from the same spot eval.py uses, so its crashes replay here.

Run from sim/:
    uv run python view.py                   # latest run in runs/, test track 0
    uv run python view.py --test 3          # test track 3
    uv run python view.py --track 3         # training track 3 (as in show_tracks.py)
    uv run python view.py --random-start    # random start points, like training
    uv run python view.py runs/<run>        # a specific run
    uv run python view.py --stochastic      # with exploration noise, like training
    uv run python view.py --car-view        # see what the car's camera sees
    uv run python view.py --last            # the final actor instead of the best one
    uv run python view.py --checkpoint actor_iter0100.pt   # a specific checkpoint
"""

import argparse
import time

import mujoco
import mujoco.viewer
import torch
import warp as wp

from rcsim.assets.car import build_model
from rcsim.assets.track_gen import generate_track
from rcsim.checkpoints import add_args, resolve
from rcsim.config import load_config
from rcsim.env.car_env import CarEnv
from rcsim.experiment import load_experiment
from rcsim.model.agent import PilotNetActor

wp.config.log_level = wp.LOG_WARNING

parser = argparse.ArgumentParser()
add_args(parser)
parser.add_argument(
    "--stochastic",
    action="store_true",
    help="sample actions like during training, instead of the mean (what the car runs)",
)
parser.add_argument(
    "--car-view",
    action="store_true",
    help="start on the car's own camera (front) instead of the chase camera",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--random-start",
    action="store_true",
    help="start each episode somewhere random, like training, instead of the same "
    "start eval.py uses",
)
which = parser.add_mutually_exclusive_group()
which.add_argument("--test", type=int, help="held-out test track number (default 0)")
which.add_argument("--track", type=int, help="training track number")
args = parser.parse_args()

run, actor_file = resolve(args)
cfg = load_experiment(run / "config.toml")
print(f"viewing {run}")

# Tracks come from the run's [tracks] settings and seed, so training track N is
# the same one the run trained on
if args.track is not None:
    track = generate_track(cfg.seed, args.track, cfg.tracks)
    print(f"training track {args.track}")
else:
    track = generate_track(cfg.seed, args.test or 0, cfg.tracks, test=True)
    print(f"test track {args.test or 0} (never trained on)")

env = CarEnv(
    num_worlds=1,
    seed=args.seed,
    render=True,  # the actor drives from the camera
    tracks=[track],
    reward=cfg.reward,
    episode_steps=cfg.env.episode_steps,
    randomize=cfg.randomize,
    visuals=cfg.visuals,
    # Same start as eval.py, so a crash it reports replays here exactly
    evaluate=not args.random_start,
)
assert env.pixel_shape is not None
actor = PilotNetActor(env.pixel_shape, env.action_dim).cuda()
print(f"loading {actor_file.name}")
actor.load_state_dict(torch.load(actor_file, map_location="cuda"))
actor.eval()

# The env's model has movable tape for the GPU; the viewer gets the track drawn in.
# Same car, joints, and cameras, so the GPU qpos copies straight over
_, model = build_model(load_config(), track)
data = mujoco.MjData(model)

start = env.reset()
obs, pixels = start.obs, start.pixels
laps = 0.0  # this episode
episodes = crashes = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Start on the chase camera that follows the car, or with --car-view the car's
    # own camera (switch either way in the viewer's panel)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = model.camera("front" if args.car_view else "chase").id
    while viewer.is_running():
        tick = time.perf_counter()

        with torch.no_grad():
            assert pixels is not None  # render=True
            dist = actor.get_dist(pixels)
            action = dist.sample() if args.stochastic else dist.mean
        out = env.step(action.clamp(-1.0, 1.0))
        obs, pixels = out.obs, out.pixels

        reward = out.reward.item()
        terms = env.reward_terms_t[0].tolist()
        done = out.done.item() > 0
        crashed = done and out.truncated.item() == 0
        laps += env.laps_t[0].item()

        # Show the GPU world's pose (already reset to a new start if done)
        data.qpos[:] = env.device_data.qpos.numpy()[0]
        mujoco.mj_forward(model, data)

        steer, throttle = action[0].tolist()
        viewer.set_texts(
            (
                None,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                "steer\nthrottle\nspeed\noffset\nlaps\nepisodes\ncrashes\n"
                "reward\n  progress\n  off track\n  center\n  steer rate",
                f"{steer:+.2f}\n{throttle:+.2f}\n{obs[0, 3].item() * env.car.max_speed:.2f} m/s\n"
                f"{obs[0, 0].item():+.2f}\n{laps:.2f}\n"
                f"{episodes}\n{crashes}\n{reward:+.3f}\n"
                + "\n".join(f"{t:+.3f}" for t in terms),
            )
        )
        viewer.sync()

        if done:
            episodes += 1
            crashes += crashed
            print(
                f"episode {episodes}: {'CRASHED' if crashed else 'timeout'} after "
                f"{laps:.2f} laps"
            )
            laps = 0.0

        # Real time: one policy step every control period
        time.sleep(max(0.0, env.car.dt - (time.perf_counter() - tick)))
