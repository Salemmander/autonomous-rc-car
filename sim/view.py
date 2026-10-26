"""Watch a trained policy drive in the MuJoCo viewer.

The car runs in the same GPU env used for training (1 world), with the run's
reward settings, so observations, rewards, and resets match training exactly.
Each step its pose is copied into a CPU MuJoCo scene for display.

Run from sim/:
    uv run python view.py                   # latest run in runs/
    uv run python view.py runs/<run>        # a specific run
    uv run python view.py --stochastic      # with exploration noise, like training
    uv run python view.py --car-view        # see what the car's camera sees
    uv run python view.py --last            # the final actor instead of the best one
    uv run python view.py --checkpoint actor_iter0100.pt   # a specific checkpoint
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import torch
import warp as wp

from env.car_env import CarEnv
from experiment import load_experiment
from model.agent import PilotNetActor

wp.config.log_level = wp.LOG_WARNING
RUNS_DIR = Path(__file__).parent / "runs"

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path, nargs="?", help="run folder (default: latest)")
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
parser.add_argument(
    "--last",
    action="store_true",
    help="load actor.pt (end of training) instead of the best checkpoint",
)
parser.add_argument("--checkpoint", help="file in the run folder, e.g. actor_iter0100.pt")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

run = args.run or max(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
cfg = load_experiment(run / "config.toml")
print(f"viewing {run}")

env = CarEnv(
    num_worlds=1,
    seed=args.seed,
    render=True,  # the actor drives from the camera
    reward=cfg.reward,
    episode_steps=cfg.env.episode_steps,
)
assert env.pixel_shape is not None
actor = PilotNetActor(env.pixel_shape, env.action_dim).cuda()
# Default: the best checkpoint (actor_iterNNNN_best.pt), else the final actor.pt
best = sorted(run.glob("actor_iter*_best.pt"))
if args.checkpoint:
    actor_file = run / args.checkpoint
elif best and not args.last:
    actor_file = best[-1]
else:
    actor_file = run / "actor.pt"
print(f"loading {actor_file.name}")
actor.load_state_dict(torch.load(actor_file, map_location="cuda"))
actor.eval()

model = env.model
data = mujoco.MjData(model)

start = env.reset()
obs, pixels = start.obs, start.pixels
distance = 0.0  # meters along the track this episode
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
        distance += env.progress_t[0].item()

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
                f"{obs[0, 0].item():+.2f}\n{distance / env.track.length:.2f}\n"
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
                f"{distance / env.track.length:.2f} laps"
            )
            distance = 0.0

        # Real time: one policy step every control period
        time.sleep(max(0.0, env.car.dt - (time.perf_counter() - tick)))
