"""Watch the trained policy drive in the MuJoCo viewer.

The car runs in the same GPU env used for training (1 world), so observations,
rewards, and resets are exactly what the policy was trained on. Each step its
pose is copied into a CPU MuJoCo scene for display.

Run from sim/ after training: uv run python view.py [--stochastic]
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import torch
import warp as wp

from env.car_env import CarEnv
from model.agent import Actor

wp.config.log_level = wp.LOG_WARNING

parser = argparse.ArgumentParser()
parser.add_argument(
    "--actor", type=Path, default=Path(__file__).parent / "models" / "actor.pt"
)
parser.add_argument("--hidden-dim", type=int, default=128)  # must match training
parser.add_argument(
    "--stochastic",
    action="store_true",
    help="sample actions like during training, instead of the mean (what the car runs)",
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

env = CarEnv(num_worlds=1, seed=args.seed)
actor = Actor(env.obs_dim, env.action_dim, args.hidden_dim).cuda()
actor.load_state_dict(torch.load(args.actor, map_location="cuda"))
actor.eval()

model = env.model
data = mujoco.MjData(model)
meters_per_reward = env.car.max_speed * env.car.dt

obs = env.reset().obs
distance = 0.0  # meters along the track this episode
episodes = crashes = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        tick = time.perf_counter()

        with torch.no_grad():
            action = actor.get_dist(obs).sample() if args.stochastic else actor(obs)
        out = env.step(action.clamp(-1.0, 1.0))
        obs = out.obs

        reward = out.reward.item()
        done = out.done.item() > 0
        crashed = done and out.truncated.item() == 0
        if not crashed:
            distance += reward * meters_per_reward

        # Show the GPU world's pose (already reset to a new start if done)
        data.qpos[:] = env.device_data.qpos.numpy()[0]
        mujoco.mj_forward(model, data)

        steer, throttle = action[0].tolist()
        viewer.set_texts(
            (
                None,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                "steer\nthrottle\nspeed\noffset\nlaps\nepisodes\ncrashes",
                f"{steer:+.2f}\n{throttle:+.2f}\n{obs[0, 3].item() * env.car.max_speed:.2f} m/s\n"
                f"{obs[0, 0].item():+.2f}\n{distance / env.track.length:.2f}\n"
                f"{episodes}\n{crashes}",
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
