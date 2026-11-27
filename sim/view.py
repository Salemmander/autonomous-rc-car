"""Watch a trained policy (or drive yourself with --manual) in the MuJoCo viewer.

The car runs in the same GPU env used for training (1 world), with the run's
reward settings, so observations, rewards, and resets match training exactly.
Each step its pose is copied into a CPU MuJoCo scene for display.

Drives one track: by default test track 0, from a set the policy never trained on.
Each episode starts from the same spot eval.py uses, so its crashes replay here.

Run from sim/:
    uv run python view.py                   # latest run in runs/, test track 0
    uv run python view.py --manual          # drive yourself with an Xbox controller (RT=go, stick=steer, A=reset)
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
from rcsim.assets.stop_sign import set_model_signs
from rcsim.assets.track_gen import generate_track
from rcsim.checkpoints import add_args, resolve
from rcsim.config import load_config
from rcsim.env.car_env import CarEnv
from rcsim.env.kernels import CMD_STEER
from rcsim.experiment import load_experiment
from rcsim.model.agent import PilotNetActor, RecurrentActor

wp.config.log_level = wp.LOG_WARNING

parser = argparse.ArgumentParser()
add_args(parser)
parser.add_argument(
    "--manual",
    action="store_true",
    help="drive with an Xbox controller instead of a policy (RT=throttle, stick=steer, A=reset)",
)
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

pad = None
actor = None
actor_file = None
if args.manual:
    from tools.controller import XboxController

    cfg = load_experiment()
    pad = XboxController()
    print(f"manual control: {pad.device.name}")
else:
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
    render=not args.manual,  # the actor drives from the camera; manual watches the viewer
    tracks=[track],
    reward=cfg.reward,
    episode_steps=cfg.env.episode_steps,
    randomize=cfg.randomize,
    visuals=cfg.visuals,
    signs=cfg.signs,
    # Same start as eval.py, so a crash it reports replays here exactly
    evaluate=not args.random_start,
)
if actor_file is not None:
    assert env.pixel_shape is not None
    print(f"loading {actor_file.name}")
    state = torch.load(actor_file, map_location="cuda")
    if "gru.weight_ih" in state:
        hidden_dim = state["gru.bias_hh"].shape[0] // 3
        actor = RecurrentActor(env.pixel_shape, env.action_dim, hidden_dim).cuda()
    else:
        actor = PilotNetActor(env.pixel_shape, env.action_dim).cuda()
    actor.load_state_dict(state)
    actor.eval()

# The env's model has movable tape for the GPU; the viewer gets the track drawn in.
# Same car, joints, and cameras, so the GPU qpos copies straight over
_, model = build_model(load_config(), track)
data = mujoco.MjData(model)


def sync_viewer_signs() -> None:
    """Copy world 0's stop-sign poses from the GPU env into the CPU viewer model."""
    n_signs = int(env.worlds.num_signs.numpy()[0])
    poses = env.worlds.sign_pose.numpy()[0]
    set_model_signs(
        model,
        [
            (float(poses[k, 0]), float(poses[k, 1]), float(poses[k, 2]))
            for k in range(n_signs)
        ],
    )


start = env.reset()
sync_viewer_signs()
obs, pixels = start.obs, start.pixels
h = (
    torch.zeros(1, actor.hidden_dim, device="cuda")
    if isinstance(actor, RecurrentActor)
    else None
)
laps = 0.0  # this episode
episodes = crashes = stops = ran_signs = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Start on the chase camera that follows the car, or with --car-view the car's
    # own camera (switch either way in the viewer's panel)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = model.camera("front" if args.car_view else "chase").id
    while viewer.is_running():
        tick = time.perf_counter()

        if pad is not None:
            pad.poll()
            if pad.reset_pressed:
                pad.reset_pressed = False
                start = env.reset()
                sync_viewer_signs()
                obs, pixels = start.obs, start.pixels
                laps = 0.0
                data.qpos[:] = env.device_data.qpos.numpy()[0]
                mujoco.mj_forward(model, data)
                viewer.sync()
                continue
            # Stick right is +1, sim steer left is +1; RT 0..1 maps to throttle -1..1
            action = torch.tensor(
                [[-pad.steering, pad.forward * 2.0 - 1.0]],
                device="cuda",
                dtype=torch.float32,
            )
        else:
            with torch.no_grad():
                assert pixels is not None and actor is not None
                if isinstance(actor, RecurrentActor):
                    assert h is not None
                    prev_cmd = obs[:, CMD_STEER : CMD_STEER + 2]
                    mean, h = actor.step(pixels, prev_cmd, h)
                    action = actor.dist(mean).sample() if args.stochastic else mean
                else:
                    dist = actor.get_dist(pixels)
                    action = dist.sample() if args.stochastic else dist.mean
        out = env.step(action.clamp(-1.0, 1.0))
        obs, pixels = out.obs, out.pixels
        if h is not None:
            h = h * (1 - out.done).unsqueeze(-1)

        reward = out.reward.item()
        terms = env.reward_terms_t[0].tolist()
        done = out.done.item() > 0
        failed = done and out.truncated.item() == 0
        ran_now = int(env.signs_ran_t[0].item())
        crashed = failed and ran_now == 0
        laps += env.laps_t[0].item()
        stops += int(env.signs_stopped_t[0].item())
        ran_signs += ran_now
        if done:
            sync_viewer_signs()

        # Show the GPU world's pose (already reset to a new start if done)
        data.qpos[:] = env.device_data.qpos.numpy()[0]
        mujoco.mj_forward(model, data)

        steer, throttle = action[0].tolist()
        viewer.set_texts(
            (
                None,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                (
                    "steer\nthrottle\nspeed\noffset\nlaps\nepisodes\ncrashes\nstops / ran\n"
                    "reward\n  progress\n  off track\n  center\n  steer rate\n"
                    "  throttle rate\n  run sign"
                ),
                (
                    f"{steer:+.2f}\n{throttle:+.2f}\n{obs[0, 3].item() * env.car.max_speed:.2f} m/s\n"
                    f"{obs[0, 0].item():+.2f}\n{laps:.2f}\n"
                    f"{episodes}\n{crashes}\n{stops} / {ran_signs}\n{reward:+.3f}\n"
                    + "\n".join(f"{t:+.3f}" for t in terms)
                ),
            )
        )
        viewer.sync()

        if done:
            episodes += 1
            crashes += crashed
            reason = "RAN SIGN" if ran_now else "CRASHED" if crashed else "timeout"
            print(f"episode {episodes}: {reason} after {laps:.2f} laps")
            laps = 0.0

        # Real time: one policy step every control period
        time.sleep(max(0.0, env.car.dt - (time.perf_counter() - tick)))
