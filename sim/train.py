"""Train a camera-based driving policy with PPO on the GPU car env.

The actor (PilotNet) only sees the car's camera; the critic sees sim-only state
during training (asymmetric actor-critic). Only the actor goes on the car.

Run from sim/:
    uv run python train.py                                  # experiments/baseline.toml
    uv run python train.py experiments/smoother.toml        # an experiment (your own file)
    uv run python train.py --set reward.steer_rate=1.0 ppo.lr=1e-4 name=smoother_slow_lr

Every knob and its baseline value is in experiments/baseline.toml; other
experiment files only list what they change.

Saved in the run folder: actor_iterNNNN.pt every `ppo.checkpoint_every`
iterations, actor_iterNNNN_best.pt at the best reward so far (only the latest best
is kept), and actor.pt at the end. Each run gets a folder in runs/
with its full config, the trained actor, and TensorBoard curves:
    uv run tensorboard --logdir runs
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import warp as wp
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from assets.track_gen import generate_tracks
from env.car_env import CarEnv
from experiment import load_experiment, to_toml
from model.agent import Critic, PilotNetActor
from ppo.graph import Graphed
from ppo.ppo import ppo_update
from ppo.rollout import RolloutBuffer

wp.config.log_level = wp.LOG_WARNING
torch.backends.cuda.matmul.allow_tf32 = True  # faster matmuls on RTX 30xx

RUNS_DIR = Path(__file__).parent / "runs"

parser = argparse.ArgumentParser(description="Train PPO on the GPU car env")
parser.add_argument("experiment", type=Path, nargs="?", help="experiments/*.toml")
parser.add_argument(
    "--set",
    nargs="+",
    default=[],
    metavar="KEY=VALUE",
    help="override knobs, e.g. reward.center=0.5 ppo.lr=3e-4 name=test",
)
parser.add_argument("--log-every", type=int, default=10, help="iterations between logs")
parser.add_argument(
    "--no-graphs",
    action="store_true",
    help="run eagerly instead of through CUDA graphs (slower, easier to debug)",
)
args = parser.parse_args()

cfg = load_experiment(args.experiment, args.set)
ppo = cfg.ppo

run_dir = RUNS_DIR / f"{cfg.name}_{datetime.now():%Y-%m-%d_%H%M%S}"
run_dir.mkdir(parents=True)
(run_dir / "config.toml").write_text(to_toml(cfg))
writer = SummaryWriter(run_dir)
print(f"run: {run_dir}\n{to_toml(cfg)}")

torch.manual_seed(cfg.seed)
print(f"generating {cfg.tracks.count} tracks...")
env = CarEnv(
    num_worlds=cfg.env.num_worlds,
    seed=cfg.seed,
    render=True,  # the actor drives from the camera
    tracks=generate_tracks(cfg.seed, cfg.tracks),
    reward=cfg.reward,
    episode_steps=cfg.env.episode_steps,
    randomize=cfg.randomize,
    visuals=cfg.visuals,
)
assert env.pixel_shape is not None
actor = PilotNetActor(env.pixel_shape, env.action_dim).cuda()
actor = actor.to(memory_format=torch.channels_last)  # faster convs
critic = Critic(env.obs_dim, ppo.hidden_dim).cuda()
# lr is a GPU tensor so the decay below can change it without re-recording the graph
lr = torch.tensor(ppo.lr, device="cuda")
optimizer = Adam(
    list(actor.parameters()) + list(critic.parameters()),
    lr=lr,
    fused=True,  # one kernel for the whole optimizer step
    capturable=True,  # optimizer step can be recorded in a CUDA graph
)
buffer = RolloutBuffer(
    ppo.rollout_steps, cfg.env.num_worlds, env.obs_dim, env.action_dim, env.pixel_shape
)


def rollout():
    stats = buffer.collect_rollout(env, actor, critic, ppo.reward_scale, ppo.gamma)
    with torch.no_grad():
        next_value = critic(env.obs_t)
    buffer.compute_gae(next_value, ppo.gamma, ppo.gae_lambda)
    return stats


def update():
    return ppo_update(
        actor,
        critic,
        optimizer,
        buffer,
        clip_epsilon=ppo.clip_epsilon,
        entropy_weight=ppo.entropy_weight,
        update_epochs=ppo.update_epochs,
        minibatch_size=ppo.minibatch_size,
        max_grad_norm=ppo.max_grad_norm,
    )


# With graphs, the env's kernels are recorded into the rollout graph, on the env's
# stream (the current one), instead of launching the env's own graph
env.use_graph = args.no_graphs
run_rollout = (
    rollout if args.no_graphs else Graphed(rollout, stream=torch.cuda.current_stream())
)
run_update = update if args.no_graphs else Graphed(update)

env.reset()
start = time.perf_counter()
steps_per_iter = ppo.rollout_steps * cfg.env.num_worlds
# Best average reward per step so far, checked at each log. reward/total is what
# PPO optimizes, so it accounts for every term (speed, centering, smoothness)
best_reward = float("-inf")
best_file: Path | None = None

try:
    for it in range(ppo.iterations):
        lr.fill_(ppo.lr * (1 - it / ppo.iterations))  # linear decay to 0
        roll = run_rollout()
        upd = run_update()

        if ppo.checkpoint_every and (it + 1) % ppo.checkpoint_every == 0:
            torch.save(actor.state_dict(), run_dir / f"actor_iter{it + 1:04d}.pt")

        if it % args.log_every == 0 or it == ppo.iterations - 1:
            torch.cuda.synchronize()  # wait for the GPU before reading the clock
            elapsed = time.perf_counter() - start
            env_steps = (it + 1) * steps_per_iter
            std = actor.log_std.exp().tolist()

            # x-axis is env steps, so runs with different batch sizes line up
            scalars = {k: v.item() for k, v in roll.items()}
            scalars |= {f"train/{k}": v.item() for k, v in upd.items()}
            scalars |= {
                "train/std_steer": std[0],
                "train/std_throttle": std[1],
                "train/lr": lr.item(),
                "perf/steps_per_s": env_steps / elapsed,
            }
            new_best = scalars["reward/total"] > best_reward
            if new_best:
                best_reward = scalars["reward/total"]
                if best_file is not None:
                    best_file.unlink()  # keep only the latest best
                best_file = run_dir / f"actor_iter{it + 1:04d}_best.pt"
                torch.save(actor.state_dict(), best_file)
            scalars["train/best_reward"] = best_reward
            for k, v in scalars.items():
                writer.add_scalar(k, v, env_steps)

            print(
                f"iter {it:4d} "
                f"| laps/min {scalars['drive/laps_per_min']:4.1f} "
                f"| crashes/min {scalars['drive/crashes_per_min']:5.2f} "
                f"| speed {scalars['drive/speed']:4.2f} m/s "
                f"| |offset| {scalars['drive/offset_abs']:.2f} "
                f"| steer change {scalars['drive/steer_change']:.3f} "
                f"| reward {scalars['reward/total']:+.3f}{'*' if new_best else ' '} "
                f"| std [{std[0]:.2f} {std[1]:.2f}] "
                f"| {env_steps / elapsed / 1e6:.2f}M steps/s | {elapsed:5.1f}s"
            )
finally:
    # Save even if stopped early with Ctrl+C
    torch.save(actor.state_dict(), run_dir / "actor.pt")
    writer.close()
    print(f"saved {run_dir / 'actor.pt'} (last)")
    if best_file is not None:
        print(f"best: {best_file.name} (reward {best_reward:+.3f})")
