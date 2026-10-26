"""Train a state-based driving policy with PPO on the GPU car env.

Run from sim/: uv run python train.py
"""

import argparse
import time
from pathlib import Path

import torch
import warp as wp
from torch.optim import Adam

from env.car_env import CarEnv
from model.agent import Actor, Critic
from ppo.graph import Graphed
from ppo.ppo import ppo_update
from ppo.rollout import RolloutBuffer

wp.config.log_level = wp.LOG_WARNING
torch.backends.cuda.matmul.allow_tf32 = (
    True  # faster matmuls on RTX 30xx, tiny precision cost
)

MODELS_DIR = Path(__file__).parent / "models"


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on the GPU car env")
    p.add_argument("--num-worlds", type=int, default=4096)
    p.add_argument("--rollout-steps", type=int, default=32)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--minibatch-size", type=int, default=32768)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--entropy-weight", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--no-graphs",
        action="store_true",
        help="run eagerly instead of through CUDA graphs (slower, easier to debug)",
    )
    return p.parse_args()


args = parse_args()
torch.manual_seed(args.seed)

env = CarEnv(num_worlds=args.num_worlds, seed=args.seed)
actor = Actor(env.obs_dim, env.action_dim, args.hidden_dim).cuda()
critic = Critic(env.obs_dim, args.hidden_dim).cuda()
# lr is a GPU tensor so the decay below can change it without re-recording the graph
lr = torch.tensor(args.lr, device="cuda")
optimizer = Adam(
    list(actor.parameters()) + list(critic.parameters()),
    lr=lr,
    fused=True,  # one kernel for the whole optimizer step
    capturable=True,  # optimizer step can be recorded in a CUDA graph
)
buffer = RolloutBuffer(args.rollout_steps, args.num_worlds, env.obs_dim, env.action_dim)


def rollout():
    # env.obs_t is the env's fixed output tensor: every step writes the new obs there
    _, stats = buffer.collect_rollout(
        env, actor, critic, env.obs_t, args.reward_scale, args.gamma
    )
    with torch.no_grad():
        next_value = critic(env.obs_t)
    buffer.compute_gae(next_value, args.gamma, args.gae_lambda)
    return stats


def update():
    return ppo_update(
        actor,
        critic,
        optimizer,
        buffer,
        entropy_weight=args.entropy_weight,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
    )


# Record on the env's (Warp's) stream so the env kernels are part of the graph
env.use_graph = args.no_graphs  # with graphs, its kernels go into the rollout graph
run_rollout = (
    rollout if args.no_graphs else Graphed(rollout, stream=torch.cuda.current_stream())
)
run_update = update if args.no_graphs else Graphed(update)

env.reset()
start = time.perf_counter()

for it in range(args.iterations):
    lr.fill_(args.lr * (1 - it / args.iterations))  # linear decay to 0 over the run
    roll = run_rollout()
    upd = run_update()

    if it % args.log_every == 0 or it == args.iterations - 1:
        torch.cuda.synchronize()  # wait for the GPU before reading the clock
        elapsed = time.perf_counter() - start
        sps = (it + 1) * args.rollout_steps * args.num_worlds / elapsed
        std = actor.log_std.exp().tolist()
        print(
            f"iter {it:4d} | len {roll['length'].item():5.1f} "
            f"| off-track {roll['off_track_frac'].item():5.1%} "
            f"| laps {roll['laps'].item():4.2f} | speed {roll['speed'].item():4.2f} m/s "
            f"| std [{std[0]:.2f} {std[1]:.2f}] | critic {upd['critic_loss'].item():.4f} "
            f"| {sps / 1e6:.2f}M steps/s | {elapsed:5.1f}s"
        )

MODELS_DIR.mkdir(exist_ok=True)
torch.save(actor.state_dict(), MODELS_DIR / "actor.pt")
print(f"saved {MODELS_DIR / 'actor.pt'}")
print(f"peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
