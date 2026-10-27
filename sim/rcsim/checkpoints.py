"""Find runs and actor checkpoints (shared by view.py and eval.py)."""

import argparse
from pathlib import Path

from rcsim.paths import RUNS_DIR


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "run", type=Path, nargs="?", help="run folder (default: latest)"
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="load actor.pt (end of training) instead of the best checkpoint",
    )
    parser.add_argument(
        "--checkpoint", help="file in the run folder, e.g. actor_iter0100.pt"
    )


def resolve(args: argparse.Namespace) -> tuple[Path, Path]:
    """(run folder, actor file) from the args above. Default: the latest run's
    best checkpoint (actor_iterNNNN_best.pt), else its final actor.pt."""
    run = args.run or max(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
    best = sorted(run.glob("actor_iter*_best.pt"))
    if args.checkpoint:
        return run, run / args.checkpoint
    if best and not args.last:
        return run, best[-1]
    return run, run / "actor.pt"
