"""Where things are, so scripts work from any directory."""

from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = SIM_DIR / "runs"  # training runs: checkpoints, logs, config
OUT_DIR = SIM_DIR / "out"  # pictures from tools/ (gitignored)
BASELINE = SIM_DIR / "experiments" / "baseline.toml"
CAR_CONFIG = SIM_DIR.parent / "config" / "car.toml"  # shared with the car (C++)
