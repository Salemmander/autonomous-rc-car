"""Load the knobs for a training run.

experiments/baseline.toml holds every knob and its value; it's the only place
defaults live. An experiment file is applied on top of it and only lists what it
changes. `--set section.key=value` on the command line goes on top of that.

The dataclasses below have no defaults on purpose: a knob missing from
baseline.toml, an unknown key, or a wrong type is an error, never a silent fallback.
"""

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

BASELINE = Path(__file__).parent / "experiments" / "baseline.toml"


@dataclass
class RewardConfig:
    progress: float
    off_track: float
    center: float
    steer_rate: float


@dataclass
class EnvConfig:
    num_worlds: int
    episode_steps: int


@dataclass
class PPOConfig:
    iterations: int
    rollout_steps: int
    lr: float
    minibatch_size: int
    update_epochs: int
    hidden_dim: int
    gamma: float
    gae_lambda: float
    reward_scale: float
    entropy_weight: float
    clip_epsilon: float
    max_grad_norm: float


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    env: EnvConfig
    reward: RewardConfig
    ppo: PPOConfig


SECTIONS = {"env": EnvConfig, "reward": RewardConfig, "ppo": PPOConfig}
TOP_LEVEL = ("name", "seed")


def _read(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _parse_value(raw: str) -> Any:
    """'0.5' -> 0.5, 'true' -> True, 'abc' -> 'abc' (TOML syntax, bare strings allowed)."""
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def _coerce(old: Any, value: Any, key: str, source: str) -> Any:
    if (
        isinstance(old, float)
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return float(value)  # allow `center = 1` for a float knob
    if type(value) is not type(old):
        raise ValueError(
            f"{source}: {key} should be {type(old).__name__}, got {value!r}"
        )
    return value


def _merge(base: dict, changes: dict, source: str) -> None:
    """Apply changes onto base in place. Only keys that already exist are allowed."""
    for key, value in changes.items():
        if key not in base:
            raise ValueError(
                f"{source}: unknown key '{key}'. Valid: {', '.join(sorted(base))}"
            )
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{source}: [{key}] must be a section")
            for k, v in value.items():
                if k not in base[key]:
                    raise ValueError(
                        f"{source}: unknown key '{key}.{k}'. "
                        f"Valid: {', '.join(sorted(base[key]))}"
                    )
                base[key][k] = _coerce(base[key][k], v, f"{key}.{k}", source)
        else:
            base[key] = _coerce(base[key], value, key, source)


def _build(raw: dict) -> ExperimentConfig:
    """dict -> dataclasses, checking baseline.toml has every knob the code expects."""
    expected = set(TOP_LEVEL) | set(SECTIONS)
    if missing := expected - raw.keys():
        raise ValueError(f"{BASELINE.name} is missing {sorted(missing)}")
    sections = {}
    for name, cls in SECTIONS.items():
        want = {f.name for f in fields(cls)}
        if missing := want - raw[name].keys():
            raise ValueError(f"{BASELINE.name} [{name}] is missing {sorted(missing)}")
        if extra := raw[name].keys() - want:
            raise ValueError(
                f"[{name}] has {sorted(extra)}, which experiment.py doesn't know. "
                f"Add a field to {cls.__name__}."
            )
        sections[name] = cls(**raw[name])
    return ExperimentConfig(name=raw["name"], seed=raw["seed"], **sections)


def load_experiment(
    path: Path | None = None, overrides: list[str] = []
) -> ExperimentConfig:
    """baseline.toml, then the experiment file (if any), then --set overrides."""
    raw = _read(BASELINE)
    if path is not None and path.resolve() != BASELINE.resolve():
        changes = _read(path)
        changes.setdefault("name", path.stem)  # name the run after the file
        _merge(raw, changes, str(path))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set {item}: expected section.key=value")
        dotted, value = item.split("=", 1)
        *section, key = dotted.strip().split(".")
        change = {key: _parse_value(value.strip())}
        _merge(raw, {section[0]: change} if section else change, f"--set {item}")
    return _build(raw)


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return repr(v)


def to_toml(cfg: ExperimentConfig) -> str:
    """The full resolved config, so a run folder says exactly what it used."""
    d = asdict(cfg)
    lines = [f"{k} = {_toml_value(d[k])}" for k in TOP_LEVEL]
    for section in SECTIONS:
        lines += ["", f"[{section}]"]
        lines += [f"{k} = {_toml_value(v)}" for k, v in d[section].items()]
    return "\n".join(lines) + "\n"
