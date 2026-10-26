"""Load config/car.toml, the config shared by the sim, training, and the Pi runtime.

Loading is strict: a missing or misspelled key raises instead of silently
falling back to a default.
"""

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "car.toml"


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fov_y_deg: float
    pos: tuple[float, float, float]  # x forward, y left, z up, from rear axle center
    pitch_deg: float  # negative = tilted down

    @property
    def fov_y(self) -> float:
        return math.radians(self.fov_y_deg)

    @property
    def pitch(self) -> float:
        return math.radians(self.pitch_deg)


@dataclass(frozen=True)
class SteeringConfig:
    servo_left: int
    servo_center: int
    servo_right: int
    max_angle_deg: float
    response_time_s: float  # servo time constant (first-order lag)

    @property
    def max_angle(self) -> float:
        return math.radians(self.max_angle_deg)


@dataclass(frozen=True)
class ThrottleConfig:
    max_speed: float  # m/s at full PWM
    min_duty: int  # % PWM that first moves the car
    response_time_s: float  # motor time constant (first-order lag)


@dataclass(frozen=True)
class GeometryConfig:
    wheelbase: float
    track_width: float
    wheel_radius: float
    length: float
    width: float


@dataclass(frozen=True)
class ControlConfig:
    rate_hz: float
    latency_s: float

    @property
    def dt(self) -> float:
        return 1.0 / self.rate_hz


@dataclass(frozen=True)
class CarConfig:
    camera: CameraConfig
    steering: SteeringConfig
    throttle: ThrottleConfig
    geometry: GeometryConfig
    control: ControlConfig


def _check_keys(name: str, cls: type, table: dict) -> None:
    expected = {f.name for f in fields(cls)}
    missing = expected - table.keys()
    unknown = table.keys() - expected
    if missing or unknown:
        raise ValueError(
            f"[{name}] in {CONFIG_PATH.name}: "
            f"missing {sorted(missing)}, unknown {sorted(unknown)}"
        )


def _build(name: str, cls: type, table: dict):
    _check_keys(name, cls, table)
    # TOML arrays load as lists; store them as tuples so the dataclass stays immutable
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in table.items()}
    return cls(**values)


def load_config(path: Path = CONFIG_PATH) -> CarConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    _check_keys("top level", CarConfig, raw)
    sections = {
        f.name: _build(f.name, f.type, raw[f.name])  # pyright: ignore[reportArgumentType]
        for f in fields(CarConfig)
    }
    return CarConfig(**sections)


if __name__ == "__main__":
    print(load_config())
