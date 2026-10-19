"""Tracks: a closed centerline, plus tape geoms drawn along its edges.

The centerline is the single source of truth. The tape visuals here and (later)
the progress and off-track reward are both computed from it.
"""

import math

import mujoco
import numpy as np

TRACK_WIDTH = 0.6  # m, about 3 car widths
TAPE_WIDTH = 0.03  # m
TAPE_HEIGHT = (
    0.001  # half-thickness; tape floats just above the floor to avoid z-fighting
)


def oval_centerline(a: float = 2.0, b: float = 1.2, n: int = 200) -> np.ndarray:
    """Ellipse with half-lengths a (x) and b (y), as an (n, 2) array, counterclockwise.

    The tightest curve (at the ends) has radius b**2 / a. Keep it above the car's
    minimum turn radius, wheelbase / tan(max steer angle), about 0.43 m.
    """
    # endpoint=False: 2*pi is the same point as 0, and the loop closes on its own
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([a * np.cos(t), b * np.sin(t)], axis=1)


def tangents(centerline: np.ndarray) -> np.ndarray:
    """Unit direction from each point to the next, wrapping from the last to the first."""
    d = np.roll(centerline, -1, axis=0) - centerline
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def start_pose(centerline: np.ndarray) -> tuple[float, float, float]:
    """(x, y, yaw) at the first centerline point, facing along the track."""
    tx, ty = tangents(centerline)[0]
    x, y = centerline[0]
    return float(x), float(y), math.atan2(ty, tx)


def _add_tape(spec: mujoco.MjSpec, points: np.ndarray, name: str) -> None:
    """One thin box per segment between consecutive points of a closed loop."""
    nxt = np.roll(points, -1, axis=0)
    mids = (points + nxt) / 2
    d = nxt - points
    lengths = np.linalg.norm(d, axis=1)
    yaws = np.arctan2(d[:, 1], d[:, 0])

    for i, (mid, length, yaw) in enumerate(zip(mids, lengths, yaws)):
        spec.worldbody.add_geom(
            name=f"{name}_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[mid[0], mid[1], TAPE_HEIGHT],
            size=[length / 2, TAPE_WIDTH / 2, TAPE_HEIGHT],
            euler=[0, 0, yaw],
            rgba=[0.95, 0.95, 0.95, 1],
        )


def build_track(
    spec: mujoco.MjSpec, centerline: np.ndarray, width: float = TRACK_WIDTH
) -> None:
    t = tangents(centerline)
    # Rotate the tangent 90 degrees counterclockwise: points to the left of the road
    normals = np.stack([-t[:, 1], t[:, 0]], axis=1)

    _add_tape(spec, centerline + normals * width / 2, "tape_left")
    _add_tape(spec, centerline - normals * width / 2, "tape_right")
