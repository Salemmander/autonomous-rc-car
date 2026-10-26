"""Tracks: a closed centerline, plus tape geoms drawn along its edges.

The centerline is the single source of truth. The tape visuals, the off-track
check, and the progress reward are all computed from it.

The numpy math here is the CPU reference, used by the viewer scripts. The GPU env
(env/kernels.py) does the same math in Warp for every world at once.
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

TRACK_WIDTH = 0.6  # m, about 3 car widths
TAPE_WIDTH = 0.03  # m
# Half-thickness; tape floats just above the floor to avoid z-fighting
TAPE_HEIGHT = 0.001


def oval_centerline(a: float = 2.0, b: float = 1.2, n: int = 200) -> np.ndarray:
    """Ellipse with half-lengths a (x) and b (y), as an (n, 2) array, counterclockwise.

    The tightest curve (at the ends) has radius b**2 / a. Keep it above the car's
    minimum turn radius, wheelbase / tan(max steer angle), about 0.43 m.
    """
    # endpoint=False: 2*pi is the same point as 0, and the loop closes on its own
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([a * np.cos(t), b * np.sin(t)], axis=1)


@dataclass(frozen=True)
class Track:
    """A closed track with everything derived from its centerline precomputed.

    Arrays are indexed by centerline point. Segment i runs from point i to i+1,
    wrapping from the last point back to the first.
    """

    centerline: np.ndarray  # (n, 2) points
    width: float
    tangents: np.ndarray  # (n, 2) unit direction of segment i
    normals: np.ndarray  # (n, 2) unit vector to the left of segment i
    s: np.ndarray  # (n,) distance along the track from point 0 to point i
    length: float  # once around the loop

    @classmethod
    def from_centerline(cls, centerline: np.ndarray, width: float = TRACK_WIDTH):
        d = np.roll(centerline, -1, axis=0) - centerline
        seg_len = np.linalg.norm(d, axis=1)
        t = d / seg_len[:, None]
        n = np.stack([-t[:, 1], t[:, 0]], axis=1)  # tangent rotated 90 degrees left
        s = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
        return cls(centerline, width, t, n, s, float(seg_len.sum()))

    def project(self, xy: np.ndarray) -> tuple[int, float, float]:
        """(nearest index, lateral offset, distance along the track) for a position.

        offset: signed distance from the centerline, positive = left. Off the track
        when abs(offset) > width / 2.
        s: how far around the loop, in [0, length). Projected onto the tangent so
        it moves smoothly between points instead of jumping point to point.

        Brute-force nearest point. The GPU version searches near the last index
        instead, which is faster and can't snap to a different part of the track.
        """
        i = int(np.argmin(np.sum((self.centerline - xy) ** 2, axis=1)))
        rel = xy - self.centerline[i]
        offset = float(rel @ self.normals[i])
        s = (self.s[i] + float(rel @ self.tangents[i])) % self.length
        return i, offset, s

    def progress(self, s_old: float, s_new: float) -> float:
        """Distance moved along the track, handling the jump at the start line.

        Crossing the start line forward takes s from ~length back to ~0. Any move
        bigger than half a lap must be that wrap, not real motion.
        """
        ds = s_new - s_old
        if ds > self.length / 2:
            ds -= self.length
        elif ds < -self.length / 2:
            ds += self.length
        return ds

    def start_pose(self, i: int = 0) -> tuple[float, float, float]:
        """(x, y, yaw) at centerline point i, facing along the track."""
        x, y = self.centerline[i]
        tx, ty = self.tangents[i]
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


def build_track(spec: mujoco.MjSpec, track: Track) -> None:
    edge = track.normals * track.width / 2
    _add_tape(spec, track.centerline + edge, "tape_left")
    _add_tape(spec, track.centerline - edge, "tape_right")
