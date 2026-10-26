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


TAPE_SPACING = 0.05  # m, length of each tape box
# Boxes are a bit longer than their spacing so curves never show a gap between them
TAPE_OVERLAP = 1.2
HIDDEN_Z = -1.0  # unused tape boxes are parked under the floor


def resample_loop(points: np.ndarray, spacing: float) -> np.ndarray:
    """Evenly spaced points along a closed polyline, about `spacing` apart."""
    closed = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(round(s[-1] / spacing), 3)
    s_new = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack(
        [np.interp(s_new, s, closed[:, 0]), np.interp(s_new, s, closed[:, 1])], axis=1
    )


def tape_poses(track: Track) -> np.ndarray:
    """(m, 3) x, y, yaw of every tape box, both edges.

    Each edge is resampled on its own, so boxes are evenly spaced even where the
    outside of a curve is much longer than the centerline. Every box is the same
    size, which lets the GPU env move one shared set of boxes onto any track.
    """
    poses = []
    edge = track.normals * track.width / 2
    for side in (track.centerline + edge, track.centerline - edge):
        pts = resample_loop(side, TAPE_SPACING)
        d = np.roll(pts, -1, axis=0) - pts
        mids = pts + d / 2
        poses.append(np.column_stack([mids, np.arctan2(d[:, 1], d[:, 0])]))
    return np.concatenate(poses)


def _add_tape_box(spec: mujoco.MjSpec, name: str, pos, yaw: float) -> None:
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,
        size=[TAPE_SPACING * TAPE_OVERLAP / 2, TAPE_WIDTH / 2, TAPE_HEIGHT],
        euler=[0, 0, yaw],
        material="tape",  # assets/car.py build_scene
    )


def build_track(spec: mujoco.MjSpec, track: Track) -> None:
    """Tape for one fixed track (CPU viewers)."""
    for k, (x, y, yaw) in enumerate(tape_poses(track)):
        _add_tape_box(spec, f"tape_{k}", [x, y, TAPE_HEIGHT], yaw)


def build_tape_pool(spec: mujoco.MjSpec, count: int) -> None:
    """`count` tape boxes parked under the floor. The GPU env moves them onto each
    world's own track (env/kernels.py write_tape)."""
    for k in range(count):
        _add_tape_box(spec, f"tape_{k}", [0, 0, HIDDEN_Z], 0.0)
