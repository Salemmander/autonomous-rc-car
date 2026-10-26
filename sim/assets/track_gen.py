"""Random closed tracks.

1. Make a random polygon: corners around a circle, each at a random distance from
   the middle. A corner pulled in makes the track bend the other way (S-bends); two
   close corners with a big direction change make a hairpin; far-apart corners
   leave long straights.
2. Round off every corner with a circular arc of a random radius in turn_radius.
   The track is straights joined by arcs, so no curve is ever tighter than
   turn_radius[0]. A corner too sharp for that radius to fit is thrown away.
3. Resample it to evenly spaced points, the same format as oval_centerline().
4. Throw it away and try again if two parts of the track are too close together
   (or cross), it's the wrong length, or it's off the floor.

Track k for a given seed is always the same track, however many are generated.
"""

from dataclasses import dataclass

import numpy as np

from assets.track import Track, resample_loop

SPACING = 0.05  # m between centerline points, same as the oval
MAX_EXTENT = 4.5  # m from the middle; the floor is 10 x 10 m (assets/car.py)


@dataclass
class TrackGenConfig:
    """The [tracks] section of an experiment. Ranges are [min, max]."""

    count: int
    corners: list[int]
    radius: list[float]
    turn_radius: list[float]
    width: list[float]
    length: list[float]
    gap: float


class _TooSharp(Exception):
    pass


def _round_corners(
    poly: np.ndarray, rng: np.random.Generator, turn_radius: list[float]
) -> np.ndarray:
    """Closed polygon (k, 2) -> closed path (m, 2): each corner replaced by an arc,
    sampled every ~1 cm. Raises _TooSharp if an arc of at least turn_radius[0]
    doesn't fit."""
    prev, nxt = np.roll(poly, 1, axis=0), np.roll(poly, -1, axis=0)
    d_in, d_out = poly - prev, nxt - poly
    len_in, len_out = np.linalg.norm(d_in, axis=1), np.linalg.norm(d_out, axis=1)
    d_in, d_out = d_in / len_in[:, None], d_out / len_out[:, None]
    # Signed direction change at each corner, + = left
    cross = d_in[:, 0] * d_out[:, 1] - d_in[:, 1] * d_out[:, 0]
    turn = np.arctan2(cross, np.sum(d_in * d_out, axis=1))

    # The arc starts and ends `cut` before/after the corner. Each corner may use up
    # to half of each edge next to it, which caps how big its radius can be.
    tan_half = np.tan(np.abs(turn) / 2)
    with np.errstate(divide="ignore"):
        r_max = np.minimum(0.5 * np.minimum(len_in, len_out) / tan_half, turn_radius[1])
    if (r_max < turn_radius[0]).any():
        raise _TooSharp
    # Log-uniform: tight and wide curves equally likely
    lo = np.log(turn_radius[0])
    radius = np.exp(rng.uniform(lo, np.log(r_max)))
    cut = radius * tan_half

    path = []
    for i in range(len(poly)):
        start = poly[i] - d_in[i] * cut[i]
        left = np.array([-d_in[i, 1], d_in[i, 0]])
        center = start + np.sign(turn[i]) * radius[i] * left
        phi0 = np.arctan2(*(start - center)[::-1])
        steps = max(int(radius[i] * abs(turn[i]) / 0.01), 1)
        phi = phi0 + turn[i] * np.linspace(0, 1, steps + 1)
        path.append(center + radius[i] * np.stack([np.cos(phi), np.sin(phi)], axis=1))
    return np.concatenate(path)  # straights are the gaps between arcs


def turn_radii(track: Track) -> np.ndarray:
    """(n,) curve radius at each point (inf on a straight), from how much the
    direction turns between neighboring segments."""
    prev = np.roll(track.tangents, 1, axis=0)
    cross = prev[:, 0] * track.tangents[:, 1] - prev[:, 1] * track.tangents[:, 0]
    dot = np.sum(prev * track.tangents, axis=1)
    turn = np.abs(np.arctan2(cross, dot))  # radians per segment
    seg_len = np.diff(np.append(track.s, track.length))
    with np.errstate(divide="ignore"):
        return seg_len / turn


def min_clearance(track: Track, skip_diameter: float) -> float:
    """Smallest distance between centerline points on different parts of the track.

    Points close along the track are always close in space, so pairs less than
    half a circle of `skip_diameter` apart along the track are skipped;
    turn_radius already covers those.
    """
    d = np.linalg.norm(track.centerline[:, None] - track.centerline[None], axis=2)
    arc = np.abs(track.s[:, None] - track.s[None])
    arc = np.minimum(arc, track.length - arc)
    d[arc < np.pi * skip_diameter / 2] = np.inf
    return float(d.min())


def _check(track: Track, cfg: TrackGenConfig) -> str | None:
    """Why the track is undrivable, or None if it's fine."""
    if not cfg.length[0] <= track.length <= cfg.length[1]:
        return "length"
    if np.abs(track.centerline).max() + track.width / 2 > MAX_EXTENT:
        return "off floor"
    # Arcs are exact; the 0.95 allows for resampling error
    if turn_radii(track).min() < 0.95 * cfg.turn_radius[0]:
        return "too tight"
    # Centerlines this close means the tapes are less than `gap` apart (or cross)
    need = track.width + cfg.gap
    if min_clearance(track, need) < need:
        return "too close"
    return None


def _propose(rng: np.random.Generator, cfg: TrackGenConfig) -> Track | str:
    """A random track, or why it failed before becoming one."""
    k = int(rng.integers(cfg.corners[0], cfg.corners[1] + 1))
    # Evenly spread angles, each nudged by up to 40% of the gap to its neighbors,
    # so the polygon never crosses itself
    angles = (np.arange(k) + rng.uniform(-0.4, 0.4, k)) * 2 * np.pi / k
    radii = rng.uniform(cfg.radius[0], cfg.radius[1], k)
    poly = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    try:
        path = _round_corners(poly, rng, cfg.turn_radius)
    except _TooSharp:
        return "too sharp"
    centerline = resample_loop(path, SPACING)
    if rng.random() < 0.5:
        centerline = centerline[::-1].copy()  # clockwise half the time
    width = float(rng.uniform(cfg.width[0], cfg.width[1]))
    return Track.from_centerline(centerline, width)


def _try(rng: np.random.Generator, cfg: TrackGenConfig) -> Track | str:
    track = _propose(rng, cfg)
    if isinstance(track, str):
        return track
    return _check(track, cfg) or track


def generate_track(
    seed: int, index: int, cfg: TrackGenConfig, test: bool = False, max_tries: int = 10_000
) -> Track:
    """Track `index` of the set for `seed`: proposes random tracks until one passes.

    test=True: a separate set the policy never trains on, to check that it learned
    to follow tape rather than memorizing the training tracks.
    """
    rng = np.random.default_rng([seed, index, 1] if test else [seed, index])
    for _ in range(max_tries):
        track = _try(rng, cfg)
        if isinstance(track, Track):
            return track
    raise RuntimeError(
        f"no valid track in {max_tries} tries; loosen the [tracks] settings"
    )


def generate_tracks(seed: int, cfg: TrackGenConfig) -> list[Track]:
    """The training tracks: 0 to cfg.count - 1 (~6 ms each)."""
    return [generate_track(seed, i, cfg) for i in range(cfg.count)]


def acceptance(cfg: TrackGenConfig, tries: int = 1000, seed: int = 0) -> dict:
    """How often proposals pass, and why the rest fail. For tuning [tracks]."""
    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}
    for _ in range(tries):
        result = _try(rng, cfg)
        reason = "ok" if isinstance(result, Track) else result
        counts[reason] = counts.get(reason, 0) + 1
    return counts
