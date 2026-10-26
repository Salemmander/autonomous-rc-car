# pyright: reportIndexIssue=false, reportArgumentType=false, reportGeneralTypeIssues=false
# Warp's array indexing and @wp.struct types (used as annotations below) aren't
# understood by pyright, so those checks are off for this file only.
"""Warp kernels for the car env: one thread per world.

Same math as assets/track.py and drive.py (the numpy reference), done for every
world at once on the GPU. Physics is the kinematic bicycle model written straight
into qpos; MuJoCo is only used to render.
"""

import warp as wp

MAX_STEPS = 600  # 30 s at 20 Hz, about 4 laps of the oval at full speed
SUBSTEPS = wp.constant(4)  # integrate the bicycle model in smaller steps for accuracy
SEARCH_WINDOW = wp.constant(10)  # +-points to search around last step's nearest point
LOOKAHEAD = wp.constant(3)  # centerline points given to the policy as observations
LOOKAHEAD_STRIDE = wp.constant(10)  # points between them (about 0.5 m on the oval)
OFF_TRACK_PENALTY = wp.constant(5.0)  # reward when leaving the track; tune this
OBS_DIM = 5 + 2 * LOOKAHEAD  # wp.constant returns the plain int, so this works

# Reset noise so the policy sees more than the perfect starting pose
START_OFFSET_FRAC = wp.constant(0.3)  # of the half width
START_YAW_NOISE = wp.constant(0.2)  # rad

# qpos layout, from the joint order in build_car
X, Y, YAW, STEER_L, STEER_R = 0, 1, 2, 3, 4


@wp.struct
class TrackData:
    centerline: wp.array1d[wp.vec2]
    tangents: wp.array1d[wp.vec2]
    normals: wp.array1d[wp.vec2]
    s: wp.array1d[float]  # distance along the track to each point
    n: int
    length: float
    half_width: float


@wp.struct
class CarParams:
    wheelbase: float
    max_steer: float  # rad
    max_speed: float  # m/s
    dt: float  # seconds per policy step


@wp.func
def nearest_local(track: TrackData, p: wp.vec2, hint: int) -> int:
    """Nearest centerline point, only checking a window around last step's.

    Fast, and can't snap to another part of the track that happens to pass close
    by. The car moves about 1.5 points per step, well inside the window.
    """
    best = hint
    best_d = wp.length_sq(track.centerline[hint] - p)
    for k in range(-SEARCH_WINDOW, SEARCH_WINDOW + 1):
        j = (hint + k + track.n) % track.n
        d = wp.length_sq(track.centerline[j] - p)
        if d < best_d:
            best_d = d
            best = j
    return best


@wp.func
def wrap_s(track: TrackData, s: float) -> float:
    return s - track.length * wp.floor(s / track.length)


@wp.func
def track_s(track: TrackData, p: wp.vec2, j: int) -> float:
    """Distance along the track, projected onto the tangent (see Track.project)."""
    return wrap_s(
        track, track.s[j] + wp.dot(p - track.centerline[j], track.tangents[j])
    )


@wp.func
def write_obs(
    i: int,
    qpos: wp.array2d[float],
    track: TrackData,
    car: CarParams,
    idx: wp.array1d[int],
    speed: wp.array1d[float],
    steer: wp.array1d[float],
    obs: wp.array2d[float],
):
    p = wp.vec2(qpos[i, X], qpos[i, Y])
    yaw = qpos[i, YAW]
    j = idx[i]
    t = track.tangents[j]

    # Where the car is on the road, scaled so the tape is at +-1
    obs[i, 0] = wp.dot(p - track.centerline[j], track.normals[j]) / track.half_width
    # Heading relative to the road, as cos/sin so it doesn't jump at +-pi
    heading_err = yaw - wp.atan2(t[1], t[0])
    obs[i, 1] = wp.cos(heading_err)
    obs[i, 2] = wp.sin(heading_err)
    obs[i, 3] = speed[i] / car.max_speed
    obs[i, 4] = steer[i] / car.max_steer

    # Upcoming centerline points in the car's frame (x forward, y left), so the
    # policy can see a curve coming before it's in it
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    for k in range(LOOKAHEAD):
        q = track.centerline[(j + (k + 1) * LOOKAHEAD_STRIDE) % track.n] - p
        obs[i, 5 + 2 * k] = c * q[0] + s * q[1]
        obs[i, 6 + 2 * k] = -s * q[0] + c * q[1]


@wp.func
def reset_world(
    i: int,
    qpos: wp.array2d[float],
    track: TrackData,
    idx: wp.array1d[int],
    s_prev: wp.array1d[float],
    speed: wp.array1d[float],
    steer: wp.array1d[float],
    steps: wp.array1d[int],
    resets: wp.array1d[int],
    seed: int,
    num_worlds: int,
):
    resets[i] += 1
    rng = wp.rand_init(seed, resets[i] * num_worlds + i)

    # Random point on the track, a bit off center, roughly facing along it
    j = wp.randi(rng, 0, track.n)
    t = track.tangents[j]
    offset = wp.randf(rng, -START_OFFSET_FRAC, START_OFFSET_FRAC) * track.half_width
    p = track.centerline[j] + track.normals[j] * offset

    qpos[i, X] = p[0]
    qpos[i, Y] = p[1]
    qpos[i, YAW] = wp.atan2(t[1], t[0]) + wp.randf(
        rng, -START_YAW_NOISE, START_YAW_NOISE
    )
    qpos[i, STEER_L] = 0.0
    qpos[i, STEER_R] = 0.0

    idx[i] = j
    s_prev[i] = track_s(track, p, j)
    speed[i] = 0.0
    steer[i] = 0.0
    steps[i] = 0


@wp.kernel
def step(
    actions: wp.array2d[float],
    qpos: wp.array2d[float],
    track: TrackData,
    car: CarParams,
    idx: wp.array1d[int],
    s_prev: wp.array1d[float],
    speed: wp.array1d[float],
    steer: wp.array1d[float],
    obs: wp.array2d[float],
    reward: wp.array1d[float],
    done: wp.array1d[float],
    truncated: wp.array1d[float],
    final_obs: wp.array2d[float],
    steps: wp.array1d[int],
    resets: wp.array1d[int],
    seed: int,
    num_worlds: int,
):
    i = wp.tid()

    # Actions in [-1, 1]. Throttle maps to [0, max_speed]: forward only, so the
    # policy can't learn to wiggle back and forth
    delta = wp.clamp(actions[i, 0], -1.0, 1.0) * car.max_steer
    v = (wp.clamp(actions[i, 1], -1.0, 1.0) + 1.0) * 0.5 * car.max_speed

    # Kinematic bicycle model (same as drive.py), in substeps
    x = qpos[i, X]
    y = qpos[i, Y]
    yaw = qpos[i, YAW]
    h = car.dt / float(SUBSTEPS)
    yaw_rate = v * wp.tan(delta) / car.wheelbase
    for _ in range(SUBSTEPS):
        x += v * wp.cos(yaw) * h
        y += v * wp.sin(yaw) * h
        yaw += yaw_rate * h

    qpos[i, X] = x
    qpos[i, Y] = y
    qpos[i, YAW] = yaw
    qpos[i, STEER_L] = delta
    qpos[i, STEER_R] = delta
    speed[i] = v
    steer[i] = delta

    # Where are we on the track now?
    p = wp.vec2(x, y)
    j = nearest_local(track, p, idx[i])
    idx[i] = j
    offset = wp.dot(p - track.centerline[j], track.normals[j])
    s = track_s(track, p, j)

    # Progress, handling the jump at the start line (see Track.progress)
    ds = s - s_prev[i]
    if ds > track.length * 0.5:
        ds -= track.length
    elif ds < -track.length * 0.5:
        ds += track.length
    s_prev[i] = s

    off_track = wp.abs(offset) > track.half_width
    steps[i] += 1
    timeout = steps[i] >= MAX_STEPS

    # Scaled so full speed along the track earns about 1 per step
    reward[i] = ds / (car.max_speed * car.dt)
    if off_track:
        reward[i] = -OFF_TRACK_PENALTY

    is_done = off_track or timeout
    done[i] = 1.0 if is_done else 0.0
    # Timeout isn't failure: PPO bootstraps the value from final_obs instead of 0
    is_truncated = timeout and not off_track
    truncated[i] = 1.0 if is_truncated else 0.0

    if is_truncated:
        write_obs(i, qpos, track, car, idx, speed, steer, final_obs)
    if is_done:
        reset_world(
            i, qpos, track, idx, s_prev, speed, steer, steps, resets, seed, num_worlds
        )

    write_obs(i, qpos, track, car, idx, speed, steer, obs)


@wp.kernel
def reset_all(
    qpos: wp.array2d[float],
    track: TrackData,
    car: CarParams,
    idx: wp.array1d[int],
    s_prev: wp.array1d[float],
    speed: wp.array1d[float],
    steer: wp.array1d[float],
    obs: wp.array2d[float],
    reward: wp.array1d[float],
    done: wp.array1d[float],
    truncated: wp.array1d[float],
    steps: wp.array1d[int],
    resets: wp.array1d[int],
    seed: int,
    num_worlds: int,
):
    i = wp.tid()
    reset_world(
        i, qpos, track, idx, s_prev, speed, steer, steps, resets, seed, num_worlds
    )
    # Start each world at a random point in its time limit. Otherwise all worlds
    # time out on the same step forever, and training sees bursts of resets
    steps[i] = wp.randi(wp.rand_init(seed + 1, i), 0, MAX_STEPS)
    write_obs(i, qpos, track, car, idx, speed, steer, obs)
    reward[i] = 0.0
    done[i] = 0.0
    truncated[i] = 0.0
