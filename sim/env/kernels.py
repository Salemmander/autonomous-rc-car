# pyright: reportIndexIssue=false, reportArgumentType=false, reportGeneralTypeIssues=false, reportAssignmentType=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false
# Warp's array indexing and @wp.struct types (used as annotations below) aren't
# understood by pyright, so those checks are off for this file only.
"""Warp kernels for the car env: one thread per world.

Same math as assets/track.py and drive.py (the numpy reference), done for every
world at once on the GPU. Physics is the kinematic bicycle model written straight
into qpos; MuJoCo is only used to render.

Each world drives its own track from a bank of tracks (track_id[i]), and gets a
new random one every reset.
"""

import warp as wp

from assets.track import HIDDEN_Z, TAPE_HEIGHT, TAPE_OVERLAP, TAPE_SPACING, TAPE_WIDTH

SUBSTEPS = wp.constant(4)  # integrate the bicycle model in smaller steps for accuracy
SEARCH_WINDOW = wp.constant(10)  # +-points to search around last step's nearest point
LOOKAHEAD = wp.constant(3)  # centerline points given to the critic as observations
LOOKAHEAD_STRIDE = wp.constant(10)  # points between them (about 0.5 m)
# Recent commands kept per world, for latency: up to (HISTORY - 1) steps late
HISTORY = wp.constant(4)
# offset, heading cos/sin, speed, steering, lookahead points, the last 2 commands,
# and this episode's car (speed scale, steer scale, trim, latency)
OBS_DIM = 5 + 2 * LOOKAHEAD + 4 + 4  # wp.constant returns the plain int, so this works
CMD_STEER = wp.constant(5 + 2 * LOOKAHEAD)  # obs index of the newest steering command
# Reward terms written out separately for logging: progress, off_track, center, steer_rate
NUM_REWARD_TERMS = 4

# Reset noise so the policy sees more than the perfect starting pose
START_OFFSET_FRAC = wp.constant(0.3)  # of the half width
START_YAW_NOISE = wp.constant(0.2)  # rad

# qpos layout, from the joint order in build_car
X, Y, YAW, STEER_L, STEER_R = 0, 1, 2, 3, 4


@wp.struct
class TrackBank:
    """Every track, packed into (tracks, points) arrays. Tracks have different
    numbers of points; n[t] says how many of a row are real, the rest is padding."""

    centerline: wp.array2d[wp.vec2]
    tangents: wp.array2d[wp.vec2]
    normals: wp.array2d[wp.vec2]
    s: wp.array2d[float]  # distance along the track to each point
    n: wp.array1d[int]
    length: wp.array1d[float]
    half_width: wp.array1d[float]
    # (tracks, tape boxes): x, y, yaw (assets/track.py tape_poses)
    tape: wp.array2d[wp.vec3]
    n_tape: wp.array1d[int]


@wp.struct
class CarParams:
    """The nominal car, from config/car.toml."""

    wheelbase: float
    max_steer: float  # rad
    max_speed: float  # m/s
    dt: float  # seconds per policy step
    latency: float  # s, camera frame to action applied
    steer_time: float  # s, servo time constant
    speed_time: float  # s, motor time constant


@wp.struct
class EnvParams:
    """Runtime knobs from experiment.py. Changing these needs no recompile, unlike
    wp.constant."""

    episode_steps: int
    # 1 = evaluation: world i always drives track i (mod the number of tracks),
    # starting at its first point, centered and straight, and all worlds start the
    # time limit together. Same every time, so results are repeatable
    evaluate: int
    # 1 = the car.toml car and the default look exactly, 0 = randomized every
    # episode ([randomize] and [visuals] ranges)
    nominal: int
    w_progress: float
    w_off_track: float
    w_center: float
    w_steer_rate: float
    # Randomization ranges (RandomizeConfig): [min, max] multipliers of the nominal car
    r_latency: wp.vec2
    r_steer_time: wp.vec2
    r_speed_time: wp.vec2
    r_max_speed: wp.vec2
    r_max_steer: wp.vec2
    r_steer_trim: float  # +- this, in steering units
    # Visual randomization ranges (VisualsConfig)
    v_floor_brightness: wp.vec2
    v_floor_saturation: float
    v_floor_scale: wp.vec2
    v_tape_width: wp.vec2
    v_tape_white: float
    v_camera_height: float
    v_camera_pitch: float  # rad
    v_exposure: wp.vec2
    v_contrast: wp.vec2
    v_white_balance: float
    v_noise: float
    v_sun: wp.vec2
    v_sun_elevation: wp.vec2  # rad
    v_ambient: wp.vec2
    v_lamp: wp.vec2
    v_lamp_height: wp.vec2
    v_light_tint: float
    v_floor_gloss: wp.vec2


@wp.struct
class Worlds:
    """Per-world state: arrays with one entry per world."""

    track_id: wp.array1d[int]  # index into the track bank
    idx: wp.array1d[int]  # nearest centerline point
    s_prev: wp.array1d[float]  # distance along the track last step
    speed: wp.array1d[float]  # m/s, actual (lags the command)
    steer: wp.array1d[float]  # rad, actual wheel angle (lags the command)
    steps: wp.array1d[int]  # in this episode
    resets: wp.array1d[int]  # count, for the random number generator
    # Recent [steer, throttle] commands, newest first, so they can arrive late
    commands: wp.array2d[wp.vec2]
    # This episode's car, randomized around the nominal one at every reset
    latency: wp.array1d[float]
    steer_time: wp.array1d[float]
    speed_time: wp.array1d[float]
    max_speed: wp.array1d[float]
    max_steer: wp.array1d[float]
    steer_trim: wp.array1d[float]
    # This episode's look (randomize_look); the rest of it lives in the model's
    # per-world arrays (floor texture and color, tape color, camera pose)
    tape_width: wp.array1d[float]
    exposure: wp.array1d[float]
    contrast: wp.array1d[float]
    color_gain: wp.array1d[wp.vec3]  # white balance
    noise: wp.array1d[float]


@wp.func
def nearest_local(bank: TrackBank, t: int, p: wp.vec2, hint: int) -> int:
    """Nearest centerline point, only checking a window around last step's.

    Fast, and can't snap to another part of the track that happens to pass close
    by. The car moves about 1.5 points per step, well inside the window.
    """
    n = bank.n[t]
    best = hint
    best_d = wp.length_sq(bank.centerline[t, hint] - p)
    for k in range(-SEARCH_WINDOW, SEARCH_WINDOW + 1):
        j = (hint + k + n) % n
        d = wp.length_sq(bank.centerline[t, j] - p)
        if d < best_d:
            best_d = d
            best = j
    return best


@wp.func
def track_s(bank: TrackBank, t: int, p: wp.vec2, j: int) -> float:
    """Distance along the track, projected onto the tangent (see Track.project)."""
    s = bank.s[t, j] + wp.dot(p - bank.centerline[t, j], bank.tangents[t, j])
    length = bank.length[t]
    return s - length * wp.floor(s / length)  # wrap into [0, length)


@wp.func
def write_obs(
    i: int,
    qpos: wp.array2d[float],
    bank: TrackBank,
    car: CarParams,
    w: Worlds,
    obs: wp.array2d[float],
):
    """The critic's observation. It can use anything the sim knows, since it only
    exists during training: including the delayed commands and this episode's car."""
    p = wp.vec2(qpos[i, X], qpos[i, Y])
    yaw = qpos[i, YAW]
    t = w.track_id[i]
    j = w.idx[i]
    tan = bank.tangents[t, j]

    # Where the car is on the road, scaled so the tape is at +-1
    obs[i, 0] = (
        wp.dot(p - bank.centerline[t, j], bank.normals[t, j]) / bank.half_width[t]
    )
    # Heading relative to the road, as cos/sin so it doesn't jump at +-pi
    heading_err = yaw - wp.atan2(tan[1], tan[0])
    obs[i, 1] = wp.cos(heading_err)
    obs[i, 2] = wp.sin(heading_err)
    obs[i, 3] = w.speed[i] / car.max_speed
    obs[i, 4] = w.steer[i] / car.max_steer

    # Upcoming centerline points in the car's frame (x forward, y left), so the
    # critic can see a curve coming before it's in it
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    for k in range(LOOKAHEAD):
        q = bank.centerline[t, (j + (k + 1) * LOOKAHEAD_STRIDE) % bank.n[t]] - p
        obs[i, 5 + 2 * k] = c * q[0] + s * q[1]
        obs[i, 6 + 2 * k] = -s * q[0] + c * q[1]

    # Commands still on their way to the wheels (latency), and the car this episode
    o = CMD_STEER
    cmd0 = w.commands[i, 0]
    cmd1 = w.commands[i, 1]
    obs[i, o + 0] = cmd0[0]
    obs[i, o + 1] = cmd0[1]
    obs[i, o + 2] = cmd1[0]
    obs[i, o + 3] = cmd1[1]
    obs[i, o + 4] = w.max_speed[i] / car.max_speed
    obs[i, o + 5] = w.max_steer[i] / car.max_steer
    obs[i, o + 6] = w.steer_trim[i]
    obs[i, o + 7] = w.latency[i] / car.dt


@wp.func
def reset_world(
    i: int,
    qpos: wp.array2d[float],
    bank: TrackBank,
    car: CarParams,
    params: EnvParams,
    w: Worlds,
    seed: int,
    num_worlds: int,
):
    w.resets[i] += 1
    rng = wp.rand_init(seed, w.resets[i] * num_worlds + i)

    # Random track, random point on it, a bit off center, roughly facing along it
    num_tracks = bank.n.shape[0]
    t = wp.randi(rng, 0, num_tracks)
    if params.evaluate != 0:
        t = i % num_tracks
    j = wp.randi(rng, 0, bank.n[t])
    offset = wp.randf(rng, -START_OFFSET_FRAC, START_OFFSET_FRAC) * bank.half_width[t]
    yaw_noise = wp.randf(rng, -START_YAW_NOISE, START_YAW_NOISE)

    # A slightly different car every episode, so the policy doesn't depend on the
    # exact values in car.toml (most of them are guesses)
    latency = wp.randf(rng, params.r_latency[0], params.r_latency[1])
    steer_time = wp.randf(rng, params.r_steer_time[0], params.r_steer_time[1])
    speed_time = wp.randf(rng, params.r_speed_time[0], params.r_speed_time[1])
    max_speed = wp.randf(rng, params.r_max_speed[0], params.r_max_speed[1])
    max_steer = wp.randf(rng, params.r_max_steer[0], params.r_max_steer[1])
    trim = wp.randf(rng, -params.r_steer_trim, params.r_steer_trim)

    if params.evaluate != 0:
        j = 0
        offset = 0.0
        yaw_noise = 0.0
    if params.nominal != 0:
        latency = 1.0
        steer_time = 1.0
        speed_time = 1.0
        max_speed = 1.0
        max_steer = 1.0
        trim = 0.0

    w.latency[i] = latency * car.latency
    w.steer_time[i] = steer_time * car.steer_time
    w.speed_time[i] = speed_time * car.speed_time
    w.max_speed[i] = max_speed * car.max_speed
    w.max_steer[i] = max_steer * car.max_steer
    w.steer_trim[i] = trim

    w.track_id[i] = t
    tan = bank.tangents[t, j]
    p = bank.centerline[t, j] + bank.normals[t, j] * offset
    qpos[i, X] = p[0]
    qpos[i, Y] = p[1]
    qpos[i, YAW] = wp.atan2(tan[1], tan[0]) + yaw_noise
    qpos[i, STEER_L] = 0.0
    qpos[i, STEER_R] = 0.0

    w.idx[i] = j
    w.s_prev[i] = track_s(bank, t, p, j)
    # Standing still, wheels straight, and the last commands were "stop, straight"
    w.speed[i] = 0.0
    w.steer[i] = 0.0
    for k in range(HISTORY):
        w.commands[i, k] = wp.vec2(0.0, -1.0)
    w.steps[i] = 0


@wp.kernel
def step(
    actions: wp.array2d[float],
    qpos: wp.array2d[float],
    bank: TrackBank,
    car: CarParams,
    params: EnvParams,
    w: Worlds,
    obs: wp.array2d[float],
    reward: wp.array1d[float],
    reward_terms: wp.array2d[float],
    laps: wp.array1d[float],
    done: wp.array1d[float],
    truncated: wp.array1d[float],
    final_obs: wp.array2d[float],
    seed: int,
    num_worlds: int,
):
    i = wp.tid()

    # Actions in [-1, 1]: [steer, throttle]. Throttle maps to [0, max_speed]:
    # forward only, so the policy can't learn to wiggle back and forth
    cmd = wp.vec2(
        wp.clamp(actions[i, 0], -1.0, 1.0), wp.clamp(actions[i, 1], -1.0, 1.0)
    )
    # Steering change since last step's command, in [-2, 2]
    steer_change = cmd[0] - w.commands[i, 0][0]
    for k in range(HISTORY - 1, 0, -1):
        w.commands[i, k] = w.commands[i, k - 1]
    w.commands[i, 0] = cmd

    # Kinematic bicycle model (same as drive.py), in substeps, plus what makes the
    # real car slower to respond:
    # - latency: a command reaches the wheels `latency` s after its camera frame, so
    #   each substep uses the newest command that has arrived by then
    # - the servo and motor approach their targets gradually (first-order lag)
    x = qpos[i, X]
    y = qpos[i, Y]
    yaw = qpos[i, YAW]
    delta = w.steer[i]
    v = w.speed[i]
    h = car.dt / float(SUBSTEPS)
    steer_alpha = 1.0 - wp.exp(-h / w.steer_time[i])
    speed_alpha = 1.0 - wp.exp(-h / w.speed_time[i])
    for m in range(SUBSTEPS):
        # Commands issued d steps ago have arrived if d * dt + (time into this
        # step) >= latency
        d = int(wp.ceil((w.latency[i] - float(m) * h) / car.dt))
        a = w.commands[i, wp.clamp(d, 0, HISTORY - 1)]
        delta_target = wp.clamp(a[0] + w.steer_trim[i], -1.0, 1.0) * w.max_steer[i]
        v_target = (a[1] + 1.0) * 0.5 * w.max_speed[i]
        delta += (delta_target - delta) * steer_alpha
        v += (v_target - v) * speed_alpha

        x += v * wp.cos(yaw) * h
        y += v * wp.sin(yaw) * h
        yaw += v * wp.tan(delta) / car.wheelbase * h

    qpos[i, X] = x
    qpos[i, Y] = y
    qpos[i, YAW] = yaw
    qpos[i, STEER_L] = delta
    qpos[i, STEER_R] = delta
    w.speed[i] = v
    w.steer[i] = delta

    # Where are we on the track now?
    t = w.track_id[i]
    p = wp.vec2(x, y)
    j = nearest_local(bank, t, p, w.idx[i])
    w.idx[i] = j
    offset = wp.dot(p - bank.centerline[t, j], bank.normals[t, j])
    s = track_s(bank, t, p, j)

    # Progress, handling the jump at the start line (see Track.progress)
    length = bank.length[t]
    ds = s - w.s_prev[i]
    if ds > length * 0.5:
        ds -= length
    elif ds < -length * 0.5:
        ds += length
    w.s_prev[i] = s
    laps[i] = ds / length  # laps driven this step, for logging

    half_width = bank.half_width[t]
    off_track = wp.abs(offset) > half_width
    w.steps[i] += 1
    timeout = w.steps[i] >= params.episode_steps

    # Reward = weighted sum of terms (weights from experiment.py RewardConfig)
    offset_n = offset / half_width  # 0 = middle, +-1 = on the tape
    r_progress = params.w_progress * ds / (car.max_speed * car.dt)  # full speed ~1
    r_center = -params.w_center * offset_n * offset_n
    r_steer_rate = -params.w_steer_rate * steer_change * steer_change
    r_off_track = 0.0
    if off_track:
        # Leaving the track is only the penalty, no credit for that last step
        r_progress = 0.0
        r_center = 0.0
        r_steer_rate = 0.0
        r_off_track = -params.w_off_track

    reward[i] = r_progress + r_off_track + r_center + r_steer_rate
    reward_terms[i, 0] = r_progress
    reward_terms[i, 1] = r_off_track
    reward_terms[i, 2] = r_center
    reward_terms[i, 3] = r_steer_rate

    is_done = off_track or timeout
    done[i] = 1.0 if is_done else 0.0
    # Timeout isn't failure: PPO bootstraps the value from final_obs instead of 0
    is_truncated = timeout and not off_track
    truncated[i] = 1.0 if is_truncated else 0.0

    if is_truncated:
        write_obs(i, qpos, bank, car, w, final_obs)
    if is_done:
        reset_world(i, qpos, bank, car, params, w, seed, num_worlds)

    write_obs(i, qpos, bank, car, w, obs)


@wp.kernel
def reset_all(
    qpos: wp.array2d[float],
    bank: TrackBank,
    car: CarParams,
    params: EnvParams,
    w: Worlds,
    obs: wp.array2d[float],
    reward: wp.array1d[float],
    done: wp.array1d[float],
    truncated: wp.array1d[float],
    seed: int,
    num_worlds: int,
):
    i = wp.tid()
    reset_world(i, qpos, bank, car, params, w, seed, num_worlds)
    # Start each world at a random point in its time limit. Otherwise all worlds
    # time out on the same step forever, and training sees bursts of resets
    if params.evaluate == 0:
        w.steps[i] = wp.randi(wp.rand_init(seed + 1, i), 0, params.episode_steps)
    write_obs(i, qpos, bank, car, w, obs)
    reward[i] = 0.0
    done[i] = 0.0
    truncated[i] = 0.0


@wp.kernel
def write_tape(
    bank: TrackBank,
    w: Worlds,
    first_geom: int,
    geom_xpos: wp.array2d[wp.vec3],
    geom_xmat: wp.array2d[wp.mat33],
    geom_size: wp.array2d[wp.vec3],
):
    """Move world i's tape boxes onto its track: one thread per (world, box).

    All worlds share one set of tape geoms in the model, but the renderer reads
    each world's own copy of their poses (geom_xpos/xmat), so overwriting those
    gives every world different tape. Boxes the track doesn't need stay under the
    floor. Kinematics never moves world-fixed geoms, so this survives until the
    next call; it runs every step anyway since it's cheap and resets happen
    inside the step kernel. geom_size is per world too, for the tape width.
    """
    i, k = wp.tid()
    t = w.track_id[i]
    g = first_geom + k
    if k < bank.n_tape[t]:
        pose = bank.tape[t, k]
        c = wp.cos(pose[2])
        s = wp.sin(pose[2])
        geom_xpos[i, g] = wp.vec3(pose[0], pose[1], TAPE_HEIGHT)
        geom_xmat[i, g] = wp.mat33(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
        geom_size[i, g] = wp.vec3(
            TAPE_SPACING * TAPE_OVERLAP * 0.5, w.tape_width[i] * 0.5, TAPE_HEIGHT
        )
    else:
        geom_xpos[i, g] = wp.vec3(0.0, 0.0, HIDDEN_Z)


@wp.func
def hue_color(h: float) -> wp.vec3:
    """A fully saturated color around the color wheel, h in [0, 1)."""
    two_pi = 6.2831853
    return wp.vec3(
        0.5 + 0.5 * wp.cos(two_pi * h),
        0.5 + 0.5 * wp.cos(two_pi * (h - 0.3333)),
        0.5 + 0.5 * wp.cos(two_pi * (h - 0.6667)),
    )


@wp.func
def light_color(t: float) -> wp.vec3:
    """White, shifted warm (t > 0, like a bulb) or cool (t < 0, like daylight)."""
    return wp.vec3(1.0 + t, 1.0, 1.0 - t)


@wp.func
def luminance(c: wp.vec3) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


@wp.func
def mul_quat_wxyz(a: wp.quat, b: wp.quat) -> wp.quat:
    """MuJoCo-order (w, x, y, z) quaternion product. mujoco_warp stores cam_quat
    in MuJoCo order, not Warp's (x, y, z, w), so wp's own * doesn't apply."""
    return wp.quat(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


@wp.struct
class LookIds:
    """Where the randomized parts of the look are in the model."""

    floor_mat: int
    tape_mat: int
    num_floor_tex: int  # texture ids 0..num_floor_tex-1 are floors, 0 = the default
    texrepeat: wp.vec2  # floor pattern scale, default
    tape_rgba: wp.vec4  # default
    cam: int  # the car's camera
    cam_pos: wp.vec3  # default
    cam_quat: wp.quat  # default, MuJoCo (w, x, y, z) order
    sun: int  # the directional light
    sun_dir: wp.vec3  # default
    sun_diffuse: wp.vec3  # default
    sun_ambient: wp.vec3  # default
    sun_specular: wp.vec3  # default
    first_lamp: int  # spot lights first_lamp..first_lamp+num_lamps-1, off by default
    num_lamps: int
    floor_specular: float  # default
    floor_shininess: float  # default


@wp.kernel
def randomize_look(
    params: EnvParams,
    w: Worlds,
    ids: LookIds,
    done: wp.array1d[float],
    only_done: int,
    seed: int,
    num_worlds: int,
    mat_texid: wp.array3d[int],
    mat_rgba: wp.array2d[wp.vec4],
    mat_texrepeat: wp.array2d[wp.vec2],
    cam_pos: wp.array2d[wp.vec3],
    cam_quat: wp.array2d[wp.quat],
    mat_specular: wp.array2d[float],
    mat_shininess: wp.array2d[float],
    light_pos: wp.array2d[wp.vec3],
    light_dir: wp.array2d[wp.vec3],
    light_diffuse: wp.array2d[wp.vec3],
    light_ambient: wp.array2d[wp.vec3],
    light_specular: wp.array2d[wp.vec3],
):
    """A new random look for each world starting an episode: floor texture, color,
    and scale, tape color and width, camera pose, lighting, and image settings for
    postprocess. Writes the model's per-world arrays, which the renderer reads
    (like write_tape does with geom poses)."""
    i = wp.tid()
    if only_done != 0 and done[i] == 0.0:
        return
    rng = wp.rand_init(seed + 2, w.resets[i] * num_worlds + i)

    floor_tex = wp.randi(rng, 0, ids.num_floor_tex)
    # Floor: a random shade, tinted toward a random hue
    brightness = wp.randf(
        rng, params.v_floor_brightness[0], params.v_floor_brightness[1]
    )
    sat = wp.randf(rng, 0.0, params.v_floor_saturation)
    floor_color = brightness * (
        (1.0 - sat) * wp.vec3(1.0, 1.0, 1.0) + sat * hue_color(wp.randf(rng))
    )
    floor_scale = wp.randf(rng, params.v_floor_scale[0], params.v_floor_scale[1])

    # Tape: white some of the time, otherwise any color
    if wp.randf(rng) < params.v_tape_white:
        tape_color = wp.randf(rng, 0.8, 1.0) * wp.vec3(1.0, 1.0, 1.0)
    else:
        sat = wp.randf(rng, 0.4, 1.0)
        tape_color = wp.randf(rng, 0.2, 1.0) * (
            (1.0 - sat) * wp.vec3(1.0, 1.0, 1.0) + sat * hue_color(wp.randf(rng))
        )
    # Keep the tape visible: if it's about as bright as the floor (textures average
    # ~0.6), make it white on a dark floor or black on a light one
    floor_lum = 0.6 * luminance(floor_color)
    if wp.abs(luminance(tape_color) - floor_lum) < 0.2:
        if floor_lum < 0.5:
            tape_color = wp.vec3(0.95, 0.95, 0.95)
        else:
            tape_color = wp.vec3(0.08, 0.08, 0.08)
    tape_width = wp.randf(rng, params.v_tape_width[0], params.v_tape_width[1])

    dz = wp.randf(rng, -params.v_camera_height, params.v_camera_height)
    pitch = wp.randf(rng, -params.v_camera_pitch, params.v_camera_pitch)

    exposure = wp.randf(rng, params.v_exposure[0], params.v_exposure[1])
    contrast = wp.randf(rng, params.v_contrast[0], params.v_contrast[1])
    wb = params.v_white_balance
    gain = wp.vec3(
        1.0 + wp.randf(rng, -wb, wb),
        1.0 + wp.randf(rng, -wb, wb),
        1.0 + wp.randf(rng, -wb, wb),
    )
    noise = wp.randf(rng, 0.0, params.v_noise)

    tape_rgba = wp.vec4(tape_color[0], tape_color[1], tape_color[2], 1.0)
    if params.nominal != 0:
        floor_tex = 0
        floor_color = wp.vec3(1.0, 1.0, 1.0)
        floor_scale = 1.0
        tape_rgba = ids.tape_rgba
        tape_width = TAPE_WIDTH
        dz = 0.0
        pitch = 0.0
        exposure = 1.0
        contrast = 1.0
        gain = wp.vec3(1.0, 1.0, 1.0)
        noise = 0.0

    mat_texid[i, ids.floor_mat, 1] = floor_tex  # 1 = the RGB texture role
    mat_rgba[i, ids.floor_mat] = wp.vec4(
        floor_color[0], floor_color[1], floor_color[2], 1.0
    )
    mat_texrepeat[i, ids.floor_mat] = ids.texrepeat / floor_scale
    mat_rgba[i, ids.tape_mat] = tape_rgba
    w.tape_width[i] = tape_width

    # Camera mount: a little higher or lower, tilted a little more or less. The
    # rotation is about the camera's own x axis (image right), which is pitch
    cam_pos[i, ids.cam] = ids.cam_pos + wp.vec3(0.0, 0.0, dz)
    tilt = wp.quat(wp.cos(pitch * 0.5), wp.sin(pitch * 0.5), 0.0, 0.0)
    cam_quat[i, ids.cam] = mul_quat_wxyz(ids.cam_quat, tilt)

    w.exposure[i] = exposure
    w.contrast[i] = contrast
    w.color_gain[i] = gain
    w.noise[i] = noise

    # Lighting. Sun: a directional light from a random direction, which shades the
    # tape's edges and the floor's shine. On its own it lights a flat floor evenly
    sun = ids.sun
    elevation = wp.randf(rng, params.v_sun_elevation[0], params.v_sun_elevation[1])
    azimuth = wp.randf(rng, 0.0, 6.2831853)
    tint = params.v_light_tint
    sun_color = light_color(wp.randf(rng, -tint, tint))
    sun_brightness = wp.randf(rng, params.v_sun[0], params.v_sun[1])
    ambient = wp.randf(rng, params.v_ambient[0], params.v_ambient[1])
    if params.nominal != 0:
        light_dir[i, sun] = ids.sun_dir
        light_diffuse[i, sun] = ids.sun_diffuse
        light_ambient[i, sun] = ids.sun_ambient
        light_specular[i, sun] = ids.sun_specular
    else:
        light_dir[i, sun] = -wp.vec3(
            wp.cos(elevation) * wp.cos(azimuth),
            wp.cos(elevation) * wp.sin(azimuth),
            wp.sin(elevation),
        )
        light_diffuse[i, sun] = sun_brightness * sun_color
        light_ambient[i, sun] = ambient * sun_color
        light_specular[i, sun] = 0.4 * sun_brightness * sun_color

    # Lamps: anywhere over the floor (tracks fit within +-MAX_EXTENT), pointing
    # roughly down, each a random brightness. They fade with distance, so they
    # light the floor unevenly
    for k in range(ids.num_lamps):
        lamp = ids.first_lamp + k
        x = wp.randf(rng, -4.5, 4.5)
        y = wp.randf(rng, -4.5, 4.5)
        z = wp.randf(rng, params.v_lamp_height[0], params.v_lamp_height[1])
        aim = wp.vec3(wp.randf(rng, -0.5, 0.5), wp.randf(rng, -0.5, 0.5), -1.0)
        lamp_color = wp.randf(rng, params.v_lamp[0], params.v_lamp[1]) * light_color(
            wp.randf(rng, -tint, tint)
        )
        if params.nominal != 0:
            lamp_color = wp.vec3(0.0, 0.0, 0.0)
        light_pos[i, lamp] = wp.vec3(x, y, z)
        light_dir[i, lamp] = wp.normalize(aim)
        light_diffuse[i, lamp] = lamp_color
        light_specular[i, lamp] = 0.5 * lamp_color

    # Floor shine: matte to glossy (glare from the lamps and sun)
    gloss = wp.randf(rng, params.v_floor_gloss[0], params.v_floor_gloss[1])
    shininess = wp.randf(rng, 0.05, 0.5)
    if params.nominal != 0:
        gloss = ids.floor_specular
        shininess = ids.floor_shininess
    mat_specular[i, ids.floor_mat] = gloss
    mat_shininess[i, ids.floor_mat] = shininess


@wp.kernel
def postprocess(
    rgb: wp.array3d[wp.vec3],
    w: Worlds,
    crop: int,
    seed: int,
    pixels: wp.array4d[wp.uint8],
):
    """Rendered frame -> what the actor sees, one thread per output pixel.

    The part the Pi does too (env/preprocess.py): crop the top rows, scale to
    0..255 uint8, channels first. Training only: this world's exposure, contrast,
    white balance, and per-frame sensor noise. With the default look these are
    all off, and the output is exactly the plain crop.
    """
    i, y, x = wp.tid()
    c = rgb[i, y + crop, x]
    gain = w.color_gain[i] * w.exposure[i]
    contrast = w.contrast[i]
    noise = w.noise[i]
    rng = wp.rand_init(
        seed + 3 + w.steps[i] * 7919 + w.resets[i] * 104729,
        (i * pixels.shape[2] + y) * pixels.shape[3] + x,
    )
    for ch in range(3):
        v = c[ch] * gain[ch]
        if contrast != 1.0:
            v = (v - 0.5) * contrast + 0.5
        if noise > 0.0:
            v += noise * wp.randn(rng)
        pixels[i, ch, y, x] = wp.uint8(wp.clamp(wp.round(v * 255.0), 0.0, 255.0))
