"""Drive the sim car around the track with an Xbox controller.

Right trigger = forward, left trigger = reverse, left stick = steer, A = reset.
Run from sim/: uv run python drive.py
"""

import math
import time

import mujoco
import mujoco.viewer

from assets.car import build_model
from assets.track import Track, oval_centerline
from config import load_config
from controller import XboxController

cfg = load_config()
track = Track.from_centerline(oval_centerline())
spec, model = build_model(cfg, track)

data = mujoco.MjData(model)
pad = XboxController()

DT = cfg.control.dt  # seconds per step (1 / rate_hz)
L = cfg.geometry.wheelbase

# qpos layout, from the joint order in build_car:
# 0 = x, 1 = y, 2 = yaw, 3 = left front steer, 4 = right front steer
X, Y, YAW, STEER_L, STEER_R = range(5)


distance = 0.0  # meters driven along the track since reset, backwards counts negative
s_prev = 0.0  # distance along the track last step, set properly by reset()


def reset():
    global distance, s_prev
    distance = 0.0
    s_prev = track.project(track.centerline[0])[2]
    data.qpos[:] = 0
    data.qpos[X], data.qpos[Y], data.qpos[YAW] = track.start_pose()
    mujoco.mj_forward(model, data)


reset()
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        pad.poll()
        if pad.reset_pressed:
            pad.reset_pressed = False
            reset()

        speed = pad.throttle * cfg.throttle.max_speed
        # Stick right is positive, but positive steering (yaw) is to the left
        steer = -pad.steering * cfg.steering.max_angle

        x, y, yaw = data.qpos[X], data.qpos[Y], data.qpos[YAW]

        # Kinematic bicycle model, same as drive_circle.py
        data.qpos[X] = x + speed * math.cos(yaw) * DT
        data.qpos[Y] = y + speed * math.sin(yaw) * DT
        data.qpos[YAW] = yaw + speed * math.tan(steer) / L * DT

        data.qpos[STEER_L] = steer
        data.qpos[STEER_R] = steer

        mujoco.mj_forward(model, data)

        i, offset, s = track.project(data.qpos[[X, Y]])
        distance += track.progress(s_prev, s)
        s_prev = s
        on_track = abs(offset) <= track.width / 2
        viewer.set_texts(
            (
                None,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                "nearest point\noffset\non track\nlap position\ndistance\nlaps",
                f"{i}\n{offset:+.2f} m\n{'yes' if on_track else 'NO'}\n"
                f"{s:.2f} / {track.length:.2f} m\n{distance:+.2f} m\n"
                f"{distance / track.length:+.2f}",
            )
        )
        viewer.sync()
        time.sleep(DT)  # run in real time
