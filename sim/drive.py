"""Drive the sim car around the track with an Xbox controller.

Right trigger = forward, left trigger = reverse, left stick = steer, A = reset.
Run from sim/: uv run python drive.py
"""

import math
import time

import mujoco
import mujoco.viewer

from assets.car import build_model
from assets.track import oval_centerline, start_pose
from config import load_config
from controller import XboxController

cfg = load_config()
centerline = oval_centerline()
spec, model = build_model(cfg, centerline)

data = mujoco.MjData(model)
pad = XboxController()

DT = cfg.control.dt  # seconds per step (1 / rate_hz)
L = cfg.geometry.wheelbase

# qpos layout, from the joint order in build_car:
# 0 = x, 1 = y, 2 = yaw, 3 = left front steer, 4 = right front steer
X, Y, YAW, STEER_L, STEER_R = range(5)


def reset():
    data.qpos[:] = 0
    data.qpos[X], data.qpos[Y], data.qpos[YAW] = start_pose(centerline)
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
        viewer.sync()
        time.sleep(DT)  # run in real time
