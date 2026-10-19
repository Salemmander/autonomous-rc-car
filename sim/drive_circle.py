import math
import time

import mujoco
import mujoco.viewer

from assets.car import build_model
from config import load_config

cfg = load_config()
spec, model = build_model(cfg)

data = mujoco.MjData(model)

SPEED = 0.5  # m/s
STEER = 0.3  # front wheel angle in radians, positive = left
DT = cfg.control.dt  # seconds per step (1 / rate_hz)
L = cfg.geometry.wheelbase

# qpos layout, from the joint order in build_car:
# 0 = x, 1 = y, 2 = yaw, 3 = left front steer, 4 = right front steer
X, Y, YAW, STEER_L, STEER_R = range(5)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        x, y, yaw = data.qpos[X], data.qpos[Y], data.qpos[YAW]

        # Kinematic bicycle model: move forward along the heading, and turn at a
        # rate set by speed, steering angle, and wheelbase
        data.qpos[X] = x + SPEED * math.cos(yaw) * DT
        data.qpos[Y] = y + SPEED * math.sin(yaw) * DT
        data.qpos[YAW] = yaw + SPEED * math.tan(STEER) / L * DT

        # Turn the front wheels so the steering is visible
        data.qpos[STEER_L] = STEER
        data.qpos[STEER_R] = STEER

        # Recompute where every geom and camera is from the new qpos (no physics)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(DT)  # run in real time
