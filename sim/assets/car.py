"""Build the RC car scene with MjSpec, sized from config/car.toml.

The car body's origin is the rear axle center, the same reference the config uses.
Poses are written straight into qpos by the kinematic bicycle model, so there are
no actuators and contacts are off.

Run from sim/: uv run python -m assets.car   (opens the viewer)
"""

import math

import mujoco
import mujoco.viewer
import numpy as np

from assets.track import build_track, oval_centerline
from config import CarConfig, load_config

WHEEL_WIDTH = 0.025  # visual only, not worth a config entry yet


def build_scene(spec: mujoco.MjSpec) -> None:
    # MjSpec defaults to degrees like XML; everything in this sim is radians
    spec.compiler.degree = False
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    # Checker floor so motion is visible in the onboard camera
    spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.35, 0.35, 0.35],
        rgb2=[0.45, 0.45, 0.45],
        width=256,
        height=256,
    )
    floor_mat = spec.add_material(name="floor", texrepeat=[10, 10])
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"

    world = spec.worldbody
    world.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.1],
        material="floor",
    )


def _xyaxes(pitch: float) -> list[float]:
    """Camera looking along the body's +x, tilted by pitch (negative = down).

    MuJoCo cameras look along their own -z with +y as image up, so xyaxes is
    (image right, image up) in the body frame.
    """
    right = [0.0, -1.0, 0.0]
    up = [-math.sin(pitch), 0.0, math.cos(pitch)]
    return right + up


def _add_camera(body, name: str, pos, pitch: float, fovy_deg: float, resolution=None):
    cam = body.add_camera(name=name, pos=pos, fovy=fovy_deg)
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = _xyaxes(pitch)
    if resolution is not None:
        cam.resolution = resolution
    return cam


def _add_wheel(body, name: str, x: float, y: float, radius: float) -> None:
    # Cylinders point along z by default; fromto lays it sideways along y
    half = WHEEL_WIDTH / 2
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[x, y - half, 0, x, y + half, 0],
        size=[radius, 0, 0],
        rgba=[0.1, 0.1, 0.1, 1],
    )


def build_car(spec: mujoco.MjSpec, cfg: CarConfig) -> None:
    geo = cfg.geometry
    r = geo.wheel_radius
    half_track = geo.track_width / 2

    car = spec.worldbody.add_body(name="car", pos=[0, 0, r])
    car.add_joint(name="x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
    car.add_joint(name="y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
    car.add_joint(name="yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])

    # Chassis centered between the axles, narrow enough to sit between the wheels
    # (box size is half-extents)
    car.add_geom(
        name="chassis",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[geo.length / 2, geo.width / 2 - WHEEL_WIDTH, 0.02],
        pos=[geo.wheelbase / 2, 0, 0.01],
        rgba=[0.2, 0.4, 0.8, 1],
    )

    _add_wheel(car, "wheel_rl", 0, half_track, r)
    _add_wheel(car, "wheel_rr", 0, -half_track, r)

    # Front wheels get their own bodies so they can steer visibly
    max_angle = cfg.steering.max_angle
    for side, y in (("l", half_track), ("r", -half_track)):
        hub = car.add_body(name=f"hub_f{side}", pos=[geo.wheelbase, y, 0])
        hub.add_joint(
            name=f"steer_f{side}",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, 1],
            range=[-max_angle, max_angle],
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        )
        _add_wheel(hub, f"wheel_f{side}", 0, 0, r)

    cam = cfg.camera
    # camera.pos is relative to the rear axle center, which is this body's origin
    _add_camera(
        car, "front", list(cam.pos), cam.pitch, cam.fov_y_deg, [cam.width, cam.height]
    )
    # Third-person view for the viewer, not used for training
    _add_camera(car, "chase", [-0.6, 0, 0.35], math.radians(-25), 60)


def build_model(
    cfg: CarConfig, centerline: np.ndarray | None = None
) -> tuple[mujoco.MjSpec, mujoco.MjModel]:
    spec = mujoco.MjSpec()
    build_scene(spec)
    if centerline is not None:
        build_track(spec, centerline)
    build_car(spec, cfg)
    return spec, spec.compile()


if __name__ == "__main__":
    spec, model = build_model(load_config(), oval_centerline())
    mujoco.viewer.launch(model)
