"""Stop-sign 3D asset (textured octagon plate + pole) and geom pool.

Built at ~1:10 RC car scale (the car is 30 cm long with its camera 15 cm high):
- Red octagonal sign face, 12 cm across, centered 14 cm above the floor
- White border and bold "STOP" lettering on the front face (+x local)
- Matte gray metal back and rim (so viewing the back of a sign across a curve
  never looks like an oncoming stop sign)
- Thin gray pole supporting the plate, placed just outside the right tape edge

Like tape boxes (rcsim/assets/track.py), the GPU env parks a shared pool of
`MAX_SIGNS` sign geoms under the floor and moves each world's copy onto its own
track before rendering (rcsim/env/kernels.py write_signs).
"""

import math

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from rcsim.assets.track import HIDDEN_Z

MAX_SIGNS = 4  # max stop signs per track in the GPU geom pool
SIGN_RADIUS = 0.06  # m, circumradius -> ~12 cm across
SIGN_THICKNESS = 0.004  # m, thin plate
SIGN_HEIGHT = 0.14  # m, center of octagon above the floor (top ~20 cm)
POLE_RADIUS = 0.005  # m, 1 cm diameter pole
SIGN_EDGE_OFFSET = 0.07  # m, outside the right tape edge so the car won't clip it


def _draw_letter(
    draw: ImageDraw.ImageDraw,
    ch: str,
    x0: float,
    y0: float,
    w: float,
    h: float,
    stroke: float,
    color: tuple[int, int, int],
) -> None:
    """Crisp geometric block letters for S, T, O, P (no external font files needed)."""
    s = stroke
    x1, y1 = x0 + w, y0 + h
    ym = y0 + h / 2

    def box(ax: float, ay: float, bx: float, by: float) -> None:
        draw.rectangle([round(ax), round(ay), round(bx), round(by)], fill=color)

    if ch == "S":
        box(x0, y0, x1, y0 + s)
        box(x0, y0, x0 + s, ym + s / 2)
        box(x0, ym - s / 2, x1, ym + s / 2)
        box(x1 - s, ym - s / 2, x1, y1)
        box(x0, y1 - s, x1, y1)
    elif ch == "T":
        box(x0, y0, x1, y0 + s)
        box(x0 + (w - s) / 2, y0, x0 + (w + s) / 2, y1)
    elif ch == "O":
        box(x0, y0, x1, y0 + s)
        box(x0, y1 - s, x1, y1)
        box(x0, y0, x0 + s, y1)
        box(x1 - s, y0, x1, y1)
    elif ch == "P":
        box(x0, y0, x0 + s, y1)
        box(x0, y0, x1, y0 + s)
        box(x1 - s, y0, x1, ym + s / 2)
        box(x0, ym - s / 2, x1, ym + s / 2)


def stop_sign_texture(size: int = 256) -> np.ndarray:
    """(size, size, 3) uint8 RGB texture: red octagon + white border + 'STOP'.

    The corners outside the octagon are matte gray; back/edge mesh vertices map
    their UVs to (0, 0) so the back of the sign renders as gray metal.
    """
    gray = (130, 134, 140)
    red = (210, 28, 28)
    white = (245, 245, 245)

    img = Image.new("RGB", (size, size), gray)
    draw = ImageDraw.Draw(img)

    cx = cy = size / 2

    def octagon(radius: float) -> list[tuple[float, float]]:
        return [
            (
                cx + radius * math.cos(math.radians(22.5 + 45 * k)),
                cy + radius * math.sin(math.radians(22.5 + 45 * k)),
            )
            for k in range(8)
        ]

    # Full-radius red octagon, inset white border ring, inner red field
    r_outer = size * 0.495
    draw.polygon(octagon(r_outer), fill=red)
    draw.polygon(octagon(r_outer * 0.94), fill=white)
    draw.polygon(octagon(r_outer * 0.86), fill=red)

    # Centered "STOP" lettering
    letter_h = size * 0.28
    letter_w = size * 0.145
    gap = size * 0.032
    stroke = size * 0.038
    total_w = 4 * letter_w + 3 * gap
    x = cx - total_w / 2
    y = cy - letter_h / 2
    for ch in "STOP":
        _draw_letter(draw, ch, x, y, letter_w, letter_h, stroke, white)
        x += letter_w + gap

    return np.array(img, dtype=np.uint8)


def _octagon_mesh_data() -> tuple[list[float], list[float], list[int]]:
    """Thin 8-sided prism in the local yz plane; front face points along +x."""
    n = 8
    r = SIGN_RADIUS
    half_t = SIGN_THICKNESS / 2
    angles = [math.radians(22.5 + 45 * k) for k in range(n)]

    verts: list[list[float]] = []
    uvs: list[list[float]] = []

    # Vertex 0: front center, Vertex 1: back center (maps to gray corner UV)
    verts.append([half_t, 0.0, 0.0])
    uvs.append([0.5, 0.5])
    verts.append([-half_t, 0.0, 0.0])
    uvs.append([0.02, 0.02])

    # Front ring (vertices 2..9): standing in front of +x looking toward -x,
    # +y is to the viewer's left and +z is up.
    for a in angles:
        y, z = r * math.cos(a), r * math.sin(a)
        verts.append([half_t, y, z])
        uvs.append([0.5 + y / (2 * r), 0.5 - z / (2 * r)])

    # Back ring (vertices 10..17): all map to gray metal corner
    for a in angles:
        y, z = r * math.cos(a), r * math.sin(a)
        verts.append([-half_t, y, z])
        uvs.append([0.02, 0.02])

    faces: list[list[int]] = []
    for k in range(n):
        f0, f1 = 2 + k, 2 + (k + 1) % n
        b0, b1 = 2 + n + k, 2 + n + (k + 1) % n
        faces.append([0, f0, f1])  # front (+x normal)
        faces.append([1, b1, b0])  # back (-x normal)
        faces.append([f0, b0, b1])  # side rim
        faces.append([f0, b1, f1])

    return (
        np.array(verts, dtype=np.float32).flatten().tolist(),
        np.array(uvs, dtype=np.float32).flatten().tolist(),
        np.array(faces, dtype=np.int32).flatten().tolist(),
    )


def add_stop_sign_assets(spec: mujoco.MjSpec) -> None:
    """Register the octagon mesh, STOP texture, and sign/pole materials."""
    verts, uvs, faces = _octagon_mesh_data()
    mesh = spec.add_mesh(name="stop_plate")
    mesh.uservert = verts
    mesh.usertexcoord = uvs
    mesh.userface = faces
    mesh.userfacetexcoord = faces

    rgb = stop_sign_texture()
    tex = spec.add_texture(
        name="stop_sign",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=rgb.shape[1],
        height=rgb.shape[0],
        nchannel=3,
    )
    tex.data = rgb.tobytes()

    sign_mat = spec.add_material(name="stop_sign", specular=0.1, shininess=0.1)
    sign_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "stop_sign"

    spec.add_material(
        name="sign_pole",
        rgba=[0.45, 0.46, 0.50, 1.0],
        specular=0.2,
        shininess=0.2,
    )


def build_sign_pool(spec: mujoco.MjSpec, count: int = MAX_SIGNS) -> None:
    """`count` stop signs (plate + pole pairs) parked under the floor.

    Geoms are added contiguously as sign_face_0..N-1 then sign_pole_0..N-1 so
    the GPU kernel can index them as `first_face + k` and `first_pole + k`.
    """
    for k in range(count):
        spec.worldbody.add_geom(
            name=f"sign_face_{k}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="stop_plate",
            material="stop_sign",
            pos=[0, 0, HIDDEN_Z],
        )
    pole_half_h = (SIGN_HEIGHT - SIGN_RADIUS * 0.5) / 2
    for k in range(count):
        spec.worldbody.add_geom(
            name=f"sign_pole_{k}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[POLE_RADIUS, pole_half_h, 0],
            material="sign_pole",
            pos=[0, 0, HIDDEN_Z],
        )


def set_model_signs(
    model: mujoco.MjModel, poses: list[tuple[float, float, float]]
) -> None:
    """Position CPU MuJoCo sign pool geoms from a list of (x, y, yaw) poses."""
    first_face = model.geom("sign_face_0").id
    first_pole = model.geom("sign_pole_0").id
    pole_z = (SIGN_HEIGHT - SIGN_RADIUS * 0.5) * 0.5
    for k in range(MAX_SIGNS):
        gf = first_face + k
        gp = first_pole + k
        # Pool geoms compile with identity quat (sameframe=BODYROT), which makes
        # mj_kinematics ignore runtime edits to model.geom_quat unless cleared.
        model.geom_sameframe[gf] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        model.geom_sameframe[gp] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        if k < len(poses):
            x, y, yaw = poses[k]
            quat = [math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)]
            model.geom_pos[gf] = [x, y, SIGN_HEIGHT]
            model.geom_quat[gf] = quat
            model.geom_pos[gp] = [
                x - math.cos(yaw) * 0.006,
                y - math.sin(yaw) * 0.006,
                pole_z,
            ]
            model.geom_quat[gp] = quat
        else:
            model.geom_pos[gf] = [0.0, 0.0, HIDDEN_Z]
            model.geom_pos[gp] = [0.0, 0.0, HIDDEN_Z]
