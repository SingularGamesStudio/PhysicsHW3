from pyglm import glm

from ..common import BoxShape, RigidBody, RigidBodyState, Quat, as_vec3, quat_identity, vec3
from .utils import vec3_zero


def _tilt_quat(rx=0.0, ry=0.0, rz=0.0):
    qx = glm.angleAxis(float(rx), vec3(1.0, 0.0, 0.0))
    qy = glm.angleAxis(float(ry), vec3(0.0, 1.0, 0.0))
    qz = glm.angleAxis(float(rz), vec3(0.0, 0.0, 1.0))
    return glm.normalize(qz * qy * qx)


def _make_box_body(shape, mass, pos, q=None, v=None, w_body=None):
    return RigidBody(
        shape=shape,
        mass=float(mass),
        state=RigidBodyState(
            x=as_vec3(pos),
            q=Quat(quat_identity() if q is None else q),
            v=as_vec3(vec3_zero() if v is None else v),
            w_body=as_vec3(vec3_zero() if w_body is None else w_body),
        ),
    )


def make_part3_box_pile_bodies(shape=None, mass=1.0, plane_y=0.0, gap=0.04, drop_height=0.90):
    """
    Ten boxes, initially close but non-overlapping, centered over the plane.
    Layout:
        4 on the bottom
        3 on the next layer
        2 above that
        1 on top
    """
    if shape is None:
        shape = BoxShape(half_extents=vec3(0.50, 0.35, 0.32))

    hx, hy, hz = shape.half_extents.x, shape.half_extents.y, shape.half_extents.z
    sx = 2.0 * hx + gap
    sy = 2.0 * hy + gap
    sz = 2.0 * hz + gap

    y0 = plane_y + hy + drop_height
    y1 = y0 + sy
    y2 = y1 + sy
    y3 = y2 + sy

    bodies = []

    base_positions = [
        vec3(-0.5 * sx, y0, -0.5 * sz),
        vec3(+0.5 * sx, y0, -0.5 * sz),
        vec3(-0.5 * sx, y0, +0.5 * sz),
        vec3(+0.5 * sx, y0, +0.5 * sz),
    ]
    for p in base_positions:
        bodies.append(_make_box_body(shape, mass, p))

    mid_positions = [
        vec3(-0.5 * sx, y1, -0.22 * sz),
        vec3(+0.5 * sx, y1, -0.22 * sz),
        vec3(0.0,       y1, +0.78 * sz),
    ]
    mid_rot = [
        _tilt_quat(0.015, 0.000, -0.010),
        _tilt_quat(-0.012, 0.008, 0.010),
        _tilt_quat(0.010, -0.010, -0.012),
    ]
    for p, q in zip(mid_positions, mid_rot):
        bodies.append(_make_box_body(shape, mass, p, q=q))

    upper_positions = [
        vec3(-0.5 * sx, y2, +0.18 * sz),
        vec3(+0.5 * sx, y2, +0.18 * sz),
    ]
    upper_rot = [
        _tilt_quat(0.018, 0.012, -0.015),
        _tilt_quat(-0.018, -0.008, 0.014),
    ]
    for p, q in zip(upper_positions, upper_rot):
        bodies.append(_make_box_body(shape, mass, p, q=q))

    bodies.append(
        _make_box_body(
            shape,
            mass,
            vec3(0.0, y3, 0.02 * sz),
            q=_tilt_quat(0.020, -0.015, 0.012),
        )
    )

    return bodies
