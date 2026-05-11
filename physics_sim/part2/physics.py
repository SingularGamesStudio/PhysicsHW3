import math

from pyglm import glm

from ..common import Vec3, rotvec_to_quat
from .types import ImpulseScratch, PositionImpulseScratch


def body_inv_mass(body):
    m = float(body.mass)
    if (not math.isfinite(m)) or m <= 0.0:
        return 0.0
    return 1.0 / m


def body_world_offset(body, local_point):
    return body.R() * local_point


def body_world_point(body, local_point):
    return body.state.x + body.R() * local_point


def body_point_velocity_world(body, local_point):
    r = body.R() * local_point
    w_world = body.R() * body.state.w_body
    return body.state.v + glm.cross(w_world, r)


def apply_world_impulse(body, impulse_world, world_point, scratch=None):
    inv_m = body_inv_mass(body)
    if inv_m == 0.0:
        return
    if scratch is None:
        scratch = ImpulseScratch()
    body.state.v += inv_m * impulse_world
    scratch.r = world_point - body.state.x
    scratch.tau = glm.cross(scratch.r, impulse_world)
    scratch.w_world = body.R() * body.state.w_body
    scratch.tmp = body.I_world_inv() * scratch.tau
    scratch.w_world += scratch.tmp
    body.state.w_body = glm.transpose(body.R()) * scratch.w_world


def apply_world_position_impulse(body, impulse_world, world_point, scratch=None):
    inv_m = body_inv_mass(body)
    if inv_m == 0.0:
        return
    if scratch is None:
        scratch = PositionImpulseScratch()
    body.state.x += inv_m * impulse_world
    scratch.r = world_point - body.state.x
    scratch.delta_theta = body.I_world_inv() * glm.cross(scratch.r, impulse_world)
    scratch.dq = rotvec_to_quat(scratch.delta_theta)
    body.set_q(glm.normalize(scratch.dq * body.state.q))


def total_kinetic_energy(bodies):
    e = 0.0
    for b in bodies:
        inv_m = body_inv_mass(b)
        if inv_m == 0.0:
            continue
        m = b.mass
        e += 0.5 * m * glm.dot(b.state.v, b.state.v)
        w = b.state.w_body
        e += 0.5 * (
            w.x * b.I_body_diag.x * w.x
            + w.y * b.I_body_diag.y * w.y
            + w.z * b.I_body_diag.z * w.z
        )
    return float(e)