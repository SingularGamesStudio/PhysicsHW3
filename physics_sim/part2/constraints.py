import math

from pyglm import glm

from ..common import Vec3, safe_norm
from .physics import body_inv_mass, body_point_velocity_world, body_world_point
from .types import MassScratch


def effective_mass_scalar_one_body(body, world_point, n_world, scratch=None):
    inv_m = body_inv_mass(body)
    if inv_m == 0.0:
        return 0.0
    if scratch is None:
        scratch = MassScratch()
    scratch.r = world_point - body.state.x
    scratch.rn = glm.cross(scratch.r, n_world)
    scratch.tmp = body.I_world_inv() * scratch.rn
    return inv_m + glm.dot(scratch.rn, scratch.tmp)


def effective_mass_scalar_two_body(body_a, point_a, body_b, point_b, n_world, scratch_a=None, scratch_b=None):
    k = 0.0
    inv_ma = body_inv_mass(body_a)
    if inv_ma > 0.0:
        if scratch_a is None:
            scratch_a = MassScratch()
        scratch_a.r = point_a - body_a.state.x
        scratch_a.rn = glm.cross(scratch_a.r, n_world)
        scratch_a.tmp = body_a.I_world_inv() * scratch_a.rn
        k += inv_ma + glm.dot(scratch_a.rn, scratch_a.tmp)
    inv_mb = body_inv_mass(body_b)
    if inv_mb > 0.0:
        if scratch_b is None:
            scratch_b = MassScratch()
        scratch_b.r = point_b - body_b.state.x
        scratch_b.rn = glm.cross(scratch_b.r, n_world)
        scratch_b.tmp = body_b.I_world_inv() * scratch_b.rn
        k += inv_mb + glm.dot(scratch_b.rn, scratch_b.tmp)
    return k


def fixed_anchor_distance_info(body, local_anchor, fixed_point, rest_length):
    p = body_world_point(body, local_anchor)
    d = p - fixed_point
    l = safe_norm(d)
    n = d / l if l > 1e-12 else Vec3(1.0, 0.0, 0.0)
    C = l - float(rest_length)
    return C, l, p, n


def two_body_distance_info(body_a, local_anchor_a, body_b, local_anchor_b, rest_length):
    p_a = body_world_point(body_a, local_anchor_a)
    p_b = body_world_point(body_b, local_anchor_b)
    d = p_b - p_a
    l = safe_norm(d)
    n = d / l if l > 1e-12 else Vec3(1.0, 0.0, 0.0)
    C = l - float(rest_length)
    return C, l, p_a, p_b, n


def fixed_anchor_spring_force(body, spring):
    C, _, p, n = fixed_anchor_distance_info(
        body, spring.local_anchor, spring.fixed_point, spring.rest_length
    )
    v_point = body_point_velocity_world(body, spring.local_anchor)
    vn = glm.dot(v_point, n)
    force_mag = -spring.stiffness * C - spring.damping * vn
    force = n * force_mag
    torque = glm.cross(p - body.state.x, force)
    return C, force, torque


def soft_constraint_coeffs(dt, hertz, damping_ratio):
    omega = 2.0 * math.pi * float(hertz)
    a1 = 2.0 * float(damping_ratio) + omega * dt
    a2 = dt * omega * a1
    a3 = 1.0 / (1.0 + a2)
    bias_rate = omega / a1 if a1 > 0.0 else 0.0
    mass_coeff = a2 * a3
    impulse_coeff = a3
    return bias_rate, mass_coeff, impulse_coeff
