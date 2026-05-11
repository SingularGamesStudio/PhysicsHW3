from pyglm import glm

from ..common import Mat3, Vec3, integrate_orientation_body, mat3_from_diag, skew_mat
from .physics import body_inv_mass
from .types import GyroScratch


def integrate_body_velocities_implicit_gyro(body, dt, force_world, torque_world, scratch: GyroScratch, newton_iters=8):
    inv_m = body_inv_mass(body)
    if inv_m > 0.0:
        body.state.v += dt * inv_m * force_world
    I_diag = body.I_body_diag
    w_n = body.state.w_body
    scratch.w = Vec3(w_n)
    scratch.tau_body = glm.transpose(body.R()) * torque_world
    for _ in range(int(newton_iters)):
        scratch.Iw = Vec3(I_diag.x * scratch.w.x, I_diag.y * scratch.w.y, I_diag.z * scratch.w.z)
        scratch.diff = Vec3(
            I_diag.x * (scratch.w.x - w_n.x),
            I_diag.y * (scratch.w.y - w_n.y),
            I_diag.z * (scratch.w.z - w_n.z),
        )
        scratch.f = scratch.diff + dt * glm.cross(scratch.w, scratch.Iw) - dt * scratch.tau_body
        if glm.length(scratch.f) < 1e-12:
            break
        scratch.m0 = skew_mat(scratch.Iw)
        scratch.m1 = skew_mat(scratch.w)
        scratch.m2 = scratch.m1 * mat3_from_diag(I_diag)
        scratch.m2 = scratch.m2 - scratch.m0
        scratch.m1 = mat3_from_diag(I_diag) + dt * scratch.m2
        scratch.delta = glm.inverse(scratch.m1) * (-scratch.f)
        scratch.w = scratch.w + scratch.delta
        if glm.length(scratch.delta) < 1e-12:
            break
    body.state.w_body = scratch.w


def integrate_body_pose_symplectic(body, dt):
    body.state.x += dt * body.state.v
    body.set_q(integrate_orientation_body(body.state.q, body.state.w_body, dt))
