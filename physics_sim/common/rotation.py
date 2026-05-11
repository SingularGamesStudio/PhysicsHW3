import math

from pyglm import glm

from .math3d import Quat, Vec3


def quat_identity():
    return Quat(1.0, 0.0, 0.0, 0.0)


def integrate_orientation_body(q, w_body, dt):
    speed = glm.length(w_body)
    if speed <= 1e-12:
        return Quat(q)
    dq = glm.normalize(glm.angleAxis(speed * dt, w_body / speed))
    return glm.normalize(q * dq)


def integrate_orientation_world(q, w_world, dt):
    speed = glm.length(w_world)
    if speed <= 1e-12:
        return Quat(q)
    dq = glm.normalize(glm.angleAxis(speed * dt, w_world / speed))
    return glm.normalize(dq * q)


def angular_velocity_world_from_quat_delta(prev_q, q, dt):
    dq = q * glm.inverse(prev_q)
    if dq.w < 0.0:
        dq = -dq
    axis = Vec3(dq.x, dq.y, dq.z)
    s = glm.length(axis)
    if s <= 1e-12:
        return Vec3(0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(s, dq.w)
    return axis * (angle / (dt * s))


def rotvec_to_quat(rotvec):
    angle = glm.length(rotvec)
    if angle <= 1e-12:
        return quat_identity()
    return glm.normalize(glm.angleAxis(angle, rotvec / angle))
