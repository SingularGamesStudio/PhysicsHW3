import math

from pyglm import glm

from ..common import Vec3, angular_velocity_world_from_quat_delta, vec3
from .constants import CONTACT_SLOP, EPS


def clampf(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def safe_inv(x):
    return 0.0 if abs(x) <= EPS else (1.0 / x)


def safe_normalize(v, fallback=vec3(1.0, 0.0, 0.0)):
    lsq = glm.dot(v, v)
    if lsq <= EPS:
        return Vec3(fallback)
    return v * (1.0 / math.sqrt(lsq))


def abs_dot(a, b):
    return abs(glm.dot(a, b))


def signf(x):
    return -1.0 if x < 0.0 else 1.0


def vec3_zero():
    return vec3(0.0, 0.0, 0.0)


def clamped_contact_error(separation, slop):
    return min(0.0, separation + slop)


def clamped_contact_error_with_speed_limit(separation, slop, dt, max_push_speed):
    C = min(0.0, separation + slop)
    return max(C, -max_push_speed * dt) if max_push_speed > 0.0 else C


def reconstruct_velocities_from_pose_delta(body, prev_x, prev_q, dt):
    body.state.v = (body.state.x - prev_x) / dt
    w_world = angular_velocity_world_from_quat_delta(prev_q, body.state.q, dt)
    body.state.w_body = glm.transpose(body.R()) * w_world


def count_total_contacts(manifolds):
    total = 0
    for m in manifolds:
        total += m.point_count
    return total
