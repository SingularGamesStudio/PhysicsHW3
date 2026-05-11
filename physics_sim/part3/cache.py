from ..common import Vec3, vec3
from .constants import BOX_FACE_AXIS_SIGN, BOX_LOCAL_CORNERS, MAX_BOX_VERTS
from .utils import safe_normalize


def box_local_corner(shape, idx: int):
    s = BOX_LOCAL_CORNERS[idx]
    h = shape.half_extents
    return vec3(s.x * h.x, s.y * h.y, s.z * h.z)


def face_local_normal(face_idx: int):
    axis, sign = BOX_FACE_AXIS_SIGN[face_idx]
    if axis == 0:
        return vec3(sign, 0.0, 0.0)
    if axis == 1:
        return vec3(0.0, sign, 0.0)
    return vec3(0.0, 0.0, sign)


def face_world_normal(cache, face_idx: int):
    axis, sign = BOX_FACE_AXIS_SIGN[face_idx]
    if axis == 0:
        return cache.axis_x * sign
    if axis == 1:
        return cache.axis_y * sign
    return cache.axis_z * sign


def face_world_center(body, cache, face_idx: int):
    axis, sign = BOX_FACE_AXIS_SIGN[face_idx]
    h = body.shape.half_extents
    if axis == 0:
        return cache.x + cache.axis_x * (sign * h.x)
    if axis == 1:
        return cache.x + cache.axis_y * (sign * h.y)
    return cache.x + cache.axis_z * (sign * h.z)


def update_body_collision_cache(body, cache, aabb_margin=0.0):
    cache.x = Vec3(body.state.x)

    R = body.R()
    cache.axis_x = safe_normalize(Vec3(R[0]), vec3(1.0, 0.0, 0.0))
    cache.axis_y = safe_normalize(Vec3(R[1]), vec3(0.0, 1.0, 0.0))
    cache.axis_z = safe_normalize(Vec3(R[2]), vec3(0.0, 0.0, 1.0))

    mn = vec3(+1.0e30, +1.0e30, +1.0e30)
    mx = vec3(-1.0e30, -1.0e30, -1.0e30)
    h = body.shape.half_extents

    for i in range(MAX_BOX_VERTS):
        s = BOX_LOCAL_CORNERS[i]
        p = cache.x + cache.axis_x * (s.x * h.x) + cache.axis_y * (s.y * h.y) + cache.axis_z * (s.z * h.z)
        cache.corners[i] = p
        mn.x = min(mn.x, p.x)
        mn.y = min(mn.y, p.y)
        mn.z = min(mn.z, p.z)
        mx.x = max(mx.x, p.x)
        mx.y = max(mx.y, p.y)
        mx.z = max(mx.z, p.z)

    margin = vec3(aabb_margin, aabb_margin, aabb_margin)
    cache.aabb.min_v = mn - margin
    cache.aabb.max_v = mx + margin
