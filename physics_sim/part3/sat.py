import math

from pyglm import glm

from ..common import Vec3, vec3
from .cache import face_world_normal
from .constants import (
    AXIS_CROSS_EPS,
    BOX_BOX_CROSS_PARALLEL_SKIP_DOT,
    BOX_BOX_CROSS_WIN_EPS,
    BOX_BOX_FACE_BIAS,
    SAT_EPS,
)
from .types import SatAxisResult
from .utils import abs_dot, clampf


def project_box_radius_on_axis(body, cache, axis):
    h = body.shape.half_extents
    return (
        abs_dot(axis, cache.axis_x) * h.x +
        abs_dot(axis, cache.axis_y) * h.y +
        abs_dot(axis, cache.axis_z) * h.z
    )


def box_box_contact_scale(body_a, body_b):
    ha = body_a.shape.half_extents
    hb = body_b.shape.half_extents
    return float(min(ha.x, ha.y, ha.z, hb.x, hb.y, hb.z))


def axis_overlap_for_boxes(body_a, cache_a, body_b, cache_b, axis, center_delta):
    dist = abs(glm.dot(center_delta, axis))
    ra = project_box_radius_on_axis(body_a, cache_a, axis)
    rb = project_box_radius_on_axis(body_b, cache_b, axis)
    return ra + rb - dist


def axis_candidate_score(overlap, axis_type, normal, prev_normal, scale):
    score = float(overlap)

    if axis_type == 2:
        score += max(2.0e-3, 0.010 * scale)

    if prev_normal is not None:
        align = clampf(abs(glm.dot(normal, prev_normal)), 0.0, 1.0)
        if axis_type == 2:
            score += (1.0 - align) * max(7.5e-4, 0.004 * scale)
        else:
            score += (1.0 - align) * max(1.5e-4, 0.00075 * scale)

    return score


def support_face_index_from_normal(cache, outward_normal_world):
    best_face = 0
    best_dot = -1.0e30
    for face_idx in range(6):
        d = glm.dot(face_world_normal(cache, face_idx), outward_normal_world)
        if d > best_dot:
            best_dot = d
            best_face = face_idx
    return best_face


def incident_face_index(cache, ref_normal_world):
    best_face = 0
    best_dot = +1.0e30
    for face_idx in range(6):
        n = face_world_normal(cache, face_idx)
        d = glm.dot(n, ref_normal_world)
        if d < best_dot:
            best_dot = d
            best_face = face_idx
    return best_face


def sat_test_box_box(body_a, cache_a, body_b, cache_b):
    d = body_b.state.x - body_a.state.x

    axes_a = (cache_a.axis_x, cache_a.axis_y, cache_a.axis_z)
    axes_b = (cache_b.axis_x, cache_b.axis_y, cache_b.axis_z)

    best_face = SatAxisResult(hit=False, penetration=1.0e30, normal=vec3(1.0, 0.0, 0.0))
    best_cross = SatAxisResult(hit=False, penetration=1.0e30, normal=vec3(1.0, 0.0, 0.0))

    for i in range(3):
        axis = axes_a[i]
        dist = abs(glm.dot(d, axis))
        ra = project_box_radius_on_axis(body_a, cache_a, axis)
        rb = project_box_radius_on_axis(body_b, cache_b, axis)
        overlap = ra + rb - dist
        if overlap < -SAT_EPS:
            return SatAxisResult(hit=False)

        n = axis if glm.dot(d, axis) >= 0.0 else -axis
        if overlap < best_face.penetration:
            best_face.hit = True
            best_face.axis_type = 0
            best_face.axis_i = i
            best_face.axis_j = -1
            best_face.normal = Vec3(n)
            best_face.penetration = float(overlap)

    for j in range(3):
        axis = axes_b[j]
        dist = abs(glm.dot(d, axis))
        ra = project_box_radius_on_axis(body_a, cache_a, axis)
        rb = project_box_radius_on_axis(body_b, cache_b, axis)
        overlap = ra + rb - dist
        if overlap < -SAT_EPS:
            return SatAxisResult(hit=False)

        n = axis if glm.dot(d, axis) >= 0.0 else -axis
        if overlap < best_face.penetration:
            best_face.hit = True
            best_face.axis_type = 1
            best_face.axis_i = -1
            best_face.axis_j = j
            best_face.normal = Vec3(n)
            best_face.penetration = float(overlap)

    for i in range(3):
        ai = axes_a[i]
        for j in range(3):
            bj = axes_b[j]

            if abs(glm.dot(ai, bj)) >= BOX_BOX_CROSS_PARALLEL_SKIP_DOT:
                continue

            axis = glm.cross(ai, bj)
            lsq = glm.dot(axis, axis)
            if lsq <= AXIS_CROSS_EPS:
                continue

            axis *= 1.0 / math.sqrt(lsq)

            dist = abs(glm.dot(d, axis))
            ra = project_box_radius_on_axis(body_a, cache_a, axis)
            rb = project_box_radius_on_axis(body_b, cache_b, axis)
            overlap = ra + rb - dist
            if overlap < -SAT_EPS:
                return SatAxisResult(hit=False)

            n = axis if glm.dot(d, axis) >= 0.0 else -axis
            if overlap < best_cross.penetration:
                best_cross.hit = True
                best_cross.axis_type = 2
                best_cross.axis_i = i
                best_cross.axis_j = j
                best_cross.normal = Vec3(n)
                best_cross.penetration = float(overlap)

    if not best_face.hit:
        return best_cross
    if not best_cross.hit:
        return best_face

    if best_face.penetration <= best_cross.penetration + max(BOX_BOX_CROSS_WIN_EPS, BOX_BOX_FACE_BIAS):
        return best_face
    return best_cross
