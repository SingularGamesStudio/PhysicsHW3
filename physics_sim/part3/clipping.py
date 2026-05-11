from pyglm import glm

from ..common import Vec3, vec3
from .constants import BOX_FACE_VERTS, CONTACT_SLOP, MANIFOLD_KEEP_SLOP
from .utils import safe_normalize


def load_face_polygon_world(cache, face_idx: int, out_clip_verts, feature_face_bias=0):
    verts = BOX_FACE_VERTS[face_idx]
    for i in range(4):
        out_clip_verts[i].p = Vec3(cache.corners[verts[i]])
        out_clip_verts[i].feature = (feature_face_bias << 4) | (verts[i] & 0xF)
    return 4


def clip_polygon_against_plane(in_verts, in_count, plane_n, plane_offset, out_verts):
    if in_count <= 0:
        return 0

    out_count = 0

    def emit(p, feature):
        nonlocal out_count
        out_verts[out_count].p = Vec3(p)
        out_verts[out_count].feature = feature
        out_count += 1

    prev = in_verts[in_count - 1]
    prev_dist = glm.dot(plane_n, prev.p) - plane_offset
    prev_in = prev_dist <= CONTACT_SLOP

    for i in range(in_count):
        curr = in_verts[i]
        curr_dist = glm.dot(plane_n, curr.p) - plane_offset
        curr_in = curr_dist <= CONTACT_SLOP

        if prev_in != curr_in:
            denom = prev_dist - curr_dist
            t = prev_dist / denom if abs(denom) > 1.0e-8 else 0.0
            emit(prev.p + (curr.p - prev.p) * t, ((prev.feature & 0xF) << 4) | (curr.feature & 0xF))
        if curr_in:
            emit(curr.p, curr.feature)

        prev = curr
        prev_dist = curr_dist
        prev_in = curr_in

    return out_count


def orthonormal_basis_from_normal(n):
    if abs(n.y) < 0.9:
        t1 = safe_normalize(glm.cross(n, vec3(0.0, 1.0, 0.0)), vec3(1.0, 0.0, 0.0))
    else:
        t1 = safe_normalize(glm.cross(n, vec3(1.0, 0.0, 0.0)), vec3(0.0, 0.0, 1.0))
    t2 = safe_normalize(glm.cross(n, t1), vec3(0.0, 0.0, 1.0))
    return t1, t2


def select_up_to_4_extreme_points(points, count, normal, out_indices):
    if count <= 4:
        for i in range(count):
            out_indices[i] = i
        return count

    t1, t2 = orthonormal_basis_from_normal(normal)

    best = [-1, -1, -1, -1]
    best_v = [-1.0e30, -1.0e30, -1.0e30, -1.0e30]

    for i in range(count):
        p = points[i]
        u = glm.dot(t1, p)
        v = glm.dot(t2, p)

        if u > best_v[0]:
            best_v[0] = u
            best[0] = i
        if -u > best_v[1]:
            best_v[1] = -u
            best[1] = i
        if v > best_v[2]:
            best_v[2] = v
            best[2] = i
        if -v > best_v[3]:
            best_v[3] = -v
            best[3] = i

    used = set()
    out_count = 0
    for idx in best:
        if idx >= 0 and idx not in used:
            used.add(idx)
            out_indices[out_count] = idx
            out_count += 1

    return out_count


def build_reference_side_planes(cache, ref_face: int, ref_n: Vec3):
    verts = BOX_FACE_VERTS[ref_face]
    face_center = sum((cache.corners[idx] for idx in verts), vec3(0.0, 0.0, 0.0)) * 0.25
    side_normals = [vec3(0.0, 0.0, 0.0) for _ in range(4)]
    side_offsets = [0.0] * 4

    for edge_i in range(4):
        v0 = cache.corners[verts[edge_i]]
        v1 = cache.corners[verts[(edge_i + 1) & 3]]
        side_n = safe_normalize(glm.cross(v1 - v0, ref_n), vec3(1.0, 0.0, 0.0))
        side_offset = glm.dot(side_n, v0)
        if glm.dot(side_n, face_center) > side_offset:
            side_n = -side_n
            side_offset = glm.dot(side_n, v0)
        side_normals[edge_i] = Vec3(side_n)
        side_offsets[edge_i] = float(side_offset)

    return side_normals, side_offsets


def point_inside_reference_side_planes(p, side_normals, side_offsets, eps=MANIFOLD_KEEP_SLOP):
    return all(glm.dot(side_normals[i], p) <= side_offsets[i] + eps for i in range(4))
