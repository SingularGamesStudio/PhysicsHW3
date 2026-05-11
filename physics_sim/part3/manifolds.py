from pyglm import glm

from ..common import Vec3, vec3
from .cache import face_world_center
from .clipping import (
    build_reference_side_planes,
    clip_polygon_against_plane,
    load_face_polygon_world,
    point_inside_reference_side_planes,
    select_up_to_4_extreme_points,
)
from .constants import (
    BOX_BOX_FACE_FALLBACK_SLOP,
    BOX_EDGES,
    CONTACT_ID_KIND_EDGE,
    CONTACT_ID_KIND_FACE,
    CONTACT_ID_KIND_PLANE,
    MANIFOLD_KEEP_SLOP,
    MAX_BOX_VERTS,
    PLANE_CORNER_KEEP_SLOP,
    PLANE_FACE_KEEP_SLOP,
    PLANE_FACE_PARALLEL_DOT,
    PLANE_BODY_INDEX,
)
from .sat import incident_face_index, sat_test_box_box, support_face_index_from_normal
from .types import ContactManifold
from .utils import clampf


def make_pair_key(a, b):
    return (a, b) if a < b else (b, a)


def pack_contact_id(kind, ref_face, inc_face, feature):
    return (kind & 0xF) << 16 | ((ref_face & 0xF) << 12) | ((inc_face & 0xF) << 8) | (feature & 0xFF)


def reset_manifold(m: ContactManifold, body_a=0, body_b=0, normal=vec3(0.0, 1.0, 0.0)):
    m.body_a = body_a
    m.body_b = body_b
    m.normal = Vec3(normal)
    m.point_count = 0
    m.pair_key = make_pair_key(body_a, body_b)
    m.axis_type = -1
    m.axis_i = -1
    m.axis_j = -1
    m.tangent1 = vec3(0.0, 0.0, 0.0)
    m.tangent2 = vec3(0.0, 0.0, 0.0)
    m.mu_s = 0.0
    m.mu_d = 0.0


def manifold_add_point(m: ContactManifold, world_a, world_b, separation, cp_id):
    if m.point_count >= len(m.points):
        return

    cp = m.points[m.point_count]
    cp.id = int(cp_id)
    cp.world_a = Vec3(world_a)
    cp.world_b = Vec3(world_b)
    cp.local_a = vec3(0.0, 0.0, 0.0)
    cp.local_b = vec3(0.0, 0.0, 0.0)
    cp.separation = float(separation)

    cp.lambda_n = 0.0
    cp.lambda_n_xpbd = 0.0
    cp.normal_mass = 0.0

    cp.tangent_mass_1 = 0.0
    cp.tangent_mass_2 = 0.0
    cp.lambda_t1 = 0.0
    cp.lambda_t2 = 0.0
    cp.friction_local_a = vec3(0.0, 0.0, 0.0)
    cp.friction_local_b = vec3(0.0, 0.0, 0.0)
    cp.friction_valid = False

    m.point_count += 1


def finalize_manifold_local_anchors(m: ContactManifold, bodies):
    a = bodies[m.body_a]
    RaT = glm.transpose(a.R())
    for i in range(m.point_count):
        cp = m.points[i]
        cp.local_a = RaT * (cp.world_a - a.state.x)

    if m.body_b >= 0:
        b = bodies[m.body_b]
        RbT = glm.transpose(b.R())
        for i in range(m.point_count):
            cp = m.points[i]
            cp.local_b = RbT * (cp.world_b - b.state.x)
    else:
        for i in range(m.point_count):
            cp = m.points[i]
            cp.local_b = Vec3(cp.world_b)


def refresh_manifold_world_points(m: ContactManifold, bodies):
    a = bodies[m.body_a]
    Ra = a.R()
    for i in range(m.point_count):
        cp = m.points[i]
        cp.world_a = a.state.x + Ra * cp.local_a

    if m.body_b >= 0:
        b = bodies[m.body_b]
        Rb = b.R()
        for i in range(m.point_count):
            cp = m.points[i]
            cp.world_b = b.state.x + Rb * cp.local_b
    else:
        for i in range(m.point_count):
            cp = m.points[i]
            cp.world_b = Vec3(cp.local_b)

    for i in range(m.point_count):
        cp = m.points[i]
        cp.separation = glm.dot(m.normal, cp.world_b - cp.world_a)


def transfer_manifold_warm_start(new_m: ContactManifold, old_m: ContactManifold):
    for i in range(new_m.point_count):
        new_cp = new_m.points[i]
        for j in range(old_m.point_count):
            old_cp = old_m.points[j]
            if old_cp.id == new_cp.id:
                new_cp.lambda_n = old_cp.lambda_n
                new_cp.lambda_n_xpbd = old_cp.lambda_n_xpbd

                new_cp.lambda_t1 = old_cp.lambda_t1
                new_cp.lambda_t2 = old_cp.lambda_t2
                new_cp.friction_local_a = Vec3(old_cp.friction_local_a)
                new_cp.friction_local_b = Vec3(old_cp.friction_local_b)
                new_cp.friction_valid = old_cp.friction_valid
                break


def _best_face_index(cache, world_normal, pick_max: bool):
    best_face = 0
    best_dot = -1.0e30 if pick_max else 1.0e30
    for face_idx in range(6):
        d = glm.dot(face_world_normal(cache, face_idx), world_normal)
        if pick_max:
            if d > best_dot:
                best_dot = d
                best_face = face_idx
        else:
            if d < best_dot:
                best_dot = d
                best_face = face_idx
    return best_face


from .cache import face_world_normal
from .constants import BOX_FACE_VERTS


def build_box_plane_manifold(body_idx, body, cache, plane_y, out_m: ContactManifold):
    reset_manifold(out_m, body_a=body_idx, body_b=PLANE_BODY_INDEX, normal=vec3(0.0, -1.0, 0.0))

    min_y = min(cache.corners[i].y for i in range(MAX_BOX_VERTS))
    if min_y - plane_y > MANIFOLD_KEEP_SLOP:
        return False

    up = vec3(0.0, 1.0, 0.0)
    best_face = _best_face_index(cache, up, False)
    best_dot = glm.dot(face_world_normal(cache, best_face), up)

    if best_dot <= -PLANE_FACE_PARALLEL_DOT:
        verts = BOX_FACE_VERTS[best_face]
        face_sep_min = 1.0e30
        face_sep_max = -1.0e30
        for vi in verts:
            sep = cache.corners[vi].y - plane_y
            face_sep_min = min(face_sep_min, sep)
            face_sep_max = max(face_sep_max, sep)
        if face_sep_min <= PLANE_FACE_KEEP_SLOP and face_sep_max <= PLANE_FACE_KEEP_SLOP:
            for vi in verts:
                pa = Vec3(cache.corners[vi])
                manifold_add_point(
                    out_m,
                    pa,
                    vec3(pa.x, plane_y, pa.z),
                    pa.y - plane_y,
                    pack_contact_id(CONTACT_ID_KIND_PLANE, best_face, 0, vi & 0xFF),
                )
            return out_m.point_count > 0

    candidate_points = []
    candidate_ids = []
    y_keep = min_y + PLANE_CORNER_KEEP_SLOP
    for i in range(MAX_BOX_VERTS):
        p = cache.corners[i]
        if p.y <= y_keep:
            candidate_points.append(Vec3(p))
            candidate_ids.append(i)
    if not candidate_points:
        return False

    keep_ids = [0, 0, 0, 0]
    keep_count = select_up_to_4_extreme_points(candidate_points, len(candidate_points), out_m.normal, keep_ids)
    for k in range(keep_count):
        idx = keep_ids[k]
        pa = candidate_points[idx]
        manifold_add_point(
            out_m,
            pa,
            vec3(pa.x, plane_y, pa.z),
            pa.y - plane_y,
            pack_contact_id(CONTACT_ID_KIND_PLANE, 0, 0, candidate_ids[idx] & 0xFF),
        )
    return out_m.point_count > 0


def select_support_edge_segment(cache, edge_axis_idx: int, along_dir_world, choose_max=True):
    best_edge = -1
    best_val = -1.0e30
    sign = 1.0 if choose_max else -1.0

    for edge_idx, (v0, v1, axis_idx) in enumerate(BOX_EDGES):
        if axis_idx != edge_axis_idx:
            continue
        s = sign * glm.dot(0.5 * (cache.corners[v0] + cache.corners[v1]), along_dir_world)
        if s > best_val:
            best_val = s
            best_edge = edge_idx

    v0, v1, _ = BOX_EDGES[best_edge]
    return Vec3(cache.corners[v0]), Vec3(cache.corners[v1]), best_edge


def closest_points_segment_segment(p1, q1, p2, q2):
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = glm.dot(d1, d1)
    e = glm.dot(d2, d2)
    f = glm.dot(d2, r)

    if a <= 1.0e-8 and e <= 1.0e-8:
        return Vec3(p1), Vec3(p2)

    if a <= 1.0e-8:
        s = 0.0
        t = clampf(f / e if e > 1.0e-8 else 0.0, 0.0, 1.0)
    else:
        c = glm.dot(d1, r)
        if e <= 1.0e-8:
            t = 0.0
            s = clampf(-c / a, 0.0, 1.0)
        else:
            b = glm.dot(d1, d2)
            denom = a * e - b * b
            s = clampf((b * f - c * e) / denom, 0.0, 1.0) if abs(denom) > 1.0e-8 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = clampf(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = clampf((b - c) / a, 0.0, 1.0)

    return Vec3(p1 + d1 * s), Vec3(p2 + d2 * t)


def build_box_box_manifold(body_a_idx, body_a, cache_a,
                           body_b_idx, body_b, cache_b,
                           scratch,
                           out_m: ContactManifold):
    sat = sat_test_box_box(body_a, cache_a, body_b, cache_b)
    if not sat.hit:
        return False

    reset_manifold(out_m, body_a=body_a_idx, body_b=body_b_idx, normal=sat.normal)
    out_m.axis_type = sat.axis_type
    out_m.axis_i = sat.axis_i
    out_m.axis_j = sat.axis_j

    if sat.axis_type in (0, 1):
        ref_is_a = sat.axis_type == 0
        ref_cache, inc_cache = (cache_a, cache_b) if ref_is_a else (cache_b, cache_a)
        ref_body = body_a if ref_is_a else body_b
        ref_n = Vec3(sat.normal) if ref_is_a else -sat.normal
        ref_face = support_face_index_from_normal(ref_cache, ref_n)
        inc_face = incident_face_index(inc_cache, ref_n)
        inc_verts = BOX_FACE_VERTS[inc_face]

        side_normals, side_offsets = build_reference_side_planes(ref_cache, ref_face, ref_n)
        ref_plane_point = face_world_center(ref_body, ref_cache, ref_face)
        in_count = load_face_polygon_world(inc_cache, inc_face, scratch.clip_in, inc_face)

        for edge_i in range(4):
            in_count = clip_polygon_against_plane(
                scratch.clip_in, in_count,
                side_normals[edge_i], side_offsets[edge_i],
                scratch.clip_out,
            )
            scratch.clip_in, scratch.clip_out = scratch.clip_out, scratch.clip_in
            if in_count <= 0:
                break

        candidate_a = []
        candidate_b = []
        candidate_sep = []
        candidate_id = []

        def append_candidate(q, feature_id):
            sep_ref = glm.dot(ref_n, q - ref_plane_point)
            if sep_ref > MANIFOLD_KEEP_SLOP or not point_inside_reference_side_planes(q, side_normals, side_offsets):
                return
            if ref_is_a:
                pb = Vec3(q)
                pa = q - ref_n * sep_ref
            else:
                pa = Vec3(q)
                pb = q - ref_n * sep_ref
            candidate_a.append(Vec3(pa))
            candidate_b.append(Vec3(pb))
            candidate_sep.append(float(glm.dot(sat.normal, pb - pa)))
            candidate_id.append(pack_contact_id(CONTACT_ID_KIND_FACE, ref_face, inc_face, feature_id & 0xFF))

        for i in range(in_count):
            append_candidate(scratch.clip_in[i].p, scratch.clip_in[i].feature)

        if not candidate_a:
            for idx in inc_verts:
                append_candidate(inc_cache.corners[idx], idx)

        if not candidate_a:
            best_idx = inc_verts[0]
            best_q = Vec3(inc_cache.corners[best_idx])
            best_sep = glm.dot(ref_n, best_q - ref_plane_point)
            for idx in inc_verts[1:]:
                q = inc_cache.corners[idx]
                sep_ref = glm.dot(ref_n, q - ref_plane_point)
                if sep_ref < best_sep:
                    best_sep = sep_ref
                    best_q = Vec3(q)
                    best_idx = idx
            if best_sep <= BOX_BOX_FACE_FALLBACK_SLOP:
                append_candidate(best_q, best_idx)

        if not candidate_a:
            return False

        mids = [0.5 * (candidate_a[i] + candidate_b[i]) for i in range(len(candidate_a))]
        keep_ids = [0, 0, 0, 0]
        keep_count = select_up_to_4_extreme_points(mids, len(mids), sat.normal, keep_ids)
        for k in range(keep_count):
            i = keep_ids[k]
            manifold_add_point(out_m, candidate_a[i], candidate_b[i], candidate_sep[i], candidate_id[i])
        return out_m.point_count > 0

    n = Vec3(sat.normal)
    ea0, ea1, edge_a = select_support_edge_segment(cache_a, sat.axis_i, n, True)
    eb0, eb1, edge_b = select_support_edge_segment(cache_b, sat.axis_j, n, False)
    pa, pb = closest_points_segment_segment(ea0, ea1, eb0, eb1)
    manifold_add_point(
        out_m,
        pa,
        pb,
        glm.dot(n, pb - pa),
        pack_contact_id(
            CONTACT_ID_KIND_EDGE,
            sat.axis_i & 0xF,
            sat.axis_j & 0xF,
            ((edge_a & 0xF) << 4) | (edge_b & 0xF),
        ),
    )
    return True


def build_contact_manifolds(
    bodies,
    body_caches,
    broadphase,
    pair_buffer,
    scratch,
    prev_manifolds: dict,
    out_manifolds: list,
    plane_y=0.0,
):
    out_manifolds.clear()

    def finish_manifold(m):
        finalize_manifold_local_anchors(m, bodies)
        old = prev_manifolds.get(m.pair_key)
        if old is not None:
            transfer_manifold_warm_start(m, old)
        out_manifolds.append(m)

    for i, body in enumerate(bodies):
        m = ContactManifold()
        if build_box_plane_manifold(i, body, body_caches[i], plane_y, m):
            finish_manifold(m)

    if hasattr(broadphase, "fill_pairs_fixed"):
        pair_count = broadphase.fill_pairs_fixed(bodies, body_caches, pair_buffer)
        for i in range(pair_count):
            pair = pair_buffer[i]
            a = pair.body_a
            b = pair.body_b
            m = ContactManifold()
            if build_box_box_manifold(
                a, bodies[a], body_caches[a],
                b, bodies[b], body_caches[b],
                scratch, m
            ):
                finish_manifold(m)
        return pair_count

    broadphase.fill_pairs(bodies, body_caches, pair_buffer)

    for pair in pair_buffer:
        a = pair.body_a
        b = pair.body_b
        m = ContactManifold()
        if build_box_box_manifold(
            a, bodies[a], body_caches[a],
            b, bodies[b], body_caches[b],
            scratch, m
        ):
            finish_manifold(m)

    return len(pair_buffer)


def rebuild_manifold_map(manifolds):
    return {m.pair_key: m for m in manifolds}