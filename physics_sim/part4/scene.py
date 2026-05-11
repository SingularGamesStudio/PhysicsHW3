from pyglm import glm

from ..common import BoxShape, Quat, quat_identity, RigidBody, RigidBodyState, as_vec3, vec3
from ..part3.utils import vec3_zero
from .solver import CASE_LABELS_PART4, Part4Case, Part4Solver
from .broadphase import SweepAndPruneBroadPhase, LBVHBroadPhase


def _u32_hash(x: int) -> int:
    x &= 0xFFFFFFFF
    x ^= (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15)
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= (x >> 16)
    return x & 0xFFFFFFFF


def _hash01(seed: int) -> float:
    return _u32_hash(seed) * (1.0 / 4294967295.0)


def _hash_signed(seed: int) -> float:
    return 2.0 * _hash01(seed) - 1.0


def _tilt_quat(rx=0.0, ry=0.0, rz=0.0):
    qx = glm.angleAxis(float(rx), vec3(1.0, 0.0, 0.0))
    qy = glm.angleAxis(float(ry), vec3(0.0, 1.0, 0.0))
    qz = glm.angleAxis(float(rz), vec3(0.0, 0.0, 1.0))
    return glm.normalize(qz * qy * qx)


def _make_box_body(shape, mass, pos, q=None, v=None, w_body=None):
    body = RigidBody(
        shape=shape,
        mass=float(mass),
        state=RigidBodyState(
            x=as_vec3(pos),
            q=Quat(quat_identity() if q is None else q),
            v=as_vec3(vec3_zero() if v is None else v),
            w_body=as_vec3(vec3_zero() if w_body is None else w_body),
        ),
    )
    return body


def make_part4_mixed_box_bodies(
    plane_y=0.0,
    count_x=12,
    count_y=10,
    count_z=8,
    gap=0.03,
    drop_height=1.0,
    min_he=vec3(0.18, 0.12, 0.14),
    max_he=vec3(0.95, 0.70, 0.85),
    density=1.0,
    tilt_scale=0.05,
):
    bodies = []
    idx = 0

    pitch_x = 2.0 * max_he.x + gap
    pitch_y = 2.0 * max_he.y + gap
    pitch_z = 2.0 * max_he.z + gap

    x0 = -0.5 * (count_x - 1) * pitch_x
    z0 = -0.5 * (count_z - 1) * pitch_z
    y0 = plane_y + max_he.y + drop_height

    for iy in range(count_y):
        layer_y = y0 + iy * pitch_y
        shift_x = 0.35 * pitch_x if (iy & 1) else 0.0
        shift_z = 0.25 * pitch_z if ((iy >> 1) & 1) else 0.0

        for iz in range(count_z):
            for ix in range(count_x):
                tx = _hash01(idx * 17 + 1)
                ty = _hash01(idx * 17 + 2)
                tz = _hash01(idx * 17 + 3)

                he = vec3(
                    min_he.x + (max_he.x - min_he.x) * tx,
                    min_he.y + (max_he.y - min_he.y) * ty,
                    min_he.z + (max_he.z - min_he.z) * tz,
                )
                shape = BoxShape(half_extents=he)

                volume = 8.0 * he.x * he.y * he.z
                mass = density * volume

                px = x0 + ix * pitch_x + shift_x + 0.15 * pitch_x * _hash_signed(idx * 17 + 4)
                py = layer_y + 0.10 * pitch_y * _hash_signed(idx * 17 + 5)
                pz = z0 + iz * pitch_z + shift_z + 0.15 * pitch_z * _hash_signed(idx * 17 + 6)

                q = _tilt_quat(
                    tilt_scale * _hash_signed(idx * 17 + 7),
                    tilt_scale * _hash_signed(idx * 17 + 8),
                    tilt_scale * _hash_signed(idx * 17 + 9),
                )

                bodies.append(
                    _make_box_body(
                        shape=shape,
                        mass=mass,
                        pos=vec3(px, py, pz),
                        q=q,
                    )
                )
                idx += 1

    return bodies


from .broadphase import SweepAndPruneBroadPhase, LBVHBroadPhase


def make_part4_sap_broadphase(bodies, aabb_margin=0.02):
    n = len(bodies)
    return SweepAndPruneBroadPhase(
        body_capacity=n,
        axis=-1,  # auto-pick longest scene axis each frame
        max_pairs=max(2048, n * 64),
    )


def make_part4_lbvh_broadphase(bodies, aabb_margin=0.02):
    n = len(bodies)
    return LBVHBroadPhase(
        body_capacity=n,
        max_pairs=max(2048, n * 64),
        traversal_work_factor=8,
    )


def make_part4_solvers(
    dt=1.0 / 120.0,
    plane_y=0.0,
    count_x=12,
    count_y=10,
    count_z=8,
    gap=0.03,
    drop_height=1.0,
    aabb_margin=0.02,
):
    bodies = make_part4_mixed_box_bodies(
        plane_y=plane_y,
        count_x=count_x,
        count_y=count_y,
        count_z=count_z,
        gap=gap,
        drop_height=drop_height,
    )

    broadphase_sap = make_part4_sap_broadphase(bodies, aabb_margin=aabb_margin)
    broadphase_lbvh = make_part4_lbvh_broadphase(bodies, aabb_margin=aabb_margin)

    return {
        Part4Case.MIXED_BOXES_XPBD_SAP: Part4Solver(
            bodies=bodies,
            dt=dt,
            case=Part4Case.MIXED_BOXES_XPBD_SAP,
            gravity=vec3(0.0, -9.81, 0.0),
            plane_y=plane_y,
            broadphase=broadphase_sap,
            xpbd_iters=14,
            xpbd_friction_iters=12,
            xpbd_post_iters=2,
            newton_iters=10,
            normal_compliance=0.0,
            tangent_compliance=0.0,
            slop=1.0e-4,
            aabb_margin=aabb_margin,
        ),
        Part4Case.MIXED_BOXES_XPBD_LBVH: Part4Solver(
            bodies=bodies,
            dt=dt,
            case=Part4Case.MIXED_BOXES_XPBD_LBVH,
            gravity=vec3(0.0, -9.81, 0.0),
            plane_y=plane_y,
            broadphase=broadphase_lbvh,
            xpbd_iters=14,
            xpbd_friction_iters=12,
            xpbd_post_iters=2,
            newton_iters=10,
            normal_compliance=0.0,
            tangent_compliance=0.0,
            slop=1.0e-4,
            aabb_margin=aabb_margin,
        ),
    }


def simulate_part4(
    dt=1.0 / 120.0,
    steps=900,
    plane_y=0.0,
    count_x=12,
    count_y=10,
    count_z=8,
    gap=0.03,
    drop_height=1.0,
    aabb_margin=0.02,
):
    solvers = make_part4_solvers(
        dt=dt,
        plane_y=plane_y,
        count_x=count_x,
        count_y=count_y,
        count_z=count_z,
        gap=gap,
        drop_height=drop_height,
        aabb_margin=aabb_margin,
    )

    results = {}
    for case, solver in solvers.items():
        samples = [solver.snapshot(0.0)]
        for i in range(steps):
            solver.step()
            samples.append(solver.snapshot((i + 1) * dt))

        results[case.name] = {
            "label": CASE_LABELS_PART4[case],
            "dt": float(dt),
            "samples": samples,
        }

    return results