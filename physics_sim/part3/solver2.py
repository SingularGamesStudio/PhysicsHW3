from pyglm import glm

from ..common import BoxShape, Quat, quat_identity, RigidBody, RigidBodyState, as_vec3, vec3
from .utils import vec3_zero
from .broadphase import SpatialGridBroadPhase
from .solver import CASE_LABELS, Part3Case, Part3Solver


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
    return RigidBody(
        shape=shape,
        mass=float(mass),
        state=RigidBodyState(
            x=as_vec3(pos),
            q=Quat(quat_identity() if q is None else q),
            v=as_vec3(vec3_zero() if v is None else v),
            w_body=as_vec3(vec3_zero() if w_body is None else w_body),
        ),
    )


def make_part3_many_box_bodies(
    shape=None,
    mass=1.0,
    plane_y=0.0,
    nx=10,
    ny=10,
    nz=10,
    gap=0.02,
    drop_height=1.25,
    tilt_scale=0.030,
    jitter_frac=0.08,
):
    """
    About 1000 bodies:
      - default is 10 x 10 x 10 = 1000 boxes
      - all boxes same size
      - tiny deterministic jitter/tilt to avoid a perfectly symmetric stack
    """
    if shape is None:
        shape = BoxShape(half_extents=vec3(0.50, 0.35, 0.32))

    hx, hy, hz = shape.half_extents.x, shape.half_extents.y, shape.half_extents.z
    sx = 2.0 * hx + gap
    sy = 2.0 * hy + gap
    sz = 2.0 * hz + gap

    x0 = -0.5 * (nx - 1) * sx
    z0 = -0.5 * (nz - 1) * sz
    y0 = plane_y + hy + drop_height

    pos_jitter_x = jitter_frac * gap
    pos_jitter_z = jitter_frac * gap

    bodies = []
    idx = 0

    for iy in range(ny):
        layer_y = y0 + iy * sy

        # Slight brick-like layer shift improves the contact graph.
        layer_shift_x = 0.5 * sx if (iy & 1) else 0.0
        layer_shift_z = 0.5 * sz if ((iy >> 1) & 1) else 0.0

        for iz in range(nz):
            for ix in range(nx):
                px = x0 + ix * sx + layer_shift_x + _hash_signed(idx * 3 + 0) * pos_jitter_x
                py = layer_y
                pz = z0 + iz * sz + layer_shift_z + _hash_signed(idx * 3 + 1) * pos_jitter_z

                q = _tilt_quat(
                    tilt_scale * _hash_signed(idx * 3 + 2),
                    tilt_scale * _hash_signed(idx * 3 + 3),
                    tilt_scale * _hash_signed(idx * 3 + 4),
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


def make_many_box_grid_broadphase(bodies, aabb_margin=0.02):
    if not bodies:
        raise ValueError("Need at least one body")

    # Conservative "same-size box" cell size:
    # use bounding sphere diameter + margin so each box stays in a small fixed
    # number of cells even while rotating.
    he = bodies[0].shape.half_extents
    radius = glm.length(he)
    cell_size = 2.0 * float(radius) + 2.0 * float(aabb_margin)

    n = len(bodies)
    return SpatialGridBroadPhase(
        body_capacity=n,
        cell_size=cell_size,
        bucket_count=max(1024, n * 8),
        max_cells_per_body=8,
        max_pairs=max(2048, n * 64),
    )


def make_part3_many_box_solvers(
    dt=1.0 / 120.0,
    plane_y=0.0,
    shape=None,
    mass=1.0,
    nx=10,
    ny=10,
    nz=10,
    gap=0.02,
    drop_height=1.25,
    aabb_margin=0.02,
    do_SI = True,
):
    bodies = make_part3_many_box_bodies(
        shape=shape,
        mass=mass,
        plane_y=plane_y,
        nx=nx,
        ny=ny,
        nz=nz,
        gap=gap,
        drop_height=drop_height,
    )

    # Separate broadphase instances because this implementation is stateful.
    broadphase_si = make_many_box_grid_broadphase(bodies, aabb_margin=aabb_margin)
    broadphase_xpbd = make_many_box_grid_broadphase(bodies, aabb_margin=aabb_margin)
    if do_SI:
        return {
            Part3Case.BOX_PILE_SI_NGS: Part3Solver(
                bodies=bodies,
                dt=dt,
                case=Part3Case.BOX_PILE_SI_NGS,
                gravity=vec3(0.0, -9.81, 0.0),
                plane_y=plane_y,
                broadphase=broadphase_si,
                vel_iters=5,
                pos_iters=5,
                xpbd_iters=0,
                newton_iters=10,
                ngs_beta=0.45,
                ngs_slop=1.0e-4,
                xpbd_compliance=0.0,
                xpbd_slop=0.0,
                aabb_margin=aabb_margin,
            ),
            Part3Case.BOX_PILE_XPBD: Part3Solver(
                bodies=bodies,
                dt=dt,
                case=Part3Case.BOX_PILE_XPBD,
                gravity=vec3(0.0, -9.81, 0.0),
                plane_y=plane_y,
                broadphase=broadphase_xpbd,
                vel_iters=0,
                pos_iters=0,
                xpbd_iters=10,
                xpbd_post_iters=4,
                newton_iters=10,
                ngs_beta=0.0,
                ngs_slop=0.0,
                xpbd_compliance=0.001,
                xpbd_slop=0.01,
                aabb_margin=aabb_margin,
            ),
        }
    else:
        return {
            Part3Case.BOX_PILE_XPBD: Part3Solver(
                bodies=bodies,
                dt=dt,
                case=Part3Case.BOX_PILE_XPBD,
                gravity=vec3(0.0, -9.81, 0.0),
                plane_y=plane_y,
                broadphase=broadphase_xpbd,
                vel_iters=0,
                pos_iters=0,
                xpbd_iters=10,
                xpbd_post_iters=4,
                newton_iters=10,
                ngs_beta=0.0,
                ngs_slop=0.0,
                xpbd_compliance=0.001,
                xpbd_slop=0.01,
                aabb_margin=aabb_margin,
            ),
        }


def simulate_part3_many_boxes(
    dt=1.0 / 120.0,
    steps=900,
    plane_y=0.0,
    shape=None,
    mass=1.0,
    nx=10,
    ny=10,
    nz=10,
    gap=0.02,
    drop_height=1.25,
    aabb_margin=0.02,
    do_SI = True,
):
    solvers = make_part3_many_box_solvers(
        dt=dt,
        plane_y=plane_y,
        shape=shape,
        mass=mass,
        nx=nx,
        ny=ny,
        nz=nz,
        gap=gap,
        drop_height=drop_height,
        aabb_margin=aabb_margin,
        do_SI=do_SI,
    )

    results = {}
    for case, solver in solvers.items():
        samples = [solver.snapshot(0.0)]
        for i in range(steps):
            solver.step()
            samples.append(solver.snapshot((i + 1) * dt))

        results[case.name] = {
            "label": CASE_LABELS[case],
            "dt": float(dt),
            "samples": samples,
        }

    return results