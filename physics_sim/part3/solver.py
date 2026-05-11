import time
from enum import Enum, auto

from ..common import Quat, Vec3, as_vec3, q_to_list, v_to_list, vec3
from ..part2.integration import integrate_body_pose_symplectic, integrate_body_velocities_implicit_gyro
from ..part2.physics import apply_world_impulse, body_point_velocity_world, total_kinetic_energy
from ..part2.types import GyroScratch, ImpulseScratch, MassScratch, PositionImpulseScratch
from .cache import update_body_collision_cache
from .constants import EPS
from .contacts import (
    prepare_contact_masses,
    solve_contact_position_ngs,
    solve_contact_position_xpbd,
    solve_contact_velocity_si,
    warm_start_contact_impulses,
)
from .manifolds import build_contact_manifolds, rebuild_manifold_map
from .stats import SolveRateTracker, StepStats
from .types import BodyCollisionCache, CollisionScratch, CandidatePair
from .utils import count_total_contacts, reconstruct_velocities_from_pose_delta
from pyglm import glm


class Part3Case(Enum):
    BOX_PILE_XPBD = auto()
    BOX_PILE_SI_NGS = auto()


CASE_LABELS = {
    Part3Case.BOX_PILE_XPBD: "XPBD",
    Part3Case.BOX_PILE_SI_NGS: "SI+NGS",
}

class Part3Solver:
    def __init__(
        self,
        bodies,
        dt,
        case,
        gravity=vec3(0.0, -9.81, 0.0),
        plane_y=0.0,
        broadphase=None,
        vel_iters=12,
        pos_iters=10,
        xpbd_iters=12,
        xpbd_post_iters=4,
        newton_iters=10,
        ngs_beta=0.45,
        ngs_slop=1.0e-4,
        xpbd_compliance=0.0,
        xpbd_slop=1.0e-4,
        xpbd_max_push_speed=2.0,
        aabb_margin=0.02,
    ):
        self.bodies = [b.clone() for b in bodies]
        self.dt = float(dt)
        self.case = case
        self.gravity = as_vec3(gravity)
        self.plane_y = float(plane_y)

        self.vel_iters = int(vel_iters)
        self.pos_iters = int(pos_iters)
        self.xpbd_iters = int(xpbd_iters)
        self.xpbd_post_iters = int(xpbd_post_iters)
        self.newton_iters = int(newton_iters)

        self.ngs_beta = float(ngs_beta)
        self.ngs_slop = float(ngs_slop)

        self.xpbd_compliance = float(xpbd_compliance)
        self.xpbd_slop = float(xpbd_slop)
        self.xpbd_max_push_speed = float(xpbd_max_push_speed)

        self.aabb_margin = float(aabb_margin)

        if broadphase is None:
            from .broadphase import BruteForceBroadPhaseFixed
            broadphase = BruteForceBroadPhaseFixed()
        self.broadphase = broadphase

        n = len(self.bodies)

        self._body_caches = [BodyCollisionCache() for _ in range(n)]
        self._gyro = [GyroScratch() for _ in range(n)]
        self._prev_x = [Vec3(0.0, 0.0, 0.0) for _ in range(n)]
        self._prev_q = [Quat(1.0, 0.0, 0.0, 0.0) for _ in range(n)]

        self._impulse_scratch = ImpulseScratch()
        self._pos_impulse_scratch = PositionImpulseScratch()
        self._mass_scratch_a = MassScratch()
        self._mass_scratch_b = MassScratch()
        self._collision_scratch = CollisionScratch()

        self._manifolds = []
        self._curr_manifold_count = 0
        self._prev_manifold_map = {}

        self._stats = StepStats()
        self._hz_tracker = SolveRateTracker()
        self._candidate_pair_count = 0

        if hasattr(self.broadphase, "pair_capacity_hint"):
            pair_capacity = int(self.broadphase.pair_capacity_hint(n))
        else:
            pair_capacity = max(16, n * 64)

        self._pair_buffer = [CandidatePair(0, 0) for _ in range(pair_capacity)]

        self._update_body_caches()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def step(self):
        if self.case == Part3Case.BOX_PILE_SI_NGS:
            self._step_si_ngs()
        elif self.case == Part3Case.BOX_PILE_XPBD:
            self._step_xpbd()
        else:
            raise ValueError(f"Unknown Part 3 case: {self.case}")

    def snapshot(self, t):
        # hz is kept as a compatibility alias to tps.
        return {
            "t": float(t),
            "bodies": [
                {
                    "x": v_to_list(b.state.x),
                    "q": q_to_list(b.state.q),
                    "v": v_to_list(b.state.v),
                    "w_body": v_to_list(b.state.w_body),
                }
                for b in self.bodies
            ],
            "E": float(total_kinetic_energy(self.bodies)),
            "contacts": int(self._count_contacts_current()),
            "manifolds": int(self._curr_manifold_count),
            "candidate_pairs": int(self._candidate_pair_count),
            "tps": float(self._stats.tps),
            "hz": float(self._stats.tps),
            "sim_hz": float(self._stats.sim_hz),
            "step_ms": float(self._stats.dt_seconds * 1000.0),
        }

    # ---------------------------------------------------------------------
    # Internal bookkeeping
    # ---------------------------------------------------------------------

    def _update_body_caches(self):
        for i, body in enumerate(self.bodies):
            update_body_collision_cache(body, self._body_caches[i], aabb_margin=self.aabb_margin)

    def _build_manifolds_current_pose(self):
        self._candidate_pair_count = build_contact_manifolds(
            self.bodies,
            self._body_caches,
            self.broadphase,
            self._pair_buffer,
            self._collision_scratch,
            self._prev_manifold_map,
            self._manifolds,
            plane_y=self.plane_y,
        )
        self._curr_manifold_count = len(self._manifolds)

    def _finish_manifold_build(self):
        self._prev_manifold_map = rebuild_manifold_map(self._manifolds)

    def _count_contacts_current(self):
        return count_total_contacts(self._manifolds)

    def _measure_step_end(self, start_t):
        dt_wall = max(1.0e-9, time.perf_counter() - start_t)
        self._stats.dt_seconds = dt_wall
        self._stats.tps = self._hz_tracker.push_dt(dt_wall)
        self._stats.sim_hz = 1.0 / self.dt
        self._stats.manifold_count = int(self._curr_manifold_count)
        self._stats.contact_count = int(self._count_contacts_current())
        self._stats.candidate_pair_count = int(self._candidate_pair_count)

    # ---------------------------------------------------------------------
    # Contact prep / solve helpers
    # ---------------------------------------------------------------------

    def _prepare_contact_masses_current(self):
        prepare_contact_masses(self._manifolds, self.bodies, self._mass_scratch_a, self._mass_scratch_b)

    def _warm_start_current_contacts(self):
        warm_start_contact_impulses(self._manifolds, self.bodies, self._impulse_scratch)

    def _solve_velocity_contacts_si_iteration(self):
        solve_contact_velocity_si(self._manifolds, self.bodies, self._impulse_scratch)

    def _solve_position_contacts_ngS_iteration(self):
        solve_contact_position_ngs(
            self._manifolds,
            self.bodies,
            self._pos_impulse_scratch,
            self._mass_scratch_a,
            self._mass_scratch_b,
            beta=self.ngs_beta,
            slop=self.ngs_slop,
        )

    def _solve_position_contacts_xpbd_iteration(self):
        solve_contact_position_xpbd(
            self._manifolds,
            self.bodies,
            self._pos_impulse_scratch,
            self._mass_scratch_a,
            self._mass_scratch_b,
            dt=self.dt,
            compliance=self.xpbd_compliance,
            slop=self.xpbd_slop,
        )

    def _relax_velocity_contacts_iteration(self):
        for m in self._manifolds:
            n = m.normal
            a = self.bodies[m.body_a]

            if m.body_b >= 0:
                b = self.bodies[m.body_b]
                for i in range(m.point_count):
                    cp = m.points[i]
                    if cp.normal_mass <= 0.0:
                        continue
                    if cp.lambda_n_xpbd <= 0.0 and cp.separation > 1.0e-3:
                        continue

                    va = body_point_velocity_world(a, cp.local_a)
                    vb = body_point_velocity_world(b, cp.local_b)
                    vn = glm.dot(vb - va, n)

                    d_lambda = -cp.normal_mass * vn
                    if abs(d_lambda) <= EPS:
                        continue

                    P = n * d_lambda
                    apply_world_impulse(a, -P, cp.world_a, self._impulse_scratch)
                    apply_world_impulse(b, +P, cp.world_b, self._impulse_scratch)
            else:
                for i in range(m.point_count):
                    cp = m.points[i]
                    if cp.normal_mass <= 0.0:
                        continue
                    if cp.lambda_n_xpbd <= 0.0 and cp.separation > 1.0e-3:
                        continue

                    va = body_point_velocity_world(a, cp.local_a)
                    vn = glm.dot(-va, n)

                    d_lambda = -cp.normal_mass * vn
                    if abs(d_lambda) <= EPS:
                        continue

                    P = n * d_lambda
                    apply_world_impulse(a, -P, cp.world_a, self._impulse_scratch)

    # ---------------------------------------------------------------------
    # Actual step implementations
    # ---------------------------------------------------------------------

    def _step_si_ngs(self):
        start_t = self._hz_tracker.begin()

        # External forces -> velocities
        for i, body in enumerate(self.bodies):
            integrate_body_velocities_implicit_gyro(
                body,
                self.dt,
                body.mass * self.gravity,
                Vec3(0.0, 0.0, 0.0),
                self._gyro[i],
                newton_iters=self.newton_iters,
            )

        # Current-pose contacts
        self._update_body_caches()
        self._build_manifolds_current_pose()
        self._prepare_contact_masses_current()
        self._warm_start_current_contacts()

        for _ in range(self.vel_iters):
            self._solve_velocity_contacts_si_iteration()

        # Symplectic pose update
        for body in self.bodies:
            integrate_body_pose_symplectic(body, self.dt)

        # Position solve
        self._update_body_caches()
        for _ in range(self.pos_iters):
            self._solve_position_contacts_ngS_iteration()

        self._update_body_caches()
        self._measure_step_end(start_t)
        self._finish_manifold_build()

    def _step_xpbd(self):
        start_t = self._hz_tracker.begin()

        for i, body in enumerate(self.bodies):
            self._prev_x[i] = Vec3(body.state.x)
            self._prev_q[i] = Quat(body.state.q)

        # External forces -> velocities
        for i, body in enumerate(self.bodies):
            integrate_body_velocities_implicit_gyro(
                body,
                self.dt,
                body.mass * self.gravity,
                Vec3(0.0, 0.0, 0.0),
                self._gyro[i],
                newton_iters=self.newton_iters,
            )

        # Predict pose
        for body in self.bodies:
            integrate_body_pose_symplectic(body, self.dt)

        # Contacts on predicted pose
        self._update_body_caches()
        self._build_manifolds_current_pose()

        # XPBD positional solve
        for _ in range(self.xpbd_iters):
            self._solve_position_contacts_xpbd_iteration()

        # Reconstruct velocities from corrected pose
        for i, body in enumerate(self.bodies):
            reconstruct_velocities_from_pose_delta(body, self._prev_x[i], self._prev_q[i], self.dt)

        # Relax solver-injected velocity. This is the main anti-launch fix.
        self._update_body_caches()
        self._prepare_contact_masses_current()
        for _ in range(self.xpbd_post_iters):
            self._relax_velocity_contacts_iteration()

        self._update_body_caches()
        self._measure_step_end(start_t)
        self._finish_manifold_build()


def make_part3_solvers(bodies, dt, plane_y=0.0, broadphase=None):
    from .broadphase import BruteForceBroadPhaseFixed

    if broadphase is None:
        broadphase = BruteForceBroadPhaseFixed()

    return {
        Part3Case.BOX_PILE_SI_NGS: Part3Solver(
            bodies=bodies,
            dt=dt,
            case=Part3Case.BOX_PILE_SI_NGS,
            gravity=vec3(0.0, -9.81, 0.0),
            plane_y=plane_y,
            broadphase=broadphase,
            vel_iters=5,
            pos_iters=5,
            xpbd_iters=0,
            newton_iters=10,
            ngs_beta=0.45,
            ngs_slop=1.0e-4,
            xpbd_compliance=0.0,
            xpbd_slop=0.0,
            aabb_margin=0.02,
        ),
        Part3Case.BOX_PILE_XPBD: Part3Solver(
            bodies=bodies,
            dt=dt,
            case=Part3Case.BOX_PILE_XPBD,
            gravity=vec3(0.0, -9.81, 0.0),
            plane_y=plane_y,
            broadphase=broadphase,
            vel_iters=0,
            pos_iters=0,
            xpbd_iters=10,
            newton_iters=10,
            ngs_beta=0.0,
            ngs_slop=0.0,
            xpbd_compliance=0.001,
            xpbd_slop=0.01,
            aabb_margin=0.02,
        ),
    }


def simulate_part3(bodies, dt=1.0 / 120.0, steps=900, plane_y=0.0, broadphase=None):
    solvers = make_part3_solvers(bodies, dt=dt, plane_y=plane_y, broadphase=broadphase)
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
