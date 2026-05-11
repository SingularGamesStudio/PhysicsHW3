import time
from enum import Enum, auto

from pyglm import glm

from ..common import Quat, Vec3, as_vec3, q_to_list, v_to_list, vec3
from ..part2.integration import integrate_body_pose_symplectic, integrate_body_velocities_implicit_gyro
from ..part2.physics import apply_world_impulse, body_point_velocity_world, total_kinetic_energy
from ..part2.types import GyroScratch, ImpulseScratch
from ..part3.cache import update_body_collision_cache
from ..part3.constants import EPS
from ..part3.manifolds import build_contact_manifolds, rebuild_manifold_map
from ..part3.stats import SolveRateTracker, StepStats
from ..part3.types import BodyCollisionCache, CollisionScratch, CandidatePair


class Part4Case(Enum):
    MIXED_BOXES_XPBD_SAP = auto()
    MIXED_BOXES_XPBD_LBVH = auto()


CASE_LABELS_PART4 = {
    Part4Case.MIXED_BOXES_XPBD_SAP: "XPBD: mixed-size rigid bodies with Sweep-and-Prune broadphase",
    Part4Case.MIXED_BOXES_XPBD_LBVH: "XPBD: mixed-size rigid bodies with LBVH broadphase",
}


def count_total_contacts(manifolds):
    total = 0
    for m in manifolds:
        total += m.point_count
    return total


def reconstruct_velocities_from_pose_delta(body, prev_x, prev_q, dt):
    inv_dt = 1.0 / dt
    body.state.v = (body.state.x - prev_x) * inv_dt

    dq = body.state.q * glm.conjugate(prev_q)
    if dq.w < 0.0:
        dq = -dq

    axis = vec3(dq.x, dq.y, dq.z)
    axis_len = glm.length(axis)
    if axis_len <= 1.0e-12:
        w_world = vec3(0.0, 0.0, 0.0)
    else:
        angle = 2.0 * glm.atan(axis_len, dq.w)
        w_world = axis * (angle / axis_len * inv_dt)

    body.state.w_body = glm.transpose(body.R()) * w_world


class Part4Solver:
    def __init__(
        self,
        bodies,
        dt,
        case,
        gravity=vec3(0.0, -9.81, 0.0),
        plane_y=0.0,
        broadphase=None,
        xpbd_iters=14,
        xpbd_friction_iters=8,
        xpbd_post_iters=4,
        newton_iters=10,
        normal_compliance=0.0,
        tangent_compliance=0.0,
        slop=1.0e-4,
        max_push_speed=2.0,
        friction_static=2.0,
        friction_dynamic=1.5,
        aabb_margin=0.02,
    ):
        self.bodies = [b.clone() for b in bodies]
        self.dt = float(dt)
        self.case = case
        self.gravity = as_vec3(gravity)
        self.plane_y = float(plane_y)

        self.xpbd_iters = int(xpbd_iters)
        self.xpbd_friction_iters = int(xpbd_friction_iters)
        self.xpbd_post_iters = int(xpbd_post_iters)
        self.newton_iters = int(newton_iters)

        self.normal_compliance = float(normal_compliance)
        self.tangent_compliance = float(tangent_compliance)
        self.slop = float(slop)
        self.max_push_speed = float(max_push_speed)

        self.friction_static = float(friction_static)
        self.friction_dynamic = float(friction_dynamic)
        self.aabb_margin = float(aabb_margin)

        if broadphase is None:
            raise ValueError("Part4Solver expects an explicit broadphase instance")
        self.broadphase = broadphase

        n = len(self.bodies)

        self._body_caches = [BodyCollisionCache() for _ in range(n)]
        self._gyro = [GyroScratch() for _ in range(n)]
        self._prev_x = [Vec3(0.0, 0.0, 0.0) for _ in range(n)]
        self._prev_q = [Quat(1.0, 0.0, 0.0, 0.0) for _ in range(n)]

        self._impulse_scratch = ImpulseScratch()
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
            pair_capacity = max(256, n * 32)

        self._pair_buffer = [CandidatePair(0, 0) for _ in range(pair_capacity)]

        self._update_body_caches()

    def step(self):
        self._step_xpbd_friction()

    def snapshot(self, t):
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

    def _inv_mass(self, body):
        m = float(body.mass)
        return 0.0 if m <= 0.0 else 1.0 / m

    def _world_inv_inertia(self, body):
        return body.I_world_inv()

    def _refresh_contact_world(self, m, cp):
        a = self.bodies[m.body_a]
        cp.world_a = a.state.x + a.R() * cp.local_a

        if m.body_b >= 0:
            b = self.bodies[m.body_b]
            cp.world_b = b.state.x + b.R() * cp.local_b
        else:
            cp.world_b = Vec3(cp.local_b)

        cp.separation = glm.dot(m.normal, cp.world_b - cp.world_a)

    def _contact_basis(self, n):
        if abs(n.x) < 0.57735026919:
            t1 = glm.normalize(glm.cross(n, vec3(1.0, 0.0, 0.0)))
        else:
            t1 = glm.normalize(glm.cross(n, vec3(0.0, 1.0, 0.0)))
        t2 = glm.cross(n, t1)
        return t1, t2

    def _compute_effective_mass(self, a, b, ra, rb, axis):
        k = 0.0

        inv_mass_a = self._inv_mass(a)
        if inv_mass_a > 0.0:
            k += inv_mass_a
            wa = glm.cross(self._world_inv_inertia(a) * glm.cross(ra, axis), ra)
            k += glm.dot(wa, axis)

        if b is not None:
            inv_mass_b = self._inv_mass(b)
            if inv_mass_b > 0.0:
                k += inv_mass_b
                wb = glm.cross(self._world_inv_inertia(b) * glm.cross(rb, axis), rb)
                k += glm.dot(wb, axis)

        if k <= EPS:
            return 0.0
        return 1.0 / k

    def _apply_position_delta(self, body, dp_world, world_point):
        inv_mass = self._inv_mass(body)
        if inv_mass <= 0.0:
            return

        x0 = Vec3(body.state.x)
        r = world_point - x0

        body.state.x = x0 + dp_world * inv_mass

        ang = self._world_inv_inertia(body) * glm.cross(r, dp_world)
        if glm.length2(ang) > EPS:
            q = body.state.q
            dq = Quat(0.0, ang.x, ang.y, ang.z) * q
            body.set_q(glm.normalize(q + 0.5 * dq))

    def _reset_friction_anchor_to_current_contact(self, m, cp):
        a = self.bodies[m.body_a]
        cp.friction_local_a = glm.transpose(a.R()) * (cp.world_a - a.state.x)

        if m.body_b >= 0:
            b = self.bodies[m.body_b]
            cp.friction_local_b = glm.transpose(b.R()) * (cp.world_b - b.state.x)
        else:
            cp.friction_local_b = Vec3(cp.world_b)

        cp.friction_valid = True

    def _refresh_friction_world(self, m, cp):
        a = self.bodies[m.body_a]
        fa = a.state.x + a.R() * cp.friction_local_a

        if m.body_b >= 0:
            b = self.bodies[m.body_b]
            fb = b.state.x + b.R() * cp.friction_local_b
        else:
            fb = Vec3(cp.friction_local_b)

        return fa, fb

    def _prepare_contact_state_current(self):
        for m in self._manifolds:
            t1, t2 = self._contact_basis(m.normal)
            m.tangent1 = t1
            m.tangent2 = t2
            m.mu_s = self.friction_static
            m.mu_d = self.friction_dynamic

            a = self.bodies[m.body_a]
            b = self.bodies[m.body_b] if m.body_b >= 0 else None

            for i in range(m.point_count):
                cp = m.points[i]
                self._refresh_contact_world(m, cp)

                if not cp.friction_valid:
                    self._reset_friction_anchor_to_current_contact(m, cp)
                    cp.lambda_t1 = 0.0
                    cp.lambda_t2 = 0.0

                ra = cp.world_a - a.state.x
                rb = (cp.world_b - b.state.x) if b is not None else vec3(0.0, 0.0, 0.0)

                cp.normal_mass = self._compute_effective_mass(a, b, ra, rb, m.normal)
                cp.tangent_mass_1 = self._compute_effective_mass(a, b, ra, rb, t1)
                cp.tangent_mass_2 = self._compute_effective_mass(a, b, ra, rb, t2)

                if cp.lambda_n < 0.0:
                    cp.lambda_n = 0.0

    def _solve_position_contacts_xpbd_iteration(self):
        alpha_n = self.normal_compliance / (self.dt * self.dt)
        max_push = self.max_push_speed * self.dt

        for m in self._manifolds:
            n = m.normal
            a = self.bodies[m.body_a]
            b = self.bodies[m.body_b] if m.body_b >= 0 else None

            for i in range(m.point_count):
                cp = m.points[i]
                self._refresh_contact_world(m, cp)

                ra = cp.world_a - a.state.x
                rb = (cp.world_b - b.state.x) if b is not None else vec3(0.0, 0.0, 0.0)

                cp.normal_mass = self._compute_effective_mass(a, b, ra, rb, n)
                if cp.normal_mass <= 0.0:
                    continue

                c = min(0.0, cp.separation + self.slop)
                if c >= 0.0:
                    continue

                old_lambda = cp.lambda_n_xpbd
                d_lambda = (-c - alpha_n * old_lambda) / (1.0 / cp.normal_mass + alpha_n)
                new_lambda = max(0.0, old_lambda + d_lambda)
                d_lambda = new_lambda - old_lambda
                if abs(d_lambda) <= EPS:
                    continue

                cp.lambda_n_xpbd = new_lambda
                if d_lambda > max_push:
                    d_lambda = max_push

                dp = n * d_lambda
                self._apply_position_delta(a, -dp, cp.world_a)
                if b is not None:
                    self._apply_position_delta(b, +dp, cp.world_b)

    def _solve_friction_contacts_xpbd_iteration(self):
        alpha_t = self.tangent_compliance / (self.dt * self.dt)

        for m in self._manifolds:
            n = m.normal
            t1 = m.tangent1
            t2 = m.tangent2

            a = self.bodies[m.body_a]
            b = self.bodies[m.body_b] if m.body_b >= 0 else None

            for i in range(m.point_count):
                cp = m.points[i]

                lam_n = cp.lambda_n_xpbd
                if lam_n <= 0.0:
                    cp.lambda_t1 = 0.0
                    cp.lambda_t2 = 0.0
                    self._reset_friction_anchor_to_current_contact(m, cp)
                    continue

                fa, fb = self._refresh_friction_world(m, cp)
                tangential_delta = (fb - fa) - n * glm.dot(fb - fa, n)

                c1 = glm.dot(tangential_delta, t1)
                c2 = glm.dot(tangential_delta, t2)

                ra = fa - a.state.x
                rb = (fb - b.state.x) if b is not None else vec3(0.0, 0.0, 0.0)

                cp.tangent_mass_1 = self._compute_effective_mass(a, b, ra, rb, t1)
                cp.tangent_mass_2 = self._compute_effective_mass(a, b, ra, rb, t2)

                dl1 = 0.0
                dl2 = 0.0
                if cp.tangent_mass_1 > 0.0:
                    dl1 = (-c1 - alpha_t * cp.lambda_t1) / (1.0 / cp.tangent_mass_1 + alpha_t)
                if cp.tangent_mass_2 > 0.0:
                    dl2 = (-c2 - alpha_t * cp.lambda_t2) / (1.0 / cp.tangent_mass_2 + alpha_t)

                cand_l1 = cp.lambda_t1 + dl1
                cand_l2 = cp.lambda_t2 + dl2

                tang_len = (cand_l1 * cand_l1 + cand_l2 * cand_l2) ** 0.5
                static_limit = m.mu_s * lam_n

                if tang_len <= static_limit + 1.0e-12:
                    new_l1 = cand_l1
                    new_l2 = cand_l2
                else:
                    dynamic_limit = m.mu_d * lam_n
                    if tang_len > EPS:
                        s = dynamic_limit / tang_len
                        new_l1 = cand_l1 * s
                        new_l2 = cand_l2 * s
                    else:
                        new_l1 = 0.0
                        new_l2 = 0.0

                    self._reset_friction_anchor_to_current_contact(m, cp)

                dl1 = new_l1 - cp.lambda_t1
                dl2 = new_l2 - cp.lambda_t2
                cp.lambda_t1 = new_l1
                cp.lambda_t2 = new_l2

                if abs(dl1) <= EPS and abs(dl2) <= EPS:
                    continue

                dp = t1 * dl1 + t2 * dl2
                self._apply_position_delta(a, -dp, fa)
                if b is not None:
                    self._apply_position_delta(b, +dp, fb)

    def _relax_velocity_contacts_iteration(self):
        for m in self._manifolds:
            n = m.normal
            a = self.bodies[m.body_a]
            b = self.bodies[m.body_b] if m.body_b >= 0 else None

            for i in range(m.point_count):
                cp = m.points[i]
                self._refresh_contact_world(m, cp)

                if cp.lambda_n_xpbd <= 0.0 and cp.separation > 1.0e-3:
                    continue

                ra = cp.world_a - a.state.x
                rb = (cp.world_b - b.state.x) if b is not None else vec3(0.0, 0.0, 0.0)

                cp.normal_mass = self._compute_effective_mass(a, b, ra, rb, n)
                if cp.normal_mass <= 0.0:
                    continue

                va = body_point_velocity_world(a, cp.local_a)

                if b is not None:
                    vb = body_point_velocity_world(b, cp.local_b)
                    vn = glm.dot(vb - va, n)
                else:
                    vn = glm.dot(-va, n)

                old_lambda = cp.lambda_n
                d_lambda = -cp.normal_mass * vn
                new_lambda = max(0.0, old_lambda + d_lambda)
                d_lambda = new_lambda - old_lambda
                cp.lambda_n = new_lambda

                if abs(d_lambda) <= EPS:
                    continue

                P = n * d_lambda
                apply_world_impulse(a, -P, cp.world_a, self._impulse_scratch)
                if b is not None:
                    apply_world_impulse(b, +P, cp.world_b, self._impulse_scratch)

    def _solve_velocity_friction_iteration(self):
        for m in self._manifolds:
            t1 = m.tangent1
            t2 = m.tangent2

            a = self.bodies[m.body_a]
            b = self.bodies[m.body_b] if m.body_b >= 0 else None

            for i in range(m.point_count):
                cp = m.points[i]
                self._refresh_contact_world(m, cp)

                normal_limit = cp.lambda_n
                if normal_limit <= EPS:
                    # Fallback so the very first post iteration still has a usable cap.
                    normal_limit = cp.lambda_n_xpbd / max(self.dt, 1.0e-9)
                if normal_limit <= EPS:
                    continue

                ra = cp.world_a - a.state.x
                rb = (cp.world_b - b.state.x) if b is not None else vec3(0.0, 0.0, 0.0)

                cp.tangent_mass_1 = self._compute_effective_mass(a, b, ra, rb, t1)
                cp.tangent_mass_2 = self._compute_effective_mass(a, b, ra, rb, t2)

                va = body_point_velocity_world(a, cp.local_a)
                if b is not None:
                    vb = body_point_velocity_world(b, cp.local_b)
                    vr = vb - va
                else:
                    vr = -va

                vt1 = glm.dot(vr, t1)
                vt2 = glm.dot(vr, t2)

                dl1 = -cp.tangent_mass_1 * vt1 if cp.tangent_mass_1 > 0.0 else 0.0
                dl2 = -cp.tangent_mass_2 * vt2 if cp.tangent_mass_2 > 0.0 else 0.0

                mag = (dl1 * dl1 + dl2 * dl2) ** 0.5
                max_mag = m.mu_d * normal_limit
                if mag > max_mag and mag > EPS:
                    s = max_mag / mag
                    dl1 *= s
                    dl2 *= s

                if abs(dl1) <= EPS and abs(dl2) <= EPS:
                    continue

                P = t1 * dl1 + t2 * dl2
                apply_world_impulse(a, -P, cp.world_a, self._impulse_scratch)
                if b is not None:
                    apply_world_impulse(b, +P, cp.world_b, self._impulse_scratch)

    def _step_xpbd_friction(self):
        start_t = self._hz_tracker.begin()

        for i, body in enumerate(self.bodies):
            self._prev_x[i] = Vec3(body.state.x)
            self._prev_q[i] = Quat(body.state.q)

        for i, body in enumerate(self.bodies):
            integrate_body_velocities_implicit_gyro(
                body,
                self.dt,
                body.mass * self.gravity,
                Vec3(0.0, 0.0, 0.0),
                self._gyro[i],
                newton_iters=self.newton_iters,
            )

        for body in self.bodies:
            integrate_body_pose_symplectic(body, self.dt)

        # Build once for the whole step.
        self._update_body_caches()
        self._build_manifolds_current_pose()
        self._prepare_contact_state_current()

        # Normal XPBD.
        for _ in range(self.xpbd_iters):
            self._solve_position_contacts_xpbd_iteration()

        # Refresh the same manifolds; do not rebuild them.
        self._prepare_contact_state_current()

        # Static-friction style positional solve.
        for _ in range(self.xpbd_friction_iters):
            self._solve_friction_contacts_xpbd_iteration()

        # Reconstruct velocities from corrected poses.
        for i, body in enumerate(self.bodies):
            reconstruct_velocities_from_pose_delta(body, self._prev_x[i], self._prev_q[i], self.dt)

        # Post velocity solve.
        self._prepare_contact_state_current()
        for m in self._manifolds:
            for i in range(m.point_count):
                m.points[i].lambda_n = 0.0

        for _ in range(self.xpbd_post_iters):
            self._relax_velocity_contacts_iteration()
            self._solve_velocity_friction_iteration()

        self._update_body_caches()
        self._prepare_contact_state_current()
        self._measure_step_end(start_t)
        self._finish_manifold_build()