from pyglm import glm

from ..common import Vec3, angular_velocity_world_from_quat_delta, as_vec3, q_to_list, v_to_list
from .constraints import (
    effective_mass_scalar_one_body,
    effective_mass_scalar_two_body,
    fixed_anchor_distance_info,
    fixed_anchor_spring_force,
    soft_constraint_coeffs,
    two_body_distance_info,
)
from .integration import integrate_body_pose_symplectic, integrate_body_velocities_implicit_gyro
from .physics import (
    apply_world_impulse,
    apply_world_position_impulse,
    body_point_velocity_world,
    total_kinetic_energy,
)
from .types import (
    CASE_LABELS,
    DistanceJoint,
    FixedAnchorSpring,
    GyroScratch,
    ImpulseScratch,
    MassScratch,
    Part2Case,
    PositionImpulseScratch,
)


class Part2Solver:
    def __init__(
        self,
        bodies,
        dt,
        case,
        spring=None,
        joint=None,
        gravity=Vec3(0.0, -9.81, 0.0),
        vel_iters=8,
        pos_iters=8,
        xpbd_iters=8,
        newton_iters=8,
        body_a_spring=None,
    ):
        self.bodies = [b.clone() for b in bodies]
        self.dt = float(dt)
        self.case = case
        self.spring = None if spring is None else FixedAnchorSpring(
            local_anchor=as_vec3(spring.local_anchor),
            fixed_point=as_vec3(spring.fixed_point),
            rest_length=float(spring.rest_length),
            stiffness=float(spring.stiffness),
            damping=float(spring.damping),
            hertz=float(spring.hertz),
            damping_ratio=float(spring.damping_ratio),
            compliance=float(spring.compliance),
            lambda_n=float(spring.lambda_n),
        )
        self.joint = None if joint is None else DistanceJoint(
            local_anchor_a=as_vec3(joint.local_anchor_a),
            local_anchor_b=as_vec3(joint.local_anchor_b),
            rest_length=float(joint.rest_length),
            hertz=float(joint.hertz),
            damping_ratio=float(joint.damping_ratio),
            beta=float(joint.beta),
            ngs_beta=float(joint.ngs_beta),
            compliance=float(joint.compliance),
            lambda_n=float(joint.lambda_n),
        )
        self.body_a_spring = body_a_spring
        self.gravity = as_vec3(gravity)
        self.vel_iters = int(vel_iters)
        self.pos_iters = int(pos_iters)
        self.xpbd_iters = int(xpbd_iters)
        self.newton_iters = int(newton_iters)
        self._gyro_a = GyroScratch()
        self._gyro_b = GyroScratch()
        self._impulse_scratch = ImpulseScratch()
        self._pos_impulse_scratch = PositionImpulseScratch()
        self._mass_scratch_a = MassScratch()
        self._mass_scratch_b = MassScratch()
        self._prev_x_a = Vec3(0.0, 0.0, 0.0)
        self._prev_x_b = Vec3(0.0, 0.0, 0.0)
        self._prev_q_a = glm.quat(1.0, 0.0, 0.0, 0.0)
        self._prev_q_b = glm.quat(1.0, 0.0, 0.0, 0.0)

    def step(self):
        if self.case == Part2Case.SPRING_FORCE_SI:
            self._step_spring_force_si()
        elif self.case == Part2Case.SPRING_SOFT_SI:
            self._step_spring_soft_si()
        elif self.case == Part2Case.SPRING_XPBD:
            self._step_spring_xpbd()
        elif self.case == Part2Case.JOINT_XPBD:
            self._step_joint_xpbd()
        elif self.case == Part2Case.JOINT_SI_BAUMGARTE:
            self._step_joint_si(mode="baumgarte")
        elif self.case == Part2Case.JOINT_SI_NGS:
            self._step_joint_si(mode="ngs")
        elif self.case == Part2Case.JOINT_SI_SOFT:
            self._step_joint_si(mode="soft")
        else:
            raise ValueError(f"Unknown case: {self.case}")

    def snapshot(self, t):
        C = self.current_constraint_error()
        lam = self.current_lambda()
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
            "C": float(C),
            "lambda": float(lam),
            "E": float(total_kinetic_energy(self.bodies)),
        }

    def current_constraint_error(self):
        if self.case in (Part2Case.SPRING_FORCE_SI, Part2Case.SPRING_SOFT_SI, Part2Case.SPRING_XPBD):
            body = self.bodies[0]
            C, _, _, _ = fixed_anchor_distance_info(
                body, self.spring.local_anchor, self.spring.fixed_point, self.spring.rest_length
            )
            return float(C)
        if self.case in (
            Part2Case.JOINT_XPBD,
            Part2Case.JOINT_SI_BAUMGARTE,
            Part2Case.JOINT_SI_NGS,
            Part2Case.JOINT_SI_SOFT,
        ):
            a, b = self.bodies
            C, _, _, _, _ = two_body_distance_info(
                a, self.joint.local_anchor_a, b, self.joint.local_anchor_b, self.joint.rest_length
            )
            return float(C)
        return 0.0

    def current_lambda(self):
        if self.spring is not None:
            return float(self.spring.lambda_n)
        if self.joint is not None:
            return float(self.joint.lambda_n)
        return 0.0

    def _step_spring_force_si(self):
        body = self.bodies[0]
        _, force, torque = fixed_anchor_spring_force(body, self.spring)
        integrate_body_velocities_implicit_gyro(
            body,
            self.dt,
            body.mass * self.gravity + force,
            torque,
            self._gyro_a,
            newton_iters=self.newton_iters,
        )
        integrate_body_pose_symplectic(body, self.dt)

    def _step_spring_soft_si(self):
        body = self.bodies[0]
        integrate_body_velocities_implicit_gyro(
            body,
            self.dt,
            body.mass * self.gravity,
            Vec3(0.0, 0.0, 0.0),
            self._gyro_a,
            newton_iters=self.newton_iters,
        )
        if abs(self.spring.lambda_n) > 0.0:
            C, _, p, n = fixed_anchor_distance_info(
                body, self.spring.local_anchor, self.spring.fixed_point, self.spring.rest_length
            )
            apply_world_impulse(body, n * self.spring.lambda_n, p, self._impulse_scratch)
        bias_rate, mass_coeff, impulse_coeff = soft_constraint_coeffs(
            self.dt, self.spring.hertz, self.spring.damping_ratio
        )
        for _ in range(self.vel_iters):
            self.spring.lambda_n = 0.0
            C, _, p, n = fixed_anchor_distance_info(
                body, self.spring.local_anchor, self.spring.fixed_point, self.spring.rest_length
            )
            k = effective_mass_scalar_one_body(body, p, n, self._mass_scratch_a)
            if k <= 1e-12:
                continue
            meff = 1.0 / k
            v_point = body_point_velocity_world(body, self.spring.local_anchor)
            vn = glm.dot(v_point, n)
            d_lambda = -mass_coeff * meff * (vn + bias_rate * C) - impulse_coeff * self.spring.lambda_n
            self.spring.lambda_n += d_lambda
            apply_world_impulse(body, n * d_lambda, p, self._impulse_scratch)
        integrate_body_pose_symplectic(body, self.dt)

    def _step_spring_xpbd(self):
        body = self.bodies[0]
        self._prev_x_a = Vec3(body.state.x)
        self._prev_q_a = glm.quat(body.state.q)
        self.spring.lambda_n = 0.0
        integrate_body_velocities_implicit_gyro(
            body,
            self.dt,
            body.mass * self.gravity,
            Vec3(0.0, 0.0, 0.0),
            self._gyro_a,
            newton_iters=self.newton_iters,
        )
        integrate_body_pose_symplectic(body, self.dt)
        alpha_tilde = self.spring.compliance / (self.dt * self.dt)
        for _ in range(self.xpbd_iters):
            C, _, p, n = fixed_anchor_distance_info(
                body, self.spring.local_anchor, self.spring.fixed_point, self.spring.rest_length
            )
            k = effective_mass_scalar_one_body(body, p, n, self._mass_scratch_a)
            if k <= 1e-12:
                continue
            d_lambda = (-C - alpha_tilde * self.spring.lambda_n) / (k + alpha_tilde)
            self.spring.lambda_n += d_lambda
            apply_world_position_impulse(body, n * d_lambda, p, self._pos_impulse_scratch)
        body.state.v = (body.state.x - self._prev_x_a) / self.dt
        w_world = angular_velocity_world_from_quat_delta(self._prev_q_a, body.state.q, self.dt)
        body.state.w_body = glm.transpose(body.R()) * w_world

    def _step_joint_xpbd(self):
        a, b = self.bodies
        self._prev_x_a = Vec3(a.state.x)
        self._prev_q_a = glm.quat(a.state.q)
        self._prev_x_b = Vec3(b.state.x)
        self._prev_q_b = glm.quat(b.state.q)
        self.joint.lambda_n = 0.0
        for body, scratch in ((a, self._gyro_a), (b, self._gyro_b)):
            force = body.mass * self.gravity
            torque = Vec3(0.0, 0.0, 0.0)
            if body is a and self.body_a_spring is not None:
                _, f, t = fixed_anchor_spring_force(a, self.body_a_spring)
                force = force + f
                torque = torque + t
            integrate_body_velocities_implicit_gyro(
                body,
                self.dt,
                force,
                torque,
                scratch,
                newton_iters=self.newton_iters,
            )
            integrate_body_pose_symplectic(body, self.dt)
        alpha_tilde = self.joint.compliance / (self.dt * self.dt)
        for _ in range(self.xpbd_iters):
            C, _, p_a, p_b, n = two_body_distance_info(
                a, self.joint.local_anchor_a, b, self.joint.local_anchor_b, self.joint.rest_length
            )
            k = effective_mass_scalar_two_body(
                a, p_a, b, p_b, n, self._mass_scratch_a, self._mass_scratch_b
            )
            if k <= 1e-12:
                continue
            d_lambda = (-C - alpha_tilde * self.joint.lambda_n) / (k + alpha_tilde)
            self.joint.lambda_n += d_lambda
            apply_world_position_impulse(a, -n * d_lambda, p_a, self._pos_impulse_scratch)
            apply_world_position_impulse(b, n * d_lambda, p_b, self._pos_impulse_scratch)
        a.state.v = (a.state.x - self._prev_x_a) / self.dt
        b.state.v = (b.state.x - self._prev_x_b) / self.dt
        a.state.w_body = glm.transpose(a.R()) * angular_velocity_world_from_quat_delta(self._prev_q_a, a.state.q, self.dt)
        b.state.w_body = glm.transpose(b.R()) * angular_velocity_world_from_quat_delta(self._prev_q_b, b.state.q, self.dt)

    def _step_joint_si(self, mode):
        a, b = self.bodies
        force_a = a.mass * self.gravity
        torque_a = Vec3(0.0, 0.0, 0.0)
        if self.body_a_spring is not None:
            _, f, t = fixed_anchor_spring_force(a, self.body_a_spring)
            force_a = force_a + f
            torque_a = torque_a + t
        integrate_body_velocities_implicit_gyro(
            a,
            self.dt,
            force_a,
            torque_a,
            self._gyro_a,
            newton_iters=self.newton_iters,
        )
        integrate_body_velocities_implicit_gyro(
            b,
            self.dt,
            b.mass * self.gravity,
            Vec3(0.0, 0.0, 0.0),
            self._gyro_b,
            newton_iters=self.newton_iters,
        )
        if abs(self.joint.lambda_n) > 0.0:
            C, _, p_a, p_b, n = two_body_distance_info(
                a, self.joint.local_anchor_a, b, self.joint.local_anchor_b, self.joint.rest_length
            )
            apply_world_impulse(a, -n * self.joint.lambda_n, p_a, self._impulse_scratch)
            apply_world_impulse(b, n * self.joint.lambda_n, p_b, self._impulse_scratch)
        if mode == "soft":
            bias_rate, mass_coeff, impulse_coeff = soft_constraint_coeffs(
                self.dt, self.joint.hertz, self.joint.damping_ratio
            )
        for _ in range(self.vel_iters):
            C, _, p_a, p_b, n = two_body_distance_info(
                a, self.joint.local_anchor_a, b, self.joint.local_anchor_b, self.joint.rest_length
            )
            k = effective_mass_scalar_two_body(
                a, p_a, b, p_b, n, self._mass_scratch_a, self._mass_scratch_b
            )
            if k <= 1e-12:
                continue
            meff = 1.0 / k
            v_a = body_point_velocity_world(a, self.joint.local_anchor_a)
            v_b = body_point_velocity_world(b, self.joint.local_anchor_b)
            vn = glm.dot(n, v_b) - glm.dot(n, v_a)
            if mode == "baumgarte":
                d_lambda = -meff * (vn + self.joint.beta * C / self.dt)
            elif mode == "ngs":
                d_lambda = -meff * vn
            elif mode == "soft":
                d_lambda = -mass_coeff * meff * (vn + bias_rate * C) - impulse_coeff * self.joint.lambda_n
            else:
                raise ValueError(f"Unknown SI mode: {mode}")
            self.joint.lambda_n += d_lambda
            apply_world_impulse(a, -n * d_lambda, p_a, self._impulse_scratch)
            apply_world_impulse(b, n * d_lambda, p_b, self._impulse_scratch)
        for body in self.bodies:
            integrate_body_pose_symplectic(body, self.dt)
        if mode == "ngs":
            for _ in range(self.pos_iters):
                C, _, p_a, p_b, n = two_body_distance_info(
                    a, self.joint.local_anchor_a, b, self.joint.local_anchor_b, self.joint.rest_length
                )
                k = effective_mass_scalar_two_body(
                    a, p_a, b, p_b, n, self._mass_scratch_a, self._mass_scratch_b
                )
                if k <= 1e-12:
                    continue
                meff = 1.0 / k
                d_pos_lambda = -self.joint.ngs_beta * meff * C
                apply_world_position_impulse(a, -n * d_pos_lambda, p_a, self._pos_impulse_scratch)
                apply_world_position_impulse(b, n * d_pos_lambda, p_b, self._pos_impulse_scratch)


def case_label(case):
    return CASE_LABELS[case]
