from pyglm import glm
from enum import Enum, auto

from ..common import (
    Mat3,
    Vec3,
    integrate_orientation_body,
    integrate_orientation_world,
    mat3_from_diag,
    skew_mat,
)


class Part1Case(Enum):
    WORLD_CONSTANT_L = auto()
    LOCAL_NO_GYRO = auto()
    LOCAL_GYRO_EXPLICIT = auto()
    LOCAL_GYRO_IMPLICIT = auto()


CASE_LABELS = {
    Part1Case.WORLD_CONSTANT_L: "World coords, constant L",
    Part1Case.LOCAL_NO_GYRO: "Local coords, no gyro",
    Part1Case.LOCAL_GYRO_EXPLICIT: "Local coords, gyro explicit",
    Part1Case.LOCAL_GYRO_IMPLICIT: "Local coords, gyro implicit",
}


class Part1Solver:
    def __init__(self, body, dt: float, case: Part1Case, newton_iters: int = 8):
        self.body = body.clone()
        self.dt = float(dt)
        self.case = case
        self.newton_iters = int(newton_iters)

        self.L0_world = Vec3(self.body.L_world())
        self.E0 = self.body.kinetic_energy_rot()
        self.I_world0 = Mat3(self.body.I_world())
        self.I_world0_inv = Mat3(self.body.I_world_inv())

    def step(self):
        if self.case == Part1Case.WORLD_CONSTANT_L:
            self._step_world_constant_L()
        elif self.case == Part1Case.LOCAL_NO_GYRO:
            self._step_local_no_gyro()
        elif self.case == Part1Case.LOCAL_GYRO_EXPLICIT:
            self._step_local_gyro_explicit()
        elif self.case == Part1Case.LOCAL_GYRO_IMPLICIT:
            self._step_local_gyro_implicit()
        else:
            raise ValueError(f"Unknown case: {self.case}")

    def _step_world_constant_L(self):
        w_world = self.I_world0_inv * self.L0_world
        q_new = integrate_orientation_world(self.body.state.q, w_world, self.dt)
        self.body.set_q(q_new)
        self.body.state.w_body = glm.transpose(self.body.R()) * w_world

    def _step_local_no_gyro(self):
        q_new = integrate_orientation_body(self.body.state.q, self.body.state.w_body, self.dt)
        self.body.set_q(q_new)

    def _step_local_gyro_explicit(self):
        I_diag = self.body.I_body_diag
        I_inv = self.body.I_body_inv_diag
        w_n = self.body.state.w_body
        Iw = Vec3(I_diag.x * w_n.x, I_diag.y * w_n.y, I_diag.z * w_n.z)
        gyro = glm.cross(w_n, Iw)
        w_new = Vec3(
            w_n.x - self.dt * I_inv.x * gyro.x,
            w_n.y - self.dt * I_inv.y * gyro.y,
            w_n.z - self.dt * I_inv.z * gyro.z,
        )
        self.body.state.w_body = w_new
        q_new = integrate_orientation_body(self.body.state.q, w_new, self.dt)
        self.body.set_q(q_new)

    def _step_local_gyro_implicit(self):
        I_diag = self.body.I_body_diag
        w_n = self.body.state.w_body
        w = Vec3(w_n)

        for _ in range(self.newton_iters):
            Iw = Vec3(I_diag.x * w.x, I_diag.y * w.y, I_diag.z * w.z)
            diff = Vec3(
                I_diag.x * (w.x - w_n.x),
                I_diag.y * (w.y - w_n.y),
                I_diag.z * (w.z - w_n.z),
            )
            f = diff + self.dt * glm.cross(w, Iw)
            if glm.length(f) < 1e-12:
                break
            m0 = skew_mat(Iw)
            m1 = skew_mat(w)
            m2 = m1 * mat3_from_diag(I_diag)
            m2 = m2 - m0
            m1 = mat3_from_diag(I_diag) + self.dt * m2
            delta = glm.inverse(m1) * (-f)
            w = w + delta
            if glm.length(delta) < 1e-12:
                break

        self.body.state.w_body = w
        q_new = integrate_orientation_body(self.body.state.q, w, self.dt)
        self.body.set_q(q_new)
