from pyglm import glm

from .constants import CONTACT_SLOP, EPS
from .manifolds import refresh_manifold_world_points
from ..part2.constraints import effective_mass_scalar_one_body, effective_mass_scalar_two_body
from ..part2.physics import apply_world_impulse, apply_world_position_impulse, body_point_velocity_world


def point_relative_normal_velocity_one_body(body, local_a, normal):
    # vb = 0 for the plane; use vb - va to match two-body convention.
    return glm.dot(-body_point_velocity_world(body, local_a), normal)


def point_relative_normal_velocity_two_body(body_a, local_a, body_b, local_b, normal):
    return glm.dot(body_point_velocity_world(body_b, local_b) - body_point_velocity_world(body_a, local_a), normal)


def prepare_contact_masses(manifolds, bodies, mass_scratch_a, mass_scratch_b):
    for m in manifolds:
        refresh_manifold_world_points(m, bodies)
        n = m.normal
        a = bodies[m.body_a]
        if m.body_b >= 0:
            b = bodies[m.body_b]
            for i in range(m.point_count):
                cp = m.points[i]
                k = effective_mass_scalar_two_body(a, cp.world_a, b, cp.world_b, n, mass_scratch_a, mass_scratch_b)
                cp.normal_mass = 0.0 if k <= EPS else 1.0 / k
        else:
            for i in range(m.point_count):
                cp = m.points[i]
                k = effective_mass_scalar_one_body(a, cp.world_a, n, mass_scratch_a)
                cp.normal_mass = 0.0 if k <= EPS else 1.0 / k


def warm_start_contact_impulses(manifolds, bodies, impulse_scratch):
    for m in manifolds:
        n = m.normal
        a = bodies[m.body_a]
        if m.body_b >= 0:
            b = bodies[m.body_b]
            for i in range(m.point_count):
                cp = m.points[i]
                if cp.lambda_n == 0.0:
                    continue
                P = n * cp.lambda_n
                apply_world_impulse(a, -P, cp.world_a, impulse_scratch)
                apply_world_impulse(b, +P, cp.world_b, impulse_scratch)
        else:
            for i in range(m.point_count):
                cp = m.points[i]
                if cp.lambda_n == 0.0:
                    continue
                apply_world_impulse(a, -(n * cp.lambda_n), cp.world_a, impulse_scratch)


def solve_contact_velocity_si(manifolds, bodies, impulse_scratch):
    for m in manifolds:
        n = m.normal
        a = bodies[m.body_a]

        if m.body_b >= 0:
            b = bodies[m.body_b]
            for i in range(m.point_count):
                cp = m.points[i]
                if cp.normal_mass <= 0.0:
                    continue
                vn = point_relative_normal_velocity_two_body(a, cp.local_a, b, cp.local_b, n)
                old_lambda = cp.lambda_n
                cp.lambda_n = max(0.0, old_lambda - cp.normal_mass * vn)
                d_lambda = cp.lambda_n - old_lambda
                if d_lambda == 0.0:
                    continue
                P = n * d_lambda
                apply_world_impulse(a, -P, cp.world_a, impulse_scratch)
                apply_world_impulse(b, +P, cp.world_b, impulse_scratch)
        else:
            for i in range(m.point_count):
                cp = m.points[i]
                if cp.normal_mass <= 0.0:
                    continue
                vn = point_relative_normal_velocity_one_body(a, cp.local_a, n)
                old_lambda = cp.lambda_n
                cp.lambda_n = max(0.0, old_lambda - cp.normal_mass * vn)
                d_lambda = cp.lambda_n - old_lambda
                if d_lambda == 0.0:
                    continue
                apply_world_impulse(a, -(n * d_lambda), cp.world_a, impulse_scratch)


def solve_contact_position_ngs(manifolds, bodies, pos_impulse_scratch,
                               mass_scratch_a, mass_scratch_b,
                               beta=0.4, slop=CONTACT_SLOP):
    for m in manifolds:
        refresh_manifold_world_points(m, bodies)
        n = m.normal
        a = bodies[m.body_a]

        if m.body_b >= 0:
            b = bodies[m.body_b]
            for i in range(m.point_count):
                cp = m.points[i]
                C = min(0.0, cp.separation + slop)
                if C >= 0.0:
                    continue
                k = effective_mass_scalar_two_body(a, cp.world_a, b, cp.world_b, n, mass_scratch_a, mass_scratch_b)
                if k <= EPS:
                    continue
                P = n * (-beta * C / k)
                apply_world_position_impulse(a, -P, cp.world_a, pos_impulse_scratch)
                apply_world_position_impulse(b, +P, cp.world_b, pos_impulse_scratch)
        else:
            for i in range(m.point_count):
                cp = m.points[i]
                C = min(0.0, cp.separation + slop)
                if C >= 0.0:
                    continue
                k = effective_mass_scalar_one_body(a, cp.world_a, n, mass_scratch_a)
                if k <= EPS:
                    continue
                apply_world_position_impulse(a, -(n * (-beta * C / k)), cp.world_a, pos_impulse_scratch)


def solve_contact_position_xpbd(manifolds, bodies, pos_impulse_scratch,
                                mass_scratch_a, mass_scratch_b,
                                dt, compliance=0.0, slop=0.0):
    alpha_tilde = compliance / (dt * dt) if compliance > 0.0 else 0.0

    for m in manifolds:
        refresh_manifold_world_points(m, bodies)
        n = m.normal
        a = bodies[m.body_a]

        if m.body_b >= 0:
            b = bodies[m.body_b]
            for i in range(m.point_count):
                cp = m.points[i]
                C = min(0.0, cp.separation + slop)
                if C >= 0.0:
                    continue
                k = effective_mass_scalar_two_body(a, cp.world_a, b, cp.world_b, n, mass_scratch_a, mass_scratch_b)
                if k <= EPS:
                    continue
                d_lambda = (-C - alpha_tilde * cp.lambda_n_xpbd) / (k + alpha_tilde)
                cp.lambda_n_xpbd += d_lambda
                P = n * d_lambda
                apply_world_position_impulse(a, -P, cp.world_a, pos_impulse_scratch)
                apply_world_position_impulse(b, +P, cp.world_b, pos_impulse_scratch)
        else:
            for i in range(m.point_count):
                cp = m.points[i]
                C = min(0.0, cp.separation + slop)
                if C >= 0.0:
                    continue
                k = effective_mass_scalar_one_body(a, cp.world_a, n, mass_scratch_a)
                if k <= EPS:
                    continue
                d_lambda = (-C - alpha_tilde * cp.lambda_n_xpbd) / (k + alpha_tilde)
                cp.lambda_n_xpbd += d_lambda
                apply_world_position_impulse(a, -(n * d_lambda), cp.world_a, pos_impulse_scratch)
