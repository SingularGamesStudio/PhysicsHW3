from ..common import vec3
from .solver import Part2Solver
from .types import DistanceJoint, FixedAnchorSpring, Part2Case


def make_part2_solvers(body_spring, body_a, body_b, dt):
    spring_force_cfg = FixedAnchorSpring(
        local_anchor=vec3(0.95, 0.15, 0.0),
        fixed_point=vec3(0.0, 2.15, 0.0),
        rest_length=1.15,
        stiffness=80.0,
        damping=7.0,
    )
    spring_soft_cfg = FixedAnchorSpring(
        local_anchor=vec3(0.95, 0.15, 0.0),
        fixed_point=vec3(0.0, 2.15, 0.0),
        rest_length=1.15,
        hertz=5.0,
        damping_ratio=1.0,
    )
    spring_xpbd_cfg = FixedAnchorSpring(
        local_anchor=vec3(0.95, 0.15, 0.0),
        fixed_point=vec3(0.0, 2.15, 0.0),
        rest_length=1.15,
        compliance=2.5e-4,
    )
    joint_cfg_xpbd = DistanceJoint(
        local_anchor_a=vec3(0.85, 0.0, 0.0),
        local_anchor_b=vec3(-0.85, 0.0, 0.0),
        rest_length=2.1,
        compliance=1.0e-5,
    )
    joint_cfg_si = DistanceJoint(
        local_anchor_a=vec3(0.85, 0.0, 0.0),
        local_anchor_b=vec3(-0.85, 0.0, 0.0),
        rest_length=2.1,
        hertz=6.0,
        damping_ratio=1.0,
        beta=0.22,
        ngs_beta=0.55,
    )
    body_a_spring_cfg = FixedAnchorSpring(
        local_anchor=vec3(-0.95, 0.25, 0.0),
        fixed_point=vec3(-2.4, 2.4, 0.0),
        rest_length=1.35,
        stiffness=70.0,
        damping=6.0,
    )
    return {
        Part2Case.SPRING_FORCE_SI: Part2Solver(
            bodies=[body_spring],
            dt=dt,
            case=Part2Case.SPRING_FORCE_SI,
            spring=spring_force_cfg,
            vel_iters=1,
            pos_iters=0,
            xpbd_iters=0,
            newton_iters=10,
        ),
        Part2Case.SPRING_SOFT_SI: Part2Solver(
            bodies=[body_spring],
            dt=dt,
            case=Part2Case.SPRING_SOFT_SI,
            spring=spring_soft_cfg,
            vel_iters=8,
            pos_iters=0,
            xpbd_iters=0,
            newton_iters=10,
        ),
        Part2Case.SPRING_XPBD: Part2Solver(
            bodies=[body_spring],
            dt=dt,
            case=Part2Case.SPRING_XPBD,
            spring=spring_xpbd_cfg,
            vel_iters=0,
            pos_iters=0,
            xpbd_iters=8,
            newton_iters=10,
        ),
        Part2Case.JOINT_XPBD: Part2Solver(
            bodies=[body_a, body_b],
            dt=dt,
            case=Part2Case.JOINT_XPBD,
            joint=joint_cfg_xpbd,
            vel_iters=0,
            pos_iters=0,
            xpbd_iters=8,
            newton_iters=10,
            body_a_spring=body_a_spring_cfg,
        ),
        Part2Case.JOINT_SI_BAUMGARTE: Part2Solver(
            bodies=[body_a, body_b],
            dt=dt,
            case=Part2Case.JOINT_SI_BAUMGARTE,
            joint=joint_cfg_si,
            vel_iters=8,
            pos_iters=0,
            xpbd_iters=0,
            newton_iters=10,
            body_a_spring=body_a_spring_cfg,
        ),
        Part2Case.JOINT_SI_NGS: Part2Solver(
            bodies=[body_a, body_b],
            dt=dt,
            case=Part2Case.JOINT_SI_NGS,
            joint=joint_cfg_si,
            vel_iters=8,
            pos_iters=8,
            xpbd_iters=0,
            newton_iters=10,
            body_a_spring=body_a_spring_cfg,
        ),
        Part2Case.JOINT_SI_SOFT: Part2Solver(
            bodies=[body_a, body_b],
            dt=dt,
            case=Part2Case.JOINT_SI_SOFT,
            joint=joint_cfg_si,
            vel_iters=8,
            pos_iters=0,
            xpbd_iters=0,
            newton_iters=10,
            body_a_spring=body_a_spring_cfg,
        ),
    }
