from dataclasses import dataclass, field
from enum import Enum, auto

from ..common import Mat3, Quat, Vec3


@dataclass(slots=True)
class ImpulseScratch:
    r: Vec3 = field(default_factory=Vec3)
    tau: Vec3 = field(default_factory=Vec3)
    w_world: Vec3 = field(default_factory=Vec3)
    tmp: Vec3 = field(default_factory=Vec3)


@dataclass(slots=True)
class PositionImpulseScratch:
    r: Vec3 = field(default_factory=Vec3)
    delta_theta: Vec3 = field(default_factory=Vec3)
    dq: Quat = field(default_factory=Quat)


@dataclass(slots=True)
class MassScratch:
    r: Vec3 = field(default_factory=Vec3)
    rn: Vec3 = field(default_factory=Vec3)
    tmp: Vec3 = field(default_factory=Vec3)


@dataclass(slots=True)
class GyroScratch:
    w: Vec3 = field(default_factory=Vec3)
    Iw: Vec3 = field(default_factory=Vec3)
    diff: Vec3 = field(default_factory=Vec3)
    f: Vec3 = field(default_factory=Vec3)
    delta: Vec3 = field(default_factory=Vec3)
    tau_body: Vec3 = field(default_factory=Vec3)
    m0: Mat3 = field(default_factory=Mat3)
    m1: Mat3 = field(default_factory=Mat3)
    m2: Mat3 = field(default_factory=Mat3)


class Part2Case(Enum):
    SPRING_FORCE_SI = auto()
    SPRING_SOFT_SI = auto()
    SPRING_XPBD = auto()
    JOINT_XPBD = auto()
    JOINT_SI_BAUMGARTE = auto()
    JOINT_SI_NGS = auto()
    JOINT_SI_SOFT = auto()


CASE_LABELS = {
    Part2Case.SPRING_FORCE_SI: "SI: off-center spring as external force",
    Part2Case.SPRING_SOFT_SI: "SI: off-center spring as soft constraint",
    Part2Case.SPRING_XPBD: "XPBD: off-center spring (distance compliance)",
    Part2Case.JOINT_XPBD: "XPBD: 2 bodies, off-center distance joint",
    Part2Case.JOINT_SI_BAUMGARTE: "SI: 2 bodies, Baumgarte position error fix",
    Part2Case.JOINT_SI_NGS: "SI: 2 bodies, Nonlinear Gauss-Seidel fix",
    Part2Case.JOINT_SI_SOFT: "SI: 2 bodies, soft-constraint spring joint",
}


@dataclass(slots=True)
class FixedAnchorSpring:
    local_anchor: Vec3
    fixed_point: Vec3
    rest_length: float
    stiffness: float = 0.0
    damping: float = 0.0
    hertz: float = 5.0
    damping_ratio: float = 1.0
    compliance: float = 0.0
    lambda_n: float = 0.0


@dataclass(slots=True)
class DistanceJoint:
    local_anchor_a: Vec3
    local_anchor_b: Vec3
    rest_length: float
    hertz: float = 5.0
    damping_ratio: float = 1.0
    beta: float = 0.2
    ngs_beta: float = 0.5
    compliance: float = 0.0
    lambda_n: float = 0.0
