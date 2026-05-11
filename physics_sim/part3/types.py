from dataclasses import dataclass, field

from ..common import Vec3, vec3
from .constants import MAX_BOX_VERTS, MAX_CLIP_VERTS, MAX_CONTACT_POINTS


@dataclass(slots=True)
class CandidatePair:
    body_a: int = 0
    body_b: int = 0


@dataclass(slots=True)
class AABB:
    min_v: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    max_v: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))


@dataclass(slots=True)
class BodyCollisionCache:
    x: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    axis_x: Vec3 = field(default_factory=lambda: vec3(1.0, 0.0, 0.0))
    axis_y: Vec3 = field(default_factory=lambda: vec3(0.0, 1.0, 0.0))
    axis_z: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 1.0))
    corners: list = field(default_factory=lambda: [vec3(0.0, 0.0, 0.0) for _ in range(MAX_BOX_VERTS)])
    aabb: AABB = field(default_factory=AABB)


@dataclass(slots=True)
class ContactPoint:
    id: int = 0

    world_a: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    world_b: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))

    local_a: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    local_b: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))

    separation: float = 0.0

    lambda_n: float = 0.0
    lambda_n_xpbd: float = 0.0
    normal_mass: float = 0.0

    tangent_mass_1: float = 0.0
    tangent_mass_2: float = 0.0
    lambda_t1: float = 0.0
    lambda_t2: float = 0.0

    friction_local_a: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    friction_local_b: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    friction_valid: bool = False


@dataclass(slots=True)
class ContactManifold:
    body_a: int = 0
    body_b: int = 0
    normal: Vec3 = field(default_factory=lambda: vec3(0.0, 1.0, 0.0))
    point_count: int = 0
    points: list = field(default_factory=lambda: [ContactPoint() for _ in range(MAX_CONTACT_POINTS)])
    pair_key: tuple = (0, 0)

    axis_type: int = -1
    axis_i: int = -1
    axis_j: int = -1
    tangent1: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    tangent2: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    mu_s: float = 0.0
    mu_d: float = 0.0


@dataclass(slots=True)
class ClipVertex:
    p: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    feature: int = 0


@dataclass(slots=True)
class CollisionScratch:
    clip_in: list = field(default_factory=lambda: [ClipVertex() for _ in range(MAX_CLIP_VERTS)])
    clip_out: list = field(default_factory=lambda: [ClipVertex() for _ in range(MAX_CLIP_VERTS)])
    reduce_ids: list = field(default_factory=lambda: [0 for _ in range(MAX_CONTACT_POINTS)])


@dataclass(slots=True)
class SatAxisResult:
    hit: bool = False
    axis_type: int = -1
    axis_i: int = -1
    axis_j: int = -1
    normal: Vec3 = field(default_factory=lambda: vec3(1.0, 0.0, 0.0))
    penetration: float = 0.0
    score: float = 0.0