from dataclasses import dataclass, field

from pyglm import glm

from .math3d import mat3_from_diag
from .math3d import Mat3, Quat, Vec3


@dataclass(slots=True)
class BoxShape:
    half_extents: Vec3  # [hx, hy, hz]


@dataclass(slots=True)
class RigidBodyState:
    x: Vec3
    q: Quat
    v: Vec3
    w_body: Vec3


@dataclass(slots=True)
class RigidBody:
    shape: BoxShape
    mass: float
    state: RigidBodyState
    _I_body_diag: Vec3 = field(init=False)
    _I_body_inv_diag: Vec3 = field(init=False)
    _R: Mat3 = field(init=False)
    _I_world: Mat3 = field(init=False)
    _I_world_inv: Mat3 = field(init=False)
    _cache_dirty: bool = field(init=False, default=True)

    def __post_init__(self):
        self.mass = float(self.mass)
        self._I_body_diag = Vec3(0.0, 0.0, 0.0)
        self._I_body_inv_diag = Vec3(0.0, 0.0, 0.0)
        self._R = Mat3(1.0)
        self._I_world = Mat3(1.0)
        self._I_world_inv = Mat3(1.0)
        self._update_inertia_diagonal()
        self._update_cache()

    def clone(self):
        return RigidBody(
            shape=BoxShape(Vec3(self.shape.half_extents.x, self.shape.half_extents.y, self.shape.half_extents.z)),
            mass=self.mass,
            state=RigidBodyState(
                x=Vec3(self.state.x.x, self.state.x.y, self.state.x.z),
                q=Quat(self.state.q),
                v=Vec3(self.state.v.x, self.state.v.y, self.state.v.z),
                w_body=Vec3(self.state.w_body.x, self.state.w_body.y, self.state.w_body.z),
            ),
        )

    def _update_inertia_diagonal(self):
        hx, hy, hz = self.shape.half_extents.x, self.shape.half_extents.y, self.shape.half_extents.z
        sx, sy, sz = 2.0 * hx, 2.0 * hy, 2.0 * hz
        ix = self.mass * (sy * sy + sz * sz) / 12.0
        iy = self.mass * (sx * sx + sz * sz) / 12.0
        iz = self.mass * (sx * sx + sy * sy) / 12.0
        self._I_body_diag = Vec3(ix, iy, iz)
        self._I_body_inv_diag = Vec3(
            0.0 if ix == 0.0 else 1.0 / ix,
            0.0 if iy == 0.0 else 1.0 / iy,
            0.0 if iz == 0.0 else 1.0 / iz,
        )

    def mark_dirty(self):
        self._cache_dirty = True

    def set_q(self, q):
        self.state.q = Quat(q)
        self._cache_dirty = True

    def _update_cache(self):
        self._R = glm.mat3_cast(self.state.q)
        self._I_world = self._build_world_inertia(self._I_body_diag)
        self._I_world_inv = self._build_world_inertia(self._I_body_inv_diag)
        self._cache_dirty = False

    def _build_world_inertia(self, diag):
        d = mat3_from_diag(diag)
        return self._R * d * glm.transpose(self._R)

    def _ensure_cache(self):
        if self._cache_dirty:
            self._update_cache()

    @property
    def I_body_diag(self):
        return self._I_body_diag

    @property
    def I_body_inv_diag(self):
        return self._I_body_inv_diag

    def R(self):
        self._ensure_cache()
        return self._R

    def I_world(self):
        self._ensure_cache()
        return self._I_world

    def I_world_inv(self):
        self._ensure_cache()
        return self._I_world_inv

    def omega_world(self, out=None):
        self._ensure_cache()
        w = self._R * self.state.w_body
        if out is None:
            return w
        out.x, out.y, out.z = w.x, w.y, w.z
        return out

    def L_body(self, out=None):
        w = self.state.w_body
        L = Vec3(self._I_body_diag.x * w.x, self._I_body_diag.y * w.y, self._I_body_diag.z * w.z)
        if out is None:
            return L
        out.x, out.y, out.z = L.x, L.y, L.z
        return out

    def L_world(self, out=None):
        self._ensure_cache()
        L = self._R * self.L_body()
        if out is None:
            return L
        out.x, out.y, out.z = L.x, L.y, L.z
        return out

    def kinetic_energy_rot(self):
        w = self.state.w_body
        return 0.5 * (
            w.x * self._I_body_diag.x * w.x
            + w.y * self._I_body_diag.y * w.y
            + w.z * self._I_body_diag.z * w.z
        )
