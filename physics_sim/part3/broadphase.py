from .types import AABB, CandidatePair
import math


def aabb_overlap(a: AABB, b: AABB):
    return (
        a.min_v.x <= b.max_v.x and a.max_v.x >= b.min_v.x and
        a.min_v.y <= b.max_v.y and a.max_v.y >= b.min_v.y and
        a.min_v.z <= b.max_v.z and a.max_v.z >= b.min_v.z
    )


class BroadPhaseBase:
    def fill_pairs(self, bodies, body_caches, out_pairs):
        raise NotImplementedError


class BruteForceBroadPhaseFixed(BroadPhaseBase):
    def fill_pairs_fixed(self, bodies, body_caches, out_pairs):
        count = 0
        n = len(bodies)
        for i in range(n):
            ai = body_caches[i].aabb
            for j in range(i + 1, n):
                if aabb_overlap(ai, body_caches[j].aabb):
                    pair = out_pairs[count]
                    pair.body_a = i
                    pair.body_b = j
                    count += 1
        return count

    def fill_pairs(self, bodies, body_caches, out_pairs):
        out_pairs.clear()
        n = len(bodies)
        for i in range(n):
            ai = body_caches[i].aabb
            for j in range(i + 1, n):
                if aabb_overlap(ai, body_caches[j].aabb):
                    out_pairs.append(CandidatePair(i, j))


def _next_pow2(x: int) -> int:
    x = max(1, int(x))
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return x + 1


class SpatialGridBroadPhase(BroadPhaseBase):
    """
    Uniform hashed 3D grid for many similarly-sized rigid bodies.

    Hot-path design:
      - no dict/set/list growth during fill_pairs_fixed()
      - stamped buckets instead of clearing full tables
      - intrusive bucket chains in flat arrays
      - per-body cached cell ranges
      - duplicate suppression via stamp array, not Python sets
    """

    __slots__ = (
        "cell_size",
        "inv_cell_size",
        "body_capacity",
        "max_cells_per_body",
        "max_pairs",
        "_bucket_count",
        "_bucket_mask",
        "_bucket_heads",
        "_bucket_stamps",
        "_entry_body",
        "_entry_next",
        "_entry_ix",
        "_entry_iy",
        "_entry_iz",
        "_entry_capacity",
        "_entry_count",
        "_body_ix0",
        "_body_iy0",
        "_body_iz0",
        "_body_ix1",
        "_body_iy1",
        "_body_iz1",
        "_query_seen",
        "_frame_stamp",
        "_query_stamp",
        "_pair_scratch",
    )

    def __init__(
        self,
        body_capacity,
        cell_size,
        bucket_count=None,
        max_cells_per_body=8,
        max_pairs=None,
    ):
        self.body_capacity = int(body_capacity)
        self.cell_size = float(cell_size)
        self.inv_cell_size = 1.0 / self.cell_size
        self.max_cells_per_body = int(max_cells_per_body)

        if bucket_count is None:
            # Low enough occupancy for ~1000 bodies without going huge.
            bucket_count = _next_pow2(max(1024, self.body_capacity * 8))
        else:
            bucket_count = _next_pow2(bucket_count)

        self._bucket_count = int(bucket_count)
        self._bucket_mask = self._bucket_count - 1
        self._bucket_heads = [-1] * self._bucket_count
        self._bucket_stamps = [0] * self._bucket_count

        self._entry_capacity = self.body_capacity * self.max_cells_per_body
        self._entry_count = 0

        self._entry_body = [0] * self._entry_capacity
        self._entry_next = [0] * self._entry_capacity
        self._entry_ix = [0] * self._entry_capacity
        self._entry_iy = [0] * self._entry_capacity
        self._entry_iz = [0] * self._entry_capacity

        self._body_ix0 = [0] * self.body_capacity
        self._body_iy0 = [0] * self.body_capacity
        self._body_iz0 = [0] * self.body_capacity
        self._body_ix1 = [0] * self.body_capacity
        self._body_iy1 = [0] * self.body_capacity
        self._body_iz1 = [0] * self.body_capacity

        self._query_seen = [0] * self.body_capacity

        self._frame_stamp = 1
        self._query_stamp = 1

        if max_pairs is None:
            # Conservative but still linear-ish for box piles / near-uniform density.
            max_pairs = max(1024, self.body_capacity * 64)
        self.max_pairs = int(max_pairs)
        self._pair_scratch = [CandidatePair(0, 0) for _ in range(self.max_pairs)]

    def pair_capacity_hint(self, body_count=None):
        return self.max_pairs

    def _hash_cell(self, ix: int, iy: int, iz: int) -> int:
        return (
            ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791))
            & self._bucket_mask
        )

    def _compute_body_cell_range(self, aabb):
        inv = self.inv_cell_size
        ix0 = math.floor(aabb.min_v.x * inv)
        iy0 = math.floor(aabb.min_v.y * inv)
        iz0 = math.floor(aabb.min_v.z * inv)
        ix1 = math.floor(aabb.max_v.x * inv)
        iy1 = math.floor(aabb.max_v.y * inv)
        iz1 = math.floor(aabb.max_v.z * inv)
        return ix0, iy0, iz0, ix1, iy1, iz1

    def _rebuild_grid(self, bodies, body_caches):
        n = len(bodies)
        if n > self.body_capacity:
            raise ValueError(
                f"SpatialGridBroadPhase body capacity exceeded: {n} > {self.body_capacity}"
            )

        self._frame_stamp += 1
        frame_stamp = self._frame_stamp
        self._entry_count = 0

        for body_idx in range(n):
            aabb = body_caches[body_idx].aabb
            ix0, iy0, iz0, ix1, iy1, iz1 = self._compute_body_cell_range(aabb)

            self._body_ix0[body_idx] = ix0
            self._body_iy0[body_idx] = iy0
            self._body_iz0[body_idx] = iz0
            self._body_ix1[body_idx] = ix1
            self._body_iy1[body_idx] = iy1
            self._body_iz1[body_idx] = iz1

            nx = ix1 - ix0 + 1
            ny = iy1 - iy0 + 1
            nz = iz1 - iz0 + 1
            span = nx * ny * nz

            if span > self.max_cells_per_body:
                raise ValueError(
                    "SpatialGridBroadPhase max_cells_per_body too small: "
                    f"body {body_idx} covers {span} cells, limit {self.max_cells_per_body}. "
                    "Increase cell_size or max_cells_per_body."
                )

            if self._entry_count + span > self._entry_capacity:
                raise ValueError(
                    "SpatialGridBroadPhase entry capacity exceeded. "
                    "Increase max_cells_per_body or body_capacity."
                )

            for iz in range(iz0, iz1 + 1):
                for iy in range(iy0, iy1 + 1):
                    for ix in range(ix0, ix1 + 1):
                        h = self._hash_cell(ix, iy, iz)

                        if self._bucket_stamps[h] != frame_stamp:
                            self._bucket_stamps[h] = frame_stamp
                            self._bucket_heads[h] = -1

                        entry = self._entry_count
                        self._entry_count += 1

                        self._entry_body[entry] = body_idx
                        self._entry_ix[entry] = ix
                        self._entry_iy[entry] = iy
                        self._entry_iz[entry] = iz
                        self._entry_next[entry] = self._bucket_heads[h]
                        self._bucket_heads[h] = entry

    def fill_pairs_fixed(self, bodies, body_caches, out_pairs):
        n = len(bodies)
        self._rebuild_grid(bodies, body_caches)

        pair_count = 0
        frame_stamp = self._frame_stamp

        for body_a in range(n):
            ai = body_caches[body_a].aabb

            self._query_stamp += 1
            stamp = self._query_stamp

            ix0 = self._body_ix0[body_a]
            iy0 = self._body_iy0[body_a]
            iz0 = self._body_iz0[body_a]
            ix1 = self._body_ix1[body_a]
            iy1 = self._body_iy1[body_a]
            iz1 = self._body_iz1[body_a]

            for iz in range(iz0, iz1 + 1):
                for iy in range(iy0, iy1 + 1):
                    for ix in range(ix0, ix1 + 1):
                        h = self._hash_cell(ix, iy, iz)
                        if self._bucket_stamps[h] != frame_stamp:
                            continue

                        entry = self._bucket_heads[h]
                        while entry != -1:
                            if (
                                self._entry_ix[entry] == ix
                                and self._entry_iy[entry] == iy
                                and self._entry_iz[entry] == iz
                            ):
                                body_b = self._entry_body[entry]
                                if body_b > body_a and self._query_seen[body_b] != stamp:
                                    self._query_seen[body_b] = stamp

                                    if aabb_overlap(ai, body_caches[body_b].aabb):
                                        if pair_count >= len(out_pairs):
                                            raise ValueError(
                                                "Candidate pair buffer too small. "
                                                "Increase broadphase max_pairs / solver pair capacity."
                                            )
                                        pair = out_pairs[pair_count]
                                        pair.body_a = body_a
                                        pair.body_b = body_b
                                        pair_count += 1

                            entry = self._entry_next[entry]

        return pair_count

    def fill_pairs(self, bodies, body_caches, out_pairs):
        # Fallback compatibility path; this allocates CandidatePair objects.
        out_pairs.clear()
        count = self.fill_pairs_fixed(bodies, body_caches, self._pair_scratch)
        for i in range(count):
            p = self._pair_scratch[i]
            out_pairs.append(CandidatePair(p.body_a, p.body_b))