from ..part3.types import AABB, CandidatePair
from ..part3.broadphase import BroadPhaseBase
import math


def aabb_overlap(a: AABB, b: AABB):
    return (
        a.min_v.x <= b.max_v.x and a.max_v.x >= b.min_v.x and
        a.min_v.y <= b.max_v.y and a.max_v.y >= b.min_v.y and
        a.min_v.z <= b.max_v.z and a.max_v.z >= b.min_v.z
    )

class SweepAndPruneBroadPhase(BroadPhaseBase):
    __slots__ = (
        "body_capacity",
        "axis",
        "max_pairs",
        "_order",
        "_mins",
        "_maxs",
        "_pair_scratch",
    )

    def __init__(self, body_capacity, axis=-1, max_pairs=None):
        self.body_capacity = int(body_capacity)
        self.axis = int(axis)

        if max_pairs is None:
            max_pairs = max(1024, self.body_capacity * 64)
        self.max_pairs = int(max_pairs)

        self._order = list(range(self.body_capacity))
        self._mins = [0.0] * self.body_capacity
        self._maxs = [0.0] * self.body_capacity
        self._pair_scratch = [CandidatePair(0, 0) for _ in range(self.max_pairs)]

    def pair_capacity_hint(self, body_count=None):
        return self.max_pairs

    def _choose_axis(self, body_caches, n):
        if self.axis in (0, 1, 2):
            return self.axis

        min_x = float("inf")
        min_y = float("inf")
        min_z = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        max_z = float("-inf")

        for i in range(n):
            aabb = body_caches[i].aabb
            if aabb.min_v.x < min_x:
                min_x = aabb.min_v.x
            if aabb.min_v.y < min_y:
                min_y = aabb.min_v.y
            if aabb.min_v.z < min_z:
                min_z = aabb.min_v.z
            if aabb.max_v.x > max_x:
                max_x = aabb.max_v.x
            if aabb.max_v.y > max_y:
                max_y = aabb.max_v.y
            if aabb.max_v.z > max_z:
                max_z = aabb.max_v.z

        ex = max_x - min_x
        ey = max_y - min_y
        ez = max_z - min_z

        if ex >= ey and ex >= ez:
            return 0
        if ey >= ez:
            return 1
        return 2

    def fill_pairs_fixed(self, bodies, body_caches, out_pairs):
        n = len(bodies)
        if n > self.body_capacity:
            raise ValueError(
                f"SweepAndPruneBroadPhase body capacity exceeded: {n} > {self.body_capacity}"
            )

        axis = self._choose_axis(body_caches, n)
        mins = self._mins
        maxs = self._maxs
        order = self._order

        if axis == 0:
            for i in range(n):
                aabb = body_caches[i].aabb
                mins[i] = aabb.min_v.x
                maxs[i] = aabb.max_v.x
        elif axis == 1:
            for i in range(n):
                aabb = body_caches[i].aabb
                mins[i] = aabb.min_v.y
                maxs[i] = aabb.max_v.y
        else:
            for i in range(n):
                aabb = body_caches[i].aabb
                mins[i] = aabb.min_v.z
                maxs[i] = aabb.max_v.z

        for i in range(1, n):
            key = order[i]
            key_min = mins[key]
            j = i - 1
            while j >= 0 and mins[order[j]] > key_min:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key

        pair_count = 0
        for s in range(n):
            body_a = order[s]
            aabb_a = body_caches[body_a].aabb
            max_a = maxs[body_a]

            t = s + 1
            while t < n:
                body_b = order[t]
                if mins[body_b] > max_a:
                    break

                if aabb_overlap(aabb_a, body_caches[body_b].aabb):
                    if pair_count >= len(out_pairs):
                        raise ValueError(
                            "Candidate pair buffer too small. "
                            "Increase broadphase max_pairs / solver pair capacity."
                        )
                    pair = out_pairs[pair_count]
                    if body_a < body_b:
                        pair.body_a = body_a
                        pair.body_b = body_b
                    else:
                        pair.body_a = body_b
                        pair.body_b = body_a
                    pair_count += 1
                t += 1

        return pair_count

    def fill_pairs(self, bodies, body_caches, out_pairs):
        out_pairs.clear()
        count = self.fill_pairs_fixed(bodies, body_caches, self._pair_scratch)
        for i in range(count):
            p = self._pair_scratch[i]
            out_pairs.append(CandidatePair(p.body_a, p.body_b))


def _expand_bits_10(v: int) -> int:
    v &= 0x3FF
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def _morton3_10bit(x: float, y: float, z: float) -> int:
    xi = min(1023, max(0, int(x * 1023.0)))
    yi = min(1023, max(0, int(y * 1023.0)))
    zi = min(1023, max(0, int(z * 1023.0)))
    return (_expand_bits_10(xi) << 2) | (_expand_bits_10(yi) << 1) | _expand_bits_10(zi)


def _clz64(x: int) -> int:
    if x == 0:
        return 64
    return 64 - x.bit_length()


class LBVHBroadPhase(BroadPhaseBase):
    __slots__ = (
        "body_capacity",
        "max_pairs",
        "_node_capacity",
        "_work_capacity",
        "_sorted_keys",
        "_sorted_body",
        "_tmp_keys",
        "_tmp_body",
        "_counts",
        "_offsets",
        "_internal_left",
        "_internal_right",
        "_node_min_x",
        "_node_min_y",
        "_node_min_z",
        "_node_max_x",
        "_node_max_y",
        "_node_max_z",
        "_build_nodes",
        "_build_state",
        "_task_type",
        "_task_a",
        "_task_b",
        "_pair_scratch",
    )

    def __init__(self, body_capacity, max_pairs=None, traversal_work_factor=8):
        self.body_capacity = int(body_capacity)

        if max_pairs is None:
            max_pairs = max(1024, self.body_capacity * 64)
        self.max_pairs = int(max_pairs)

        self._node_capacity = max(1, 2 * self.body_capacity - 1)
        self._work_capacity = max(64, traversal_work_factor * self._node_capacity)

        self._sorted_keys = [0] * self.body_capacity
        self._sorted_body = [0] * self.body_capacity
        self._tmp_keys = [0] * self.body_capacity
        self._tmp_body = [0] * self.body_capacity
        self._counts = [0] * 256
        self._offsets = [0] * 256

        self._internal_left = [0] * max(1, self.body_capacity - 1)
        self._internal_right = [0] * max(1, self.body_capacity - 1)

        self._node_min_x = [0.0] * self._node_capacity
        self._node_min_y = [0.0] * self._node_capacity
        self._node_min_z = [0.0] * self._node_capacity
        self._node_max_x = [0.0] * self._node_capacity
        self._node_max_y = [0.0] * self._node_capacity
        self._node_max_z = [0.0] * self._node_capacity

        self._build_nodes = [0] * (2 * self._node_capacity)
        self._build_state = [0] * (2 * self._node_capacity)

        self._task_type = [0] * self._work_capacity
        self._task_a = [0] * self._work_capacity
        self._task_b = [0] * self._work_capacity

        self._pair_scratch = [CandidatePair(0, 0) for _ in range(self.max_pairs)]

    def pair_capacity_hint(self, body_count=None):
        return self.max_pairs

    def _delta(self, n, i, j):
        if j < 0 or j >= n:
            return -1
        return _clz64(self._sorted_keys[i] ^ self._sorted_keys[j])

    def _determine_range(self, n, idx):
        d_next = self._delta(n, idx, idx + 1)
        d_prev = self._delta(n, idx, idx - 1)
        direction = 1 if d_next >= d_prev else -1

        delta_min = self._delta(n, idx, idx - direction)

        l_max = 2
        while self._delta(n, idx, idx + l_max * direction) > delta_min:
            l_max <<= 1

        length = 0
        step = l_max >> 1
        while step > 0:
            new_length = length + step
            if self._delta(n, idx, idx + new_length * direction) > delta_min:
                length = new_length
            step >>= 1

        j = idx + length * direction
        if idx < j:
            return idx, j
        return j, idx

    def _find_split(self, first, last):
        first_code = self._sorted_keys[first]
        last_code = self._sorted_keys[last]

        common_prefix = _clz64(first_code ^ last_code)

        split = first
        step = last - first
        while step > 1:
            step = (step + 1) >> 1
            new_split = split + step
            if new_split < last:
                split_prefix = _clz64(first_code ^ self._sorted_keys[new_split])
                if split_prefix > common_prefix:
                    split = new_split
        return split

    def _radix_sort_64(self, n):
        src_keys = self._sorted_keys
        src_body = self._sorted_body
        dst_keys = self._tmp_keys
        dst_body = self._tmp_body
        counts = self._counts
        offsets = self._offsets

        for shift in range(0, 64, 8):
            for i in range(256):
                counts[i] = 0

            for i in range(n):
                counts[(src_keys[i] >> shift) & 0xFF] += 1

            total = 0
            for i in range(256):
                offsets[i] = total
                total += counts[i]

            for i in range(n):
                bucket = (src_keys[i] >> shift) & 0xFF
                dst_pos = offsets[bucket]
                dst_keys[dst_pos] = src_keys[i]
                dst_body[dst_pos] = src_body[i]
                offsets[bucket] = dst_pos + 1

            src_keys, dst_keys = dst_keys, src_keys
            src_body, dst_body = dst_body, src_body

        self._sorted_keys = src_keys
        self._tmp_keys = dst_keys
        self._sorted_body = src_body
        self._tmp_body = dst_body

    def _build_leaf_keys(self, n, body_caches):
        scene_min_x = float("inf")
        scene_min_y = float("inf")
        scene_min_z = float("inf")
        scene_max_x = float("-inf")
        scene_max_y = float("-inf")
        scene_max_z = float("-inf")

        for i in range(n):
            aabb = body_caches[i].aabb
            if aabb.min_v.x < scene_min_x:
                scene_min_x = aabb.min_v.x
            if aabb.min_v.y < scene_min_y:
                scene_min_y = aabb.min_v.y
            if aabb.min_v.z < scene_min_z:
                scene_min_z = aabb.min_v.z
            if aabb.max_v.x > scene_max_x:
                scene_max_x = aabb.max_v.x
            if aabb.max_v.y > scene_max_y:
                scene_max_y = aabb.max_v.y
            if aabb.max_v.z > scene_max_z:
                scene_max_z = aabb.max_v.z

        ext_x = max(scene_max_x - scene_min_x, 1.0e-9)
        ext_y = max(scene_max_y - scene_min_y, 1.0e-9)
        ext_z = max(scene_max_z - scene_min_z, 1.0e-9)

        for i in range(n):
            aabb = body_caches[i].aabb
            cx = 0.5 * (aabb.min_v.x + aabb.max_v.x)
            cy = 0.5 * (aabb.min_v.y + aabb.max_v.y)
            cz = 0.5 * (aabb.min_v.z + aabb.max_v.z)

            nx = (cx - scene_min_x) / ext_x
            ny = (cy - scene_min_y) / ext_y
            nz = (cz - scene_min_z) / ext_z

            morton = _morton3_10bit(nx, ny, nz)
            self._sorted_keys[i] = (morton << 32) | i
            self._sorted_body[i] = i

        self._radix_sort_64(n)

    def _build_hierarchy(self, n):
        if n <= 1:
            return

        leaf_base = 0
        internal_base = n

        for idx in range(n - 1):
            first, last = self._determine_range(n, idx)
            split = self._find_split(first, last)

            if split == first:
                left_id = leaf_base + split
            else:
                left_id = internal_base + split

            if split + 1 == last:
                right_id = leaf_base + (split + 1)
            else:
                right_id = internal_base + (split + 1)

            self._internal_left[idx] = left_id
            self._internal_right[idx] = right_id

    def _refit_bounds(self, n, body_caches):
        total_nodes = 2 * n - 1
        leaf_base = 0
        internal_base = n

        min_x = self._node_min_x
        min_y = self._node_min_y
        min_z = self._node_min_z
        max_x = self._node_max_x
        max_y = self._node_max_y
        max_z = self._node_max_z

        for leaf in range(n):
            body_idx = self._sorted_body[leaf]
            aabb = body_caches[body_idx].aabb
            min_x[leaf_base + leaf] = aabb.min_v.x
            min_y[leaf_base + leaf] = aabb.min_v.y
            min_z[leaf_base + leaf] = aabb.min_v.z
            max_x[leaf_base + leaf] = aabb.max_v.x
            max_y[leaf_base + leaf] = aabb.max_v.y
            max_z[leaf_base + leaf] = aabb.max_v.z

        if n == 1:
            return

        stack_nodes = self._build_nodes
        stack_state = self._build_state
        top = 0
        stack_nodes[0] = internal_base
        stack_state[0] = 0

        while top >= 0:
            node_id = stack_nodes[top]
            if node_id < internal_base:
                top -= 1
                continue

            internal_idx = node_id - internal_base
            state = stack_state[top]

            if state == 0:
                stack_state[top] = 1

                left_id = self._internal_left[internal_idx]
                right_id = self._internal_right[internal_idx]

                top += 1
                stack_nodes[top] = right_id
                stack_state[top] = 0

                top += 1
                stack_nodes[top] = left_id
                stack_state[top] = 0
            else:
                left_id = self._internal_left[internal_idx]
                right_id = self._internal_right[internal_idx]

                min_x[node_id] = min(min_x[left_id], min_x[right_id])
                min_y[node_id] = min(min_y[left_id], min_y[right_id])
                min_z[node_id] = min(min_z[left_id], min_z[right_id])
                max_x[node_id] = max(max_x[left_id], max_x[right_id])
                max_y[node_id] = max(max_y[left_id], max_y[right_id])
                max_z[node_id] = max(max_z[left_id], max_z[right_id])

                top -= 1

        _ = total_nodes  # silence intent; useful when debugging capacities

    def _node_overlap(self, a_id, b_id):
        return (
            self._node_min_x[a_id] <= self._node_max_x[b_id] and self._node_max_x[a_id] >= self._node_min_x[b_id] and
            self._node_min_y[a_id] <= self._node_max_y[b_id] and self._node_max_y[a_id] >= self._node_min_y[b_id] and
            self._node_min_z[a_id] <= self._node_max_z[b_id] and self._node_max_z[a_id] >= self._node_min_z[b_id]
        )

    def _node_surface_area(self, node_id):
        ex = self._node_max_x[node_id] - self._node_min_x[node_id]
        ey = self._node_max_y[node_id] - self._node_min_y[node_id]
        ez = self._node_max_z[node_id] - self._node_min_z[node_id]
        return 2.0 * (ex * ey + ey * ez + ez * ex)

    def _push_task(self, top, task_type, a, b):
        top += 1
        if top >= self._work_capacity:
            raise ValueError(
                "LBVHBroadPhase traversal stack overflow. "
                "Increase traversal_work_factor."
            )
        self._task_type[top] = task_type
        self._task_a[top] = a
        self._task_b[top] = b
        return top

    def _traverse_pairs(self, n, out_pairs):
        if n <= 1:
            return 0

        leaf_base = 0
        internal_base = n
        root_id = internal_base

        top = 0
        self._task_type[0] = 0
        self._task_a[0] = root_id
        self._task_b[0] = -1

        pair_count = 0

        while top >= 0:
            task_type = self._task_type[top]
            a_id = self._task_a[top]
            b_id = self._task_b[top]
            top -= 1

            if task_type == 0:
                if a_id < internal_base:
                    continue

                internal_idx = a_id - internal_base
                left_id = self._internal_left[internal_idx]
                right_id = self._internal_right[internal_idx]

                top = self._push_task(top, 1, left_id, right_id)

                if right_id >= internal_base:
                    top = self._push_task(top, 0, right_id, -1)
                if left_id >= internal_base:
                    top = self._push_task(top, 0, left_id, -1)

                continue

            if not self._node_overlap(a_id, b_id):
                continue

            a_is_leaf = a_id < internal_base
            b_is_leaf = b_id < internal_base

            if a_is_leaf and b_is_leaf:
                body_a = self._sorted_body[a_id - leaf_base]
                body_b = self._sorted_body[b_id - leaf_base]

                if body_a == body_b:
                    continue

                if pair_count >= len(out_pairs):
                    raise ValueError(
                        "Candidate pair buffer too small. "
                        "Increase broadphase max_pairs / solver pair capacity."
                    )

                pair = out_pairs[pair_count]
                if body_a < body_b:
                    pair.body_a = body_a
                    pair.body_b = body_b
                else:
                    pair.body_a = body_b
                    pair.body_b = body_a
                pair_count += 1
                continue

            if a_is_leaf:
                b_internal_idx = b_id - internal_base
                top = self._push_task(top, 1, a_id, self._internal_right[b_internal_idx])
                top = self._push_task(top, 1, a_id, self._internal_left[b_internal_idx])
                continue

            if b_is_leaf:
                a_internal_idx = a_id - internal_base
                top = self._push_task(top, 1, self._internal_right[a_internal_idx], b_id)
                top = self._push_task(top, 1, self._internal_left[a_internal_idx], b_id)
                continue

            if self._node_surface_area(a_id) >= self._node_surface_area(b_id):
                a_internal_idx = a_id - internal_base
                top = self._push_task(top, 1, self._internal_right[a_internal_idx], b_id)
                top = self._push_task(top, 1, self._internal_left[a_internal_idx], b_id)
            else:
                b_internal_idx = b_id - internal_base
                top = self._push_task(top, 1, a_id, self._internal_right[b_internal_idx])
                top = self._push_task(top, 1, a_id, self._internal_left[b_internal_idx])

        return pair_count

    def fill_pairs_fixed(self, bodies, body_caches, out_pairs):
        n = len(bodies)
        if n > self.body_capacity:
            raise ValueError(
                f"LBVHBroadPhase body capacity exceeded: {n} > {self.body_capacity}"
            )
        if n <= 1:
            return 0

        self._build_leaf_keys(n, body_caches)
        self._build_hierarchy(n)
        self._refit_bounds(n, body_caches)
        return self._traverse_pairs(n, out_pairs)

    def fill_pairs(self, bodies, body_caches, out_pairs):
        out_pairs.clear()
        count = self.fill_pairs_fixed(bodies, body_caches, self._pair_scratch)
        for i in range(count):
            p = self._pair_scratch[i]
            out_pairs.append(CandidatePair(p.body_a, p.body_b))