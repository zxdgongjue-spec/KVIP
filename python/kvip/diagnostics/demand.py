# SPDX-License-Identifier: Apache-2.0
"""Selection demand against the frozen pre-call residency, independent of I/O.

Deduplicate the selected logical IDs across each request's four queries. The
same ID in another request remains a separate payload. ABC takes precedence
over hot residency, as in the serving maps and DecodeOracle.
"""

import numpy as np


DEMAND_ACCOUNTING = {
    "unit": "logical token payloads, union across Q4 within each request, summed across requests",
    "denominator": "selected_unique: selected per-request Q4 union, including initial hot and ABC hits",
    "cold_demand": "selected payload absent from both initial valid hot slots and initial ABC",
    "candidate_hot_refetch": (
        "selected initial-hot payload in an LRU candidate slot; required refetch in serial LIDU, "
        "potential refetch demand in fusion, not an observed transfer counter"
    ),
    "cold_miss_rate": "cold_demand / selected_unique (fraction, not percent)",
    "hot_hit": "selected initial valid hot payload outside the LRU candidate slots and ABC",
    "abc_hit": "selected initial ABC payload; takes precedence over any hot copy",
    "selected_read_demand": "cold_demand + candidate_hot_refetch; potential demand, not fusion's observed I/O",
    "transfer_accounting": (
        "prefetch + final counts actual transfers; useful_transfers subtracts unused prefetch, "
        "and may include initially hot payloads; neither is pure cold demand"
    ),
}

_COUNTS = ("selected_unique", "cold_demand", "candidate_hot_refetch", "hot_hit", "abc_hit")


def classify_initial_slots(slots, candidates, resident):
    """Classify one request's unique selection using its pre-call slot map."""
    slots, candidates = np.asarray(slots), np.asarray(candidates)
    cold = slots < 0
    abc = slots >= resident
    candidate = ~cold & ~abc & np.isin(slots, candidates)
    return dict(
        selected_unique=int(slots.size),
        cold_demand=int(np.count_nonzero(cold)),
        candidate_hot_refetch=int(np.count_nonzero(candidate)),
        hot_hit=int(np.count_nonzero(~cold & ~abc & ~candidate)),
        abc_hit=int(np.count_nonzero(abc)),
    )


def sum_demand(requests):
    """Aggregate request unions without treating repeated Q entries as misses."""
    result = {key: sum(row[key] for row in requests) for key in _COUNTS}
    result["selected_read_demand"] = result["cold_demand"] + result["candidate_hot_refetch"]
    result["cold_miss_rate"] = result["cold_demand"] / result["selected_unique"] if result["selected_unique"] else 0.0
    return result


class SnapshotDemand:
    """Precompute initial logical-to-slot maps; classify actual TND outputs."""

    def __init__(self, archive):
        geometry = archive.manifest["geometry"]
        batch, history, resident, stride = (
            geometry[name] for name in ("batch", "history", "resident", "output_stride")
        )
        ids = archive.array("pre.logical_ids")
        hot = archive.array("pre.hot_valid")
        metadata = archive.array("pre.metadata")
        bounds = archive.array("pre.tail_bounds")
        candidates = archive.array("pre.lru_slots")
        if (
            ids.ndim != 2
            or ids.shape[1] != history
            or hot.shape != (len(ids), resident)
            or metadata.shape != (5, batch * 4)
            or metadata.dtype != np.int64
            or bounds.shape != (batch, 2)
            or candidates.ndim != 2
            or candidates.shape[0] != batch
            or stride <= resident
        ):
            raise ValueError("Initial demand residency/metadata geometry differs")
        active = metadata[3] != 0
        sources = metadata[2, active]
        if np.any(sources < 0) or np.any(sources >= batch * 4) or len(np.unique(sources)) != len(sources):
            raise ValueError("Active TND outputs must name distinct legal packed queries")
        owners = metadata[4].view(np.int32)[:batch]
        packed_rows = np.unique(sources // 4)
        if (
            np.any(owners[packed_rows] <= 0)
            or np.any(owners[packed_rows] >= len(ids))
            or len(np.unique(owners[packed_rows])) != len(packed_rows)
        ):
            raise ValueError("Demand classification requires distinct live request owners")
        self.active, self.resident, self.history = active, resident, history
        self.requests = []
        self.prefetch_requests = {}
        identity = np.arange(history)
        for row in packed_rows:
            owner = owners[row]
            if not np.array_equal(np.sort(ids[owner]), identity):
                raise ValueError("Initial logical IDs must be a complete permutation")
            begin, end = map(int, bounds[row])
            if not 0 <= begin <= end <= history or end - begin > stride - resident:
                raise ValueError("Initial ABC bounds are outside the resident ring")
            lru = candidates[row]
            if np.any(lru < 0) or np.any(lru >= resident) or len(np.unique(lru)) != len(lru):
                raise ValueError("Initial LRU candidates must be distinct hot slots")
            forward = np.full(history, -1, dtype=np.int32)
            physical = np.flatnonzero(hot[owner] != 0)
            forward[ids[owner, physical]] = physical
            abc = np.arange(begin, end)
            forward[abc] = resident + abc % (stride - resident)
            tnd_rows = np.flatnonzero(active & (metadata[2] // 4 == row))
            self.requests.append((tnd_rows, forward, lru))
            self.prefetch_requests[int(row)] = (ids[owner], forward, lru)

    def classify_prefetch(self, mapping, selected):
        """Classify completed prefetch mappings, using pre-call residency.

        Mapping/selected coordinates address the original physical Index;
        convert through that owner's frozen logical IDs before classifying.
        The final resolver does not rewrite these speculative mappings.
        """
        mapping, selected = np.asarray(mapping), np.asarray(selected)
        if mapping.ndim != 2 or mapping.shape[1] != self.history or selected.shape[0] != len(mapping):
            raise ValueError("Prefetch mapping and selected rows differ from captured geometry")
        categories = ("cold", "candidate_hot", "other_hot", "abc")
        result = {prefix + category: 0 for prefix in ("prefetch_", "unused_prefetch_") for category in categories}
        for row, values in enumerate(mapping):
            fetched = np.flatnonzero((values >= self.resident) & (values < self.resident + 2048))
            if not len(fetched):
                continue
            if row not in self.prefetch_requests:
                raise ValueError("Inactive request contains speculative transfers")
            ids, forward, lru = self.prefetch_requests[row]
            slots = forward[ids[fetched]]
            cold, abc = slots < 0, slots >= self.resident
            candidate = ~cold & ~abc & np.isin(slots, lru)
            flags = (cold, candidate, ~cold & ~abc & ~candidate, abc)
            unused = ~np.isin(fetched, selected[row].reshape(-1))
            for category, mask in zip(categories, flags):
                result["prefetch_" + category] += int(np.count_nonzero(mask))
                result["unused_prefetch_" + category] += int(np.count_nonzero(mask & unused))
        result["prefetch_classified"] = sum(result["prefetch_" + category] for category in categories)
        return result

    def classify(self, logical):
        logical = np.asarray(logical)
        if logical.size % len(self.active):
            raise ValueError("Logical output does not match TND query capacity")
        logical = logical.reshape(len(self.active), -1)
        if np.any(logical[~self.active] != -1):
            raise ValueError("Inactive TND outputs must remain padding")
        requests = []
        for rows, forward, candidates in self.requests:
            selected = np.unique(logical[rows])
            if np.any(selected < 0) or np.any(selected >= self.history):
                raise ValueError("Selected logical ID is outside captured history")
            requests.append(classify_initial_slots(forward[selected], candidates, self.resident))
        return sum_demand(requests)


def snapshot_demand(archive):
    """CPU-only demand for the fusion selection saved in a service capture."""
    return SnapshotDemand(archive).classify(archive.array("post.logical_output"))
