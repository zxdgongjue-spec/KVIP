"""Host caller lifecycle driver; all mutations occur outside fused timing/capture."""

import numpy as np
from contract_payload import payload
from contract_reference import bitmap
from kvop.v9.contract import TOPK


class Lifecycle:
    def __init__(self, geometry):
        self.g = geometry
        self.ages = np.zeros((geometry.pool_capacity, geometry.resident_capacity), np.int64)
        self.generations = np.zeros(geometry.pool_capacity, np.int64)
        self.clock = 0

    def prepare(self, arrays, scores, step):
        g = self.g
        event = {"step": step, "restore_slots": 0, "rebound": None}
        if step:
            # Shuffle packed requests while preserving each pool's maps/payload.
            order = np.roll(np.arange(g.batch), 1)
            for name in ("pool_rows", "budgets", "resident_table", "index_table", "row_modes"):
                arrays[name][:] = arrays[name][order].copy()
            for name in ("query", "weights", "query_scale"):
                rows = arrays[name].reshape(g.batch, g.queries, *arrays[name].shape[1:])
                rows[:] = np.roll(rows[order].copy(), 1, axis=1)
            scores[:] = np.roll(scores[order].copy(), 1, axis=1)
        if step == 2:
            # A caller may release/rebind only after the prior completion gate.
            old = int(arrays["pool_rows"][0])
            new = next(p for p in range(g.max_requests) if p not in arrays["pool_rows"])
            for p in (old, new):
                arrays["forward"][p].fill(-1)
                arrays["reverse"][p].fill(-1)
                arrays["cursors"][p] = 0
                self.ages[p].fill(0)
                self.generations[p] += 1
            arrays["pool_rows"][0] = new
            arrays["resident_output"][arrays["resident_table"][0]] = 0xA5
            event["rebound"] = {"old_pool": old, "new_pool": new}
        if step == 1:
            # Model a completed pending restore before selection. Poison then
            # restore an existing prefix, so verification cannot accept stale KV.
            for b, p in enumerate(arrays["pool_rows"]):
                slots = np.flatnonzero(arrays["reverse"][p] >= 0)[:16]
                tokens = arrays["reverse"][p, slots].copy()
                physical = arrays["resident_table"][b, slots // 128] * 128 + slots % 128
                out = arrays["resident_output"].reshape(-1, 656)
                out[physical] = 0xDE
                out[physical] = payload(int(p) * g.num_layers + g.layer_id, tokens)
                event["restore_slots"] += len(slots)
        for b, p in enumerate(arrays["pool_rows"]):
            budget = int(arrays["budgets"][b])
            # Every final slot was touched by commit(); unused prefetch was not.
            order = np.lexsort((np.arange(budget), self.ages[p, :budget]))
            arrays["victims"][b] = order[:TOPK]
            arrays["victim_bitmap"][b] = bitmap(order[:TOPK])
        event["pool_rows"] = arrays["pool_rows"].tolist()
        event["generations"] = self.generations[arrays["pool_rows"]].tolist()
        return event

    def commit(self, arrays):
        if np.any(arrays["status"] != 0):
            raise ValueError("LRU feedback requires successful checked completion")
        self.clock += 2
        rows = arrays["destination"].reshape(self.g.batch, -1)
        for b, p in enumerate(arrays["pool_rows"]):
            slots = np.unique(rows[b][rows[b] >= 0])
            extras = arrays["extra_prefetch_slots"][b]
            self.ages[p, extras[extras >= 0]] = self.clock - 1
            self.ages[p, slots] = self.clock
