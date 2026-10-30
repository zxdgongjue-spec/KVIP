"""CPU contract simulation: no CANN, network or performance claims."""

import numpy as np
from contract_index_reference import ColdOrder
from contract_payload import payload
from contract_reference import (
    ContractError,
    address,
    commit_plan,
    eligible_victims,
    frozen_hot,
    plan_row,
    select_topk,
    validate_equal_values,
)
from kvop.v9.contract import TOPK, Status


def seed_output(arrays, g):
    output = arrays["resident_output"].reshape(-1, 656)
    for b, p in enumerate(arrays["pool_rows"]):
        budget = int(arrays["budgets"][b])
        slots = np.flatnonzero(arrays["reverse"][p, :budget] >= 0)
        tokens = arrays["reverse"][p, slots]
        physical = arrays["resident_table"][b, slots // 128] * 128 + slots % 128
        output[physical] = payload(int(p) * g.num_layers + g.layer_id, tokens)
        c, logical = int(arrays["candidate_lens"][b]), int(arrays["logical_lens"][b])
        slots = np.arange(budget, budget + logical - c)
        physical = arrays["resident_table"][b, slots // 128] * 128 + slots % 128
        output[physical] = payload(int(p) * g.num_layers + g.layer_id, np.arange(c, logical))


def admission(scores, hot, q, limit, window_limit, rounds, cold_ids=None):
    """Reference window service; all producer rows ready at each chosen boundary."""
    c = scores.shape[-1]
    cold = [t for t in range(c) if t not in hot] if cold_ids is None else cold_ids
    pages = max(1, (len(cold) + 127) // 128)
    rounds = min(rounds, pages)
    threshold = []
    for i in range(q):
        valid = [t for t in hot if t < c - q + i + 1]
        threshold.append(np.partition(scores[i, valid], -TOPK)[-TOPK] if len(valid) >= TOPK else -np.inf)
    result, consumed = [], 0
    for epoch in range(rounds):
        end = min(len(cold), pages * (epoch + 1) // rounds * 128)
        eligible = [
            t
            for t in cold[consumed:end]
            if t not in hot and any(t < c - q + i + 1 and scores[i, t] >= threshold[i] for i in range(q))
        ]
        room = min(window_limit, limit - len(result))
        result.extend(eligible[:room])
        consumed = end
    return result


class Simulator:
    def __init__(self, geometry, arrays):
        self.g, self.arrays = geometry, arrays
        self.sticky = False
        self.layouts = {}
        self.trace = []

    def launch(self, scores, *, prefetch_limit=2048, window_limit=480, rounds=32, failure=None):
        g, a = self.g, self.arrays
        for name in ("topk", "destination", "extra_prefetch_slots"):
            a[name].fill(-1)
        for name in ("miss_counts", "resident_lengths", "transfers", "extra_prefetch_counts"):
            a[name].fill(0)
        a["status"].fill(Status.PENDING)
        self.trace = []
        if self.sticky:
            a["status"].fill(Status.POISONED)
            return
        try:
            validate_equal_values(a, g)
            if a["profile"][0] == 2 or np.all(a["row_modes"] == 0):
                a["status"].fill(Status.SKIPPED)
                return
            # All directory entries are preflighted before any transfer.
            for b in range(g.batch):
                for t in range(0, int(a["candidate_lens"][b]), 128):
                    address(a, g, b, t, 0)
        except ContractError as error:
            a["status"].fill(error.status)
            self.sticky = True
            return
        for b, p in enumerate(a["pool_rows"]):
            c, budget = int(a["candidate_lens"][b]), int(a["budgets"][b])
            hot = frozen_hot(a["forward"][p], a["reverse"][p], a["victims"][b], c, budget)
            selected = select_topk(scores[b], g.queries)
            a["topk"].reshape(g.batch, g.queries, TOPK)[b] = selected
            layout = self.layouts.setdefault(int(p), ColdOrder())
            cold = layout.reconcile(hot, c)
            eligible = eligible_victims(scores[b], hot, a["victims"][b])
            admitted = admission(
                scores[b], hot, g.queries, min(prefetch_limit, len(eligible)), window_limit, rounds, cold
            )
            reservations = dict(zip(admitted, eligible))
            trace = dict(reservations=reservations, descriptors=[], hot=hot, completed=False)
            self.trace.append(trace)
            output = a["resident_output"].reshape(-1, 656)
            for token, slot in reservations.items():
                d, off, u, remote = address(a, g, b, token, slot)
                output[u] = payload(int(p) * g.num_layers + g.layer_id, [token])[0]
                trace["descriptors"].append((d, off, u, remote))
            a["transfers"][b, 0] = len(reservations)
            try:
                if failure == "prefetch_cq":
                    raise ContractError(Status.COMMUNICATION, "injected prefetch CQ failure")
                plan = plan_row(selected, hot, a["victims"][b], budget, int(a["cursors"][p]), reservations)
                trace["plan"] = plan
                for token, slot in plan.supplement.items():
                    d, off, u, remote = address(a, g, b, token, slot)
                    output[u] = payload(int(p) * g.num_layers + g.layer_id, [token])[0]
                    trace["descriptors"].append((d, off, u, remote))
                a["transfers"][b, 1] = len(plan.supplement)
                if failure == "final_cq":
                    raise ContractError(Status.COMMUNICATION, "injected final CQ failure")
                a["cursors"][p] = commit_plan(a["forward"][p], a["reverse"][p], plan, reservations)
                a["destination"].reshape(g.batch, g.queries, TOPK)[b] = plan.destination
                a["miss_counts"][b] = len(plan.misses)
                a["resident_lengths"][b] = budget + int(a["logical_lens"][b]) - c
                a["transfers"][b, 3] = len(plan.discarded)
                for rank, (token, slot) in enumerate(reservations.items()):
                    if token in plan.extra:
                        a["extra_prefetch_slots"][b, rank] = slot
                a["extra_prefetch_counts"][b] = len(plan.extra)
                a["status"][b] = Status.OK
                trace["completed"] = True
                layout.reconcile(frozen_hot(a["forward"][p], a["reverse"][p], a["victims"][b], c, budget), c)
            except ContractError as error:
                a["status"][b] = error.status
                a["miss_counts"][b] = -1
                self.sticky = True
            a["transfers"][b, 2] = a["transfers"][b, :2].sum()
        a["miss_counts"][-1] = a["miss_counts"][:-1][a["miss_counts"][:-1] >= 0].sum()
