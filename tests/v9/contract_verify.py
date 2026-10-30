"""Independent acceptance checks shared by NPU and CPU benchmark runners."""

import numpy as np
from contract_payload import payload
from contract_reference import address, commit_plan, eligible_victims, frozen_hot, plan_row, verify_selection
from kvop.v9.contract import TOPK, Status


def snapshot(arrays):
    return {
        name: arrays[name].copy()
        for name in (
            "query",
            "key",
            "key_scale",
            "weights",
            "query_scale",
            "cumulative_query",
            "forward",
            "reverse",
            "cursors",
            "resident_output",
            "victims",
            "victim_bitmap",
            "pool_rows",
            "budgets",
            "candidate_lens",
            "logical_lens",
            "resident_table",
            "index_table",
            "valid_table",
            "address_table",
            "profile",
            "row_modes",
        )
    }


def verify_success(g, before, after, scores, reservations, *, descriptor_rows=None):
    expected = {name: before[name].copy() for name in ("forward", "reverse", "cursors", "resident_output")}
    counts, query_rates, scan_stats = [], [], []
    selected = after["topk"].reshape(g.batch, g.queries, TOPK)
    exported = after["destination"].reshape(g.batch, g.queries, TOPK)
    for b, p in enumerate(before["pool_rows"]):
        if after["status"][b] != Status.OK:
            raise AssertionError(f"request {b}: non-success {after['status'][b]}")
        try:
            verify_selection(selected[b], scores[b])
        except AssertionError as error:
            raise AssertionError(f"request={b}, pool={int(p)}: {error}") from error
        c, budget = int(before["candidate_lens"][b]), int(before["budgets"][b])
        hot = frozen_hot(before["forward"][p], before["reverse"][p], before["victims"][b], c, budget)
        reserve = reservations[b]
        eligible = eligible_victims(scores[b], hot, before["victims"][b])
        if list(reserve.values()) != eligible[: len(reserve)]:
            actual = list(reserve.values())
            rank = next(i for i, slot in enumerate(actual) if i >= len(eligible) or slot != eligible[i])
            wanted = eligible[rank] if rank < len(eligible) else None
            token = int(before["reverse"][p, wanted]) if wanted is not None else -1
            owner = int(before["forward"][p, token]) if 0 <= token < c else None
            raise AssertionError(
                f"prefetch did not use the filtered victim order: request={b}, pool={p}, "
                f"rank={rank}, actual_slot={actual[rank]}, expected_slot={wanted}, "
                f"expected_slot_reverse={token}, token_forward_owner={owner}, "
                f"reserved={len(actual)}, eligible={len(eligible)}"
            )
        if len(reserve) > TOPK or any(t in hot or not 0 <= t < c for t in reserve):
            raise AssertionError("prefetch included hot/invalid tokens or exceeded its reservation budget")
        # Require each admission to meet at least one causally visible query threshold.
        thresholds = []
        for q in range(g.queries):
            ids = [t for t in hot if t < c - g.queries + q + 1]
            thresholds.append(np.partition(scores[b, q, ids], -TOPK)[-TOPK] if len(ids) >= TOPK else -np.inf)
        candidates = np.zeros(c, dtype=bool)
        for q, cutoff in enumerate(thresholds):
            visible = max(0, c - g.queries + q + 1)
            candidates[:visible] |= scores[b, q, :visible] >= cutoff
        candidates[list(hot)] = False
        for token in reserve:
            if not any(
                token < c - g.queries + q + 1 and scores[b, q, token] >= thresholds[q] for q in range(g.queries)
            ):
                raise AssertionError("prefetch violated the frozen hot threshold")
        plan = plan_row(selected[b], hot, before["victims"][b], budget, int(before["cursors"][p]), reserve)
        extras = after["extra_prefetch_slots"][b]
        if set(map(int, extras[extras >= 0])) != set(plan.extra.values()) or after["extra_prefetch_counts"][b] != len(
            plan.extra
        ):
            raise AssertionError("retained extra prefetch feedback mismatch")
        expected["cursors"][p] = commit_plan(expected["forward"][p], expected["reverse"][p], plan, reserve)
        if not np.array_equal(exported[b], plan.destination):
            raise AssertionError("miss-first destination or logical slot mismatch")
        if after["resident_lengths"][b] != budget + int(before["logical_lens"][b]) - c:
            raise AssertionError("resident length/tail mismatch")
        transfers = [len(reserve), len(plan.supplement), len(reserve) + len(plan.supplement), len(plan.discarded)]
        if not np.array_equal(after["transfers"][b], transfers):
            raise AssertionError(f"transfer counters mismatch: {after['transfers'][b]} != {transfers}")
        out = expected["resident_output"].reshape(-1, 656)
        for writes in (reserve, plan.supplement):
            tokens = list(writes)
            physical = [address(before, g, b, t, writes[t])[2] for t in tokens]
            out[physical] = payload(int(p) * g.num_layers + g.layer_id, tokens)
        if descriptor_rows is not None:
            want = [address(before, g, b, t, s)[:3] for t, s in plan.supplement.items()]
            if sorted(map(tuple, descriptor_rows[b])) != sorted(want):
                raise AssertionError("directory/offset/physical destination descriptor mismatch")
        scan_stats.append(
            {
                "hot_tokens": len(hot),
                "cold_tokens": c - len(hot),
                "protected_victims": TOPK - len(eligible),
                "eligible_victims": len(eligible),
                "threshold_candidate_union": int(candidates.sum()),
                "prefetch_submitted_and_completed": len(reserve),
                "supplement_submitted_and_completed": len(plan.supplement),
                "retained_non_topk_prefetch": len(plan.extra),
                "discarded_prefetch": len(plan.discarded),
                "extra_feedback_count": int(after["extra_prefetch_counts"][b]),
            }
        )
        counts.append(len(plan.misses))
        query_rates.append(
            [
                sum(int(t) not in hot for t in row if t >= 0) / max(1, int(np.count_nonzero(row >= 0)))
                for row in selected[b]
            ]
        )
    if not np.array_equal(after["miss_counts"], counts + [sum(counts)]):
        raise AssertionError("unique logical miss counters mismatch")
    for name, value in expected.items():
        if not np.array_equal(after[name], value):
            raise AssertionError(f"{name}: publication, payload, tail or guard mismatch")
    return {
        "status": "pass",
        "payload": "all_bytes",
        "maps": "all_rows",
        "tail_and_guards": "unchanged",
        "logical_misses": counts,
        "per_request_scan_and_fetch": scan_stats,
        "per_query_miss_fraction": query_rates,
    }


def verify_skipped(before, after):
    if not np.all(after["status"] == Status.SKIPPED):
        raise AssertionError("dummy/PAD call was not skipped")
    for name in ("forward", "reverse", "cursors", "resident_output"):
        if not np.array_equal(before[name], after[name]):
            raise AssertionError(f"dummy/PAD changed {name}")
    for name in ("topk", "destination", "extra_prefetch_slots"):
        if not np.all(after[name] == -1):
            raise AssertionError(f"dummy/PAD did not initialize {name}")
    for name in ("miss_counts", "resident_lengths", "transfers", "extra_prefetch_counts"):
        if np.any(after[name]):
            raise AssertionError(f"dummy/PAD produced {name}")


def logical_score_workspace(workspace, hot_ids, cold_ids):
    """Untimed ABI 11 diagnostic conversion; use the frozen init directories."""
    physical = np.asarray(workspace)
    logical = np.zeros_like(physical)
    for b, (hot, cold) in enumerate(zip(hot_ids, cold_ids)):
        base = (len(hot) + 127) // 128 * 128
        logical[b][:, hot] = physical[b, :, : len(hot)]
        logical[b][:, cold] = physical[b, :, base : base + len(cold)]
    return logical


def score_workspace_diagnostic(workspace, reference):
    """Exact BF16 sortable-score check for the benchmark's finite fixtures.

    Run immediately after LI, before fetch reuses the score workspace as scratch.
    Only causally visible logical tokens participate. This is never timed.
    """
    actual = np.asarray(workspace).view(np.uint16)
    reference = np.asarray(reference, dtype=np.float32)
    rows = []
    for b in range(reference.shape[0]):
        for q in range(reference.shape[1]):
            visible = max(0, reference.shape[-1] - reference.shape[1] + q + 1)
            values = reference[b, q, :visible]
            bits = values.view(np.uint32)
            # Round-to-nearest-even BF16, then native monotonic float-sort keys.
            bf16 = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            expected = np.where(bf16 & 0x8000, ~bf16, bf16 ^ 0x8000).astype(np.uint16)
            bad = np.flatnonzero(actual[b, q, :visible] != expected)
            sample = bad[:8]
            rows.append(
                {
                    "request": b,
                    "query": q,
                    "visible": visible,
                    "mismatches": int(len(bad)),
                    "tokens": sample.tolist(),
                    "expected_scores": values[sample].tolist(),
                    "expected_bits": expected[sample].tolist(),
                    "actual_bits": actual[b, q, sample].tolist(),
                }
            )
    return {"status": "failed" if any(r["mismatches"] for r in rows) else "pass", "rows": rows}
