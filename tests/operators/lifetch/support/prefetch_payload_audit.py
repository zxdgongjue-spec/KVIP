"""Strict CPU-side P1-C0 audit for completed direct-URMA V9 prefetch.

This audit must run only after v9_wait and NPU stream synchronization.
Rows count tokens, whereas status.completed_batch_count counts completed
CQ groups. Logical mapping slots are not necessarily physical HBM slots:
for the first 2048 speculative slots, PhysicalSlot() dereferences lru_slots.
"""

from __future__ import annotations

import hashlib

import numpy as np


STATUS_MAGIC = 0x4B5641464247444F
LRU_SLOTS = 2048


def audit_prefetch_mapping(
    mapping,
    lru_slots,
    rows,
    status_words,
    key_lengths,
    *,
    resident,
    output_stride,
    rounds,
    selected_for_post,
    use_lru_slots=True,
):
    """Plan exact (request, token, logical slot, physical HBM slot) checks.

    Return JSON-serializable diagnostics even when an invariant fails.
    Callers must persist failures and MUST NOT treat an invalid CQ as safe
    to release an in-flight KVAF lease.
    """
    mapping = np.asarray(mapping)
    lru_slots = np.asarray(lru_slots)
    rows = np.asarray(rows)
    status_words = np.asarray(status_words)
    key_lengths = np.asarray(key_lengths, dtype=np.int64)
    if mapping.ndim != 2:
        raise ValueError("mapping must be [B,H]")
    batch, history = mapping.shape
    if (
        batch <= 0 or rows.shape != (batch, 256)
        or lru_slots.shape != (batch, LRU_SLOTS)
        or status_words.ndim != 2
        or status_words.shape[0] != batch * 2
        or status_words.shape[1] < 8
        or key_lengths.shape != (batch,)
        or not (0 < resident <= history)
        or output_stride <= resident
        or not (1 <= rounds <= 32)
        or rows.dtype.kind not in "iu"
        or mapping.dtype.kind not in "iu"
        or lru_slots.dtype.kind not in "iu"
        or status_words.dtype.kind not in "iu"
    ):
        raise ValueError("invalid P1-C0 mapping/row/status ABI shape")
    if np.any(key_lengths < 0) or np.any(key_lengths > history):
        raise ValueError("key_lengths outside history")

    report = {
        "schema": "kvip.prefetch_payload_audit.v1",
        "geometry": {
            "batch": int(batch), "history": int(history),
            "resident": int(resident), "output_stride": int(output_stride),
            "rounds": int(rounds), "use_lru_slots": bool(use_lru_slots),
        },
        "selected_for_post_event_total": (
            None if selected_for_post is None else int(selected_for_post)
        ),
        "rows": [],
        "errors": [],
    }
    total_requests = total_mapped = total_groups = 0
    global_owners = set()
    errors = report["errors"]

    for row in range(batch):
        requests = int(rows[row, 9])
        slots = int(rows[row, 8])
        suppressed = int(rows[row, 10])
        pending = int(rows[row, 211])
        row_error = int(rows[row, 11])
        epochs = [int(x) for x in rows[row, 16:16 + rounds]]
        epoch_suppressed = [int(x) for x in rows[row, 48:48 + rounds]]
        cq = [int(x) for x in status_words[2 * row, :8]]
        cq_unused = [int(x) for x in status_words[2 * row + 1, :8]]
        completed_groups = cq[3]
        token_ids = np.flatnonzero(mapping[row] >= 0)
        logical_slots = mapping[row, token_ids].astype(np.int64, copy=False)
        records = []
        seen_logical = set()
        seen_physical = set()

        if requests < 0 or slots < 0 or suppressed < 0:
            errors.append(f"row={row}: negative request/slot/suppressed")
        if slots != requests:
            errors.append(f"row={row}: slot count {slots} != request count {requests}")
        if requests != sum(epochs):
            errors.append(
                f"row={row}: sum(epochCount)={sum(epochs)} != requests={requests}"
            )
        if len(token_ids) != requests:
            errors.append(
                f"row={row}: mapping entries={len(token_ids)} != requests={requests}"
            )
        if row_error or pending:
            errors.append(f"row={row}: row_error={row_error} pending={pending}")
        if cq[0] != STATUS_MAGIC or cq[2] or cq[1] != requests:
            errors.append(
                f"row={row}: invalid prefetch CQ magic/result/processed: "
                f"{cq[:4]} expected requests={requests}"
            )
        if cq_unused[0] != STATUS_MAGIC or cq_unused[1] or cq_unused[2] or cq_unused[3]:
            errors.append(f"row={row}: odd final-owned status unexpectedly active")
        if (requests and not 0 < completed_groups <= requests) or (
            not requests and completed_groups
        ):
            errors.append(
                f"row={row}: completed CQ groups={completed_groups} "
                f"invalid for token requests={requests}"
            )

        for token, logical in zip(token_ids.tolist(), logical_slots.tolist()):
            physical = logical
            if use_lru_slots and resident <= logical < resident + LRU_SLOTS:
                ordinal = logical - resident
                physical = int(lru_slots[row, ordinal])
            if not (resident <= logical < output_stride):
                errors.append(f"row={row} token={token}: invalid logical slot {logical}")
            if not (0 <= physical < output_stride):
                errors.append(f"row={row} token={token}: invalid physical slot {physical}")
            if token >= int(key_lengths[row]):
                errors.append(
                    f"row={row} token={token}: exceeds key length {int(key_lengths[row])}"
                )
            if logical in seen_logical:
                errors.append(f"row={row}: logical slot collision {logical}")
            if physical in seen_physical:
                errors.append(f"row={row}: physical HBM slot collision {physical}")
            if (row, physical) in global_owners:
                errors.append(f"row={row}: physical slot reused {physical}")
            seen_logical.add(logical)
            seen_physical.add(physical)
            global_owners.add((row, physical))
            records.append({
                "token": int(token),
                "resident_token": bool(token < resident),
                "logical_slot": int(logical),
                "physical_slot": int(physical),
            })

        report["rows"].append({
            "row": row,
            "key_length": int(key_lengths[row]),
            "requests": requests,
            "slots": slots,
            "suppressed": suppressed,
            "epoch_count": epochs,
            "epoch_suppressed": epoch_suppressed,
            "mapping_count": len(token_ids),
            "resident_candidate_count": sum(x["resident_token"] for x in records),
            "cold_candidate_count": sum(not x["resident_token"] for x in records),
            "cq_processed_tokens": cq[1],
            "cq_completed_groups": completed_groups,
            "cq_pending_after_wait": pending,
            "mappings": records,
        })
        total_requests += requests
        total_mapped += len(token_ids)
        total_groups += completed_groups

    if selected_for_post is not None and total_requests != int(selected_for_post):
        errors.append(
            f"selectedForPost={selected_for_post} != Row.requests={total_requests}"
        )
    report["summary"] = {
        "requests": total_requests,
        "mapping_count": total_mapped,
        "cq_completed_groups": total_groups,
        "resident_candidate_count": sum(
            x["resident_candidate_count"] for x in report["rows"]
        ),
        "cold_candidate_count": sum(
            x["cold_candidate_count"] for x in report["rows"]
        ),
        "payload_tokens_checked": 0,
        "payload_bytes_checked": 0,
        "validation": "PASS" if not errors else "FAIL",
    }
    return report


def verify_prefetch_payload(plan, *, read_hbm, read_ddr, token_bytes):
    """Verify each mapped output token against the immutable DDR fixture.

    read_hbm(row, physical_slots) and read_ddr(row, token_ids) must return
    uint8[n,token_bytes] arrays. A caller must verify the CQ *before* use.
    """
    if plan["errors"]:
        raise ValueError("refuse payload reads when mapping/CQ audit failed")
    if token_bytes <= 0:
        raise ValueError("token_bytes must be positive")
    errors = plan["errors"]
    checked = 0
    total_bytes = 0

    for row in plan["rows"]:
        entries = row["mappings"]
        if not entries:
            row["payload"] = {"checked": 0, "bytes": 0}
            continue
        tokens = [record["token"] for record in entries]
        physical = [record["physical_slot"] for record in entries]
        actual = np.asarray(read_hbm(row["row"], physical))
        expected = np.asarray(read_ddr(row["row"], tokens))
        expected_shape = (len(entries), token_bytes)
        if (
            actual.dtype != np.uint8 or expected.dtype != np.uint8
            or actual.shape != expected_shape
            or expected.shape != expected_shape
        ):
            errors.append(
                f"row={row['row']}: bad payload dtype/shape "
                f"actual={actual.dtype}/{actual.shape} "
                f"expected={expected.dtype}/{expected.shape}"
            )
            continue
        mismatch = np.argwhere(actual != expected)
        if mismatch.size:
            index, byte_offset = (int(x) for x in mismatch[0])
            token = tokens[index]
            errors.append(
                f"row={row['row']} token={token} physical={physical[index]} "
                f"payload mismatch offset={byte_offset}: "
                f"HBM={int(actual[index, byte_offset])} "
                f"DDR={int(expected[index, byte_offset])}"
            )
        row["payload"] = {
            "checked": len(entries),
            "bytes": len(entries) * token_bytes,
            "hbm_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "ddr_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        }
        checked += len(entries)
        total_bytes += len(entries) * token_bytes

    plan["summary"]["payload_tokens_checked"] = checked
    plan["summary"]["payload_bytes_checked"] = total_bytes
    plan["summary"]["validation"] = "PASS" if not errors else "FAIL"
    return plan
