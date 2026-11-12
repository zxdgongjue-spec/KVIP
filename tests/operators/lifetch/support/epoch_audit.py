"""CPU-side decoding of the stable 128-byte V9 Rotation Event ABI.

The event ABI is defined in csrc/operators/lightning_indexer_with_fetch/include/context.h.
Telemetry fields are only meaningful in a binary compiled with
V9_INTERMEDIATE_PROFILING=1. Completion/skipped/core/epoch are always recorded.

This module deliberately does not interpret "postedCount" as completed DMA.
"""

from __future__ import annotations

import numpy as np


EVENT_BYTES = 128
EVENT_WORDS = 32


def _trigger_token(history, resident, rounds, epoch):
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    cold_tiles = (history - resident) // 128
    return resident + 128 * (cold_tiles * epoch // max(1, rounds - 1))


def _owner_core(batch, split_q, rank):
    return (rank % batch) * 2 + rank // batch if split_q else rank


def audit_rotation_events(
    events,
    *,
    batch,
    split_q,
    li_cores,
    rounds,
    service_aics,
    service_lanes,
    history,
    resident,
    profiling_enabled,
):
    """Decode and validate one post-v9_wait event snapshot.

    Always return a JSON-serializable report, including validation errors.
    A caller should persist the report before rejecting an invalid snapshot.
    """
    if (
        batch <= 0 or rounds <= 0 or service_aics <= 0
        or service_lanes != 2 * service_aics or li_cores <= 0
        or rounds * service_aics > li_cores
        or history <= resident or history % 128 or resident % 128
    ):
        raise ValueError("unsupported Rotation Event audit geometry")

    data = np.ascontiguousarray(events)
    if data.dtype != np.uint8 or data.shape != (rounds, service_lanes, EVENT_BYTES):
        raise ValueError(
            "Rotation Event ABI requires uint8[rounds,service_lanes,128], "
            f"got dtype={data.dtype} shape={data.shape}"
        )

    words = data.view(np.uint32).reshape(rounds, service_lanes, EVENT_WORDS)
    words64 = data.view(np.uint64).reshape(rounds, service_lanes, EVENT_BYTES // 8)

    report = {
        "schema": "kvip.rotation_epoch_audit.v1",
        "profiling_enabled": bool(profiling_enabled),
        "geometry": {
            "batch": int(batch),
            "split_q": bool(split_q),
            "li_cores": int(li_cores),
            "rounds": int(rounds),
            "service_aics": int(service_aics),
            "service_lanes": int(service_lanes),
            "history": int(history),
            "resident": int(resident),
        },
        "events": [],
        "errors": [],
    }
    completed = skipped_total = served_total = 0
    candidate_total = selected_total = 0

    for epoch in range(rounds):
        for lane in range(service_lanes):
            expected_rank = epoch * service_aics + lane // 2
            expected_core = _owner_core(batch, bool(split_q), expected_rank)
            core = int(words[epoch, lane, 4])
            recorded_epoch = int(words[epoch, lane, 5])
            complete = int(words[epoch, lane, 6])
            skipped = int(words[epoch, lane, 7])
            entry = {
                "epoch": epoch,
                "lane": lane,
                "aiv_sub": lane % 2,
                "expected_core": expected_core,
                "recorded_core": core,
                "recorded_epoch": recorded_epoch,
                "trigger_token": _trigger_token(history, resident, rounds, epoch),
                "complete": complete,
                "skipped": skipped,
                "skip_reason": (
                    "trigger_not_reached_or_no_li_work" if skipped == 1 else None
                ),
                "previous_epoch": epoch - 1 if epoch else None,
                "candidate_count": None,
                "selected_for_post_count": None,
                "bitmap_bytes": None,
                "scan_cycles": None,
                "candidate_cycles": None,
                "post_cycles": None,
                "wait_cycles": None,
                "service_cycles": None,
            }

            if complete != 1:
                report["errors"].append(
                    f"epoch={epoch} lane={lane}: complete={complete}, expected 1"
                )
            else:
                completed += 1
            if skipped not in (0, 1):
                report["errors"].append(
                    f"epoch={epoch} lane={lane}: skipped={skipped} not in (0,1)"
                )
            elif skipped:
                skipped_total += 1
            else:
                served_total += 1
            if core != expected_core or core >= li_cores:
                report["errors"].append(
                    f"epoch={epoch} lane={lane}: recorded core {core} "
                    f"!= expected core {expected_core}"
                )
            if recorded_epoch != epoch:
                report["errors"].append(
                    f"epoch={epoch} lane={lane}: recorded epoch {recorded_epoch}"
                )

            # These six counters and timestamps are inside V9_FETCH_PROFILE.
            # Skipped epochs bypass FastService, so leave its counters as null
            # even in profiling builds.
            if profiling_enabled and skipped == 0:
                candidate_count = int(words64[epoch, lane, 9])
                selected_count = int(words64[epoch, lane, 10])
                entry.update(
                    candidate_count=candidate_count,
                    selected_for_post_count=selected_count,
                    bitmap_bytes=int(words64[epoch, lane, 8]),
                    scan_cycles=int(words64[epoch, lane, 5]),
                    candidate_cycles=int(words64[epoch, lane, 6]),
                    post_cycles=int(words64[epoch, lane, 7]),
                )
                wait_begin = int(words64[epoch, lane, 4])
                begin = int(words64[epoch, lane, 0])
                end = int(words64[epoch, lane, 1])
                if wait_begin and begin >= wait_begin:
                    entry["wait_cycles"] = begin - wait_begin
                if begin and end >= begin:
                    entry["service_cycles"] = end - begin
                if selected_count > candidate_count:
                    report["errors"].append(
                        f"epoch={epoch} lane={lane}: selected {selected_count} "
                        f"> candidate {candidate_count}"
                    )
                candidate_total += candidate_count
                selected_total += selected_count

            report["events"].append(entry)

    report["summary"] = {
        "total_lanes": rounds * service_lanes,
        "complete_lanes": completed,
        "skipped_lanes": skipped_total,
        "served_lanes": served_total,
        "candidate_count_observed": candidate_total if profiling_enabled else None,
        "selected_for_post_count_observed": selected_total if profiling_enabled else None,
        "skipped_epoch_candidate_count": None,
        "confirmed_dma_completed": None,
        "validation": "PASS" if not report["errors"] else "FAIL",
    }
    return report


def _fmt_count(value):
    return "N/A" if value is None else str(value)


def format_rotation_table(report):
    lines = [
        "Epoch Lane Core Expected Trigger Complete Skipped Candidate SelectedForPost",
    ]
    for entry in report["events"]:
        lines.append(
            f"{entry['epoch']:5} {entry['lane']:4} "
            f"{entry['recorded_core']:4} {entry['expected_core']:8} "
            f"{entry['trigger_token']:7} {entry['complete']:8} "
            f"{entry['skipped']:7} {_fmt_count(entry['candidate_count']):>9} "
            f"{_fmt_count(entry['selected_for_post_count']):>15}"
        )
    summary = report["summary"]
    lines.append(
        f"RESULT {summary['validation']}: complete={summary['complete_lanes']}/"
        f"{summary['total_lanes']} skipped={summary['skipped_lanes']} "
        f"profiling={'on' if report['profiling_enabled'] else 'off'}"
    )
    return "\n".join(lines) + "\n"
