# SPDX-License-Identifier: Apache-2.0
"""Decode a separate pruning-counter replay; these words are not timestamps."""

from collections import Counter

import numpy as np

MAGIC = 0x54504B31
STATUSES = {
    1: "compile_disabled", 2: "guard_fallback", 3: "zero_threshold",
    4: "invalid_compaction", 5: "insufficient_candidates", 6: "success",
    7: "empty_query",
}
REASONS = (
    "score_type", "prefix", "topk_only", "load_distribution", "query_tile",
    "unsplit_query", "task_count", "topk_size", "trunk_size", "short_resident",
    "short_topk", "scratch_size",
)


def decode(raw, batch):
    raw = np.asarray(raw)
    if raw.dtype != np.int64 or raw.ndim != 1 or batch <= 0 or raw.size < batch * 128:
        raise ValueError("Invalid TopK pruning probe tensor")
    rows = raw[batch * 64:batch * 128].view(np.uint64).reshape(batch * 4, 16)[:, 12:16]
    queries = []
    for index, words in enumerate(rows):
        w0, w1, w2, w3 = map(int, words)
        status = (w0 >> 48) & 255
        reasons = (w0 >> 32) & 65535
        if w0 & 0xFFFFFFFF != MAGIC or status not in STATUSES or w0 >> 57 or reasons >> len(REASONS):
            raise ValueError(f"Missing/unknown/incomplete TopK probe: query={index}")
        row = dict(
            query=index, status=STATUSES[status], activated=bool((w0 >> 56) & 1),
            reasons=[name for bit, name in enumerate(REASONS) if reasons & (1 << bit)],
            visited=w1 & 0xFFFFFFFF, survivors=w1 >> 32,
            radix_inputs=w2 & 0xFFFFFFFF, radix_calls=w2 >> 32,
            valid_end=w3 & 0xFFFFFFFF, threshold_key=(w3 >> 32) & 65535, blocks=w3 >> 48,
        )
        if row["survivors"] > row["visited"] or row["visited"] > row["valid_end"]:
            raise ValueError(f"Inconsistent TopK probe counts: query={index}")
        if status == 6 and (not row["activated"] or row["reasons"] or row["visited"] != row["valid_end"]):
            raise ValueError(f"Inconsistent successful TopK probe: query={index}")
        queries.append(row)
    totals = {key: sum(row[key] for row in queries) for key in ("visited", "survivors", "radix_inputs", "radix_calls")}
    return dict(
        included_in_timing=False, protocol="TPK1", query_count=len(queries),
        activated_queries=sum(row["activated"] for row in queries),
        statuses=dict(Counter(row["status"] for row in queries)),
        guard_reasons=dict(Counter(reason for row in queries for reason in row["reasons"])),
        totals=totals,
        survivor_fraction=totals["survivors"] / totals["visited"] if totals["visited"] else None,
        queries=queries,
        note="Independent diagnostic binary and replay, checked by the original oracle; no latency result. Visits exclude any subsequent fallback; radix inputs include padding and carried TopK.",
    )


def print_summary(probe):
    fraction = probe["survivor_fraction"]
    ratio = "n/a" if fraction is None else f"{100 * fraction:.2f}%"
    statuses = ",".join(f"{name}:{count}" for name, count in sorted(probe["statuses"].items()))
    reasons = ",".join(f"{name}:{count}" for name, count in sorted(probe["guard_reasons"].items())) or "none"
    print(f"TOPK_PROBE activated={probe['activated_queries']}/{probe['query_count']} survivor={ratio} "
          f"radix_inputs={probe['totals']['radix_inputs']} radix_calls={probe['totals']['radix_calls']} "
          f"status={statuses} fallback={reasons} timing=excluded", flush=True)
