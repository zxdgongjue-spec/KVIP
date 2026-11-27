# SPDX-License-Identifier: Apache-2.0
"""Attribute private-Q terminal batches using completed ten-call graph journals."""
import argparse
from pathlib import Path
from statistics import mean

from b24_window_report import serial_reference, stats, values
from export_large_rank_evidence import IDENTITY_KEYS
from g7_x4_local import ADVANCE, MODE, MODERATE, check_earned_limits, choose, placement, valid_local

BENCH = "fe37fdc78a3292eef34b71334437dd768380f583"


def chunks(n, split):
    if not n:
        return ()
    if not split or n == 1:
        return (n,)
    # PlanTerminalBatches uses exclusive 1/16, 6/16 and 11/16 boundaries.
    ends = sorted({(n * i + 15) // 16 for i in (1, 6, 11)} | {n})
    return tuple(r - l for l, r in zip((0, *ends[:-1]), ends, strict=True))


def inspect_call(call, batch):
    final = call["row_query_final"]
    prefetch = call["row_window_prefetch"]
    if (len(final) != batch or len(prefetch) != batch
            or any(len(row) != 4 for row in final) or any(len(row) != 5 for row in prefetch)
            or any(type(n) is not int or n < 0 for row in [*final, *prefetch] for n in row)
            or sum(map(sum, prefetch)) != call["prefetch"]
            or sum(map(sum, final)) != call["final"]):
        raise ValueError("Window or per-Q issue counts disagree with measured totals")
    bursts = [0] * 4
    long_rows = split_queries = completions = max_first = 0
    first_rows = []
    for row in final:
        split = sum(row) >= 256
        long_rows += split
        first_row = 0
        for n in row:
            parts = chunks(n, split)
            cq = min(n, 4) if split else int(n != 0)
            if len(parts) != cq or sum(parts) != n:
                raise ValueError("Native terminal plan disagrees with receipt-validation rule")
            split_queries += cq > 1
            completions += cq
            if parts:
                first_row += parts[0]
                max_first = max(max_first, parts[0])
            for i, count in enumerate(parts):
                bursts[i] += count
        first_rows.append(first_row)
    if "final_cqes" in call and call["final_cqes"] != completions:
        raise ValueError("Archived measured CQ total differs from recalculated private batches")
    if sum(bursts) != call["final"]:
        raise ValueError("Incomplete reconstruction of final descriptors")
    return dict(long_rows=long_rows, split_q=split_queries, cq=completions,
                cq_total_saved="final_cqes" in call, parts=bursts, max_q_first=max_first,
                first16_min=sum(sorted(first_rows)[:16]), first16_max=sum(sorted(first_rows)[-16:]))


def find(area, batch, mode, reader):
    state, ref, arm = choose(area, batch, reader, mode)
    serial = None
    if state == "paired-cache":
        r, _, identity = values(arm, ref, 7, batch, mode)
        if mode == MODERATE:
            serial = serial_reference(ref, identity, 7, batch)
    elif state == "local-cache":
        r = valid_local(arm, batch, mode)
        if mode == MODERATE:
            local_id = {k: r["identity"][k] for k in IDENTITY_KEYS}
            serial = serial_reference(arm.parents[3], (local_id, placement(arm / "thp.json")), 7, batch)
    else:
        return None
    if mode in (ADVANCE, MODERATE):
        check_earned_limits(r, batch, mode)
    return arm, r, placement(arm / "thp.json"), serial


def output(area, batch, moderate_reader=None):
    found = []
    modes = (MODE, ADVANCE, MODERATE) if moderate_reader else (MODE, ADVANCE)
    for mode in modes:
        hit = find(area, batch, mode, moderate_reader if mode == MODERATE else BENCH)
        found.append(hit)
        tag = f"G7 BS{batch} {mode}"
        if hit is None:
            print(f"{tag} UNAVAILABLE missing_completed_same_host_NPU4_archive")
            continue
        arm, r, _, _ = hit
        if len(r["traffic"]) != 10:
            raise ValueError("Expected results from all ten measured calls")
        calls = [inspect_call(t, batch) for t in r["traffic"]]
        cq_source = ("device_total_checked" if all(c["cq_total_saved"] for c in calls)
                     else "from_actual_per_Q_counts_device_receipts_validated_during_replay")
        print(f"{tag} archive={arm}")
        print(f"{tag} graph/LI/tail_mean_us={r['graph_mean_us']:.2f}/{r['mean_us']['LI']:.2f}/{r['mean_us']['tail']:.2f} "
              f"final_GET_mean={mean(t['final'] for t in r['traffic']):.1f} "
              f"rows_at_least_256_mean={mean(c['long_rows'] for c in calls):.1f} "
              f"split_private_Q_mean={mean(c['split_q'] for c in calls):.1f} "
              f"private_CQ_mean={mean(c['cq'] for c in calls):.1f} CQ_source={cq_source}")
        parts = "/".join(f"{mean(c['parts'][i] for c in calls):.1f}" for i in range(4))
        print(f"{tag} all_rows_tokens_per_private_doorbell_1_to_4_mean={parts} "
              f"largest_one_Q_first_mean={mean(c['max_q_first'] for c in calls):.1f} "
              f"any_16_rows_first_doorbell_range_mean={mean(c['first16_min'] for c in calls):.1f}.."
              f"{mean(c['first16_max'] for c in calls):.1f} timestamp_peak_4us=UNAVAILABLE")
    target = found[-1]
    for mode, ref_hit in zip(modes[:-1], found[:-1], strict=True):
        label = "ordinary_five" if mode == MODE else "earned_five"
        if target is None or ref_hit is None:
            print(f"G7 BS{batch} vs_{label}=UNAVAILABLE no_complete_pair")
            continue
        _, old, old_loc, _ = ref_hit
        _, new, new_loc, _ = target
        if (old_loc != new_loc or old["identity"] != new["identity"]
                or old["tail_text"] != new["tail_text"] or old["original_library"] != new["original_library"]):
            print(f"G7 BS{batch} vs_{label}=UNAVAILABLE mismatched_input_placement_or_tail")
            continue
        saved = "/".join(f"{a-b:+.2f}" for a, b in zip(stats(old), stats(new), strict=True))
        gain = 100 * (1 - new["graph_mean_us"] / old["graph_mean_us"])
        print(f"G7 BS{batch} vs_{label} graph/LI/LI1/LI23/tail_saved_mean_us={saved} graph_gain_pct={gain:+.2f}")
    if moderate_reader and target:
        _, new, _, serial = target
        if serial:
            value = new["graph_mean_us"]
            baseline = serial["graph"]
            print(f"G7 BS{batch} vs_strong_serial graph_saved_mean_us={baseline-value:+.2f} "
                  f"graph_gain_pct={100*(1-value/baseline):+.2f} gap_to_15/20_us={value-.85*baseline:+.2f}/{value-.8*baseline:+.2f}")
        else:
            print(f"G7 BS{batch} vs_strong_serial=UNAVAILABLE no_matching_same_card_archive")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("peer_baseline", type=Path)
    p.add_argument("--moderate-reader")
    p.add_argument("--batch", type=int, choices=(32, 28))
    args = p.parse_args()
    for size in (args.batch,) if args.batch else (32, 28):
        output(args.peer_baseline, size, args.moderate_reader)
