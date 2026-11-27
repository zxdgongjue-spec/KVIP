# SPDX-License-Identifier: Apache-2.0
"""Read one independent terminal diagnostic per cell of the B24 NPU3 fixed-Q4 archive."""
import argparse
import json
import math
from pathlib import Path

ARCHIVE = Path("/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold/peer-baseline/fixed-q4.ygIfRl")
SINGLE = "one_restored_graph_per_device_event_with_profiler"
NATIVE = "af52a02b97daddadd5d93b9d100f2ff1b03b499d"
GROUPS = (7, 4, 17, 2, 10)
BATCHES = (20, 24, 28, 32)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def marker(root, name):
    path = root / name
    return path.read_text().strip() if path.is_file() else None


def load_record(path, group, batch):
    data = json.loads(path.read_text())
    identity = data["identity"]
    expected = ("fused", group, batch, 3, SINGLE)
    actual = (data["arm"], identity["group"], identity["batch"], identity["physical_npu"], identity["protocol"])
    require(expected == actual, f"wrong arm, group, batch, physical card or recording protocol: {actual}")
    donors = list(range(16)) + [i * 16 // (batch - 16) for i in range(batch - 16)]
    require(identity["donor_rows"] == donors, "unexpected donor row assignment")
    root = path.parents[3]
    for name, key in (("host.txt", "host"), ("native.txt", "native"), ("physical_npu.txt", "physical_npu")):
        value = marker(root, name)
        require(value in (None, str(identity[key])), f"archive marker disagrees with arm: {name}")
    require(len(identity["sequence_sha256"]) == 64, "missing input fingerprint")
    return data


def find_record(root, group, batch):
    path = root / f"g{group}/bs{batch}/fused/ARM_RESULT.json"
    data = load_record(path, group, batch)
    target = data["identity"]
    require(target["native"] == NATIVE, f"target differs from the fixed-Q4 baseline: {target['native']}")
    required = ("separate_terminal_query_stamps", "separate_terminal_teams", "separate_LI_diagnostic")
    if all(key in data for key in required) or not data.get("comparison_archive"):
        return data, path, "direct"
    # comparison_archive identifies a previous run. Never silently present another kernel's
    # timings as a measurement of the requested candidate.
    alternate = Path(data["comparison_archive"]) / "fused/ARM_RESULT.json"
    previous = load_record(alternate, group, batch)
    for key in ("native", "host", "sequence_sha256", "kvaf_get_kernel_sha256", "hardware", "cann", "donor_rows"):
        require(previous["identity"][key] == target[key], f"fallback {alternate} differs in {key}; target has no raw stamps")
    machine = marker(root, "machine_id.txt")
    require(machine and machine == marker(alternate.parents[3], "machine_id.txt"), "fallback machine identity unavailable or different")
    return previous, alternate, "linked_same_identity"


def query_spans(stamp, q):
    t = stamp
    require(len(t) == 32 and all(isinstance(x, int) and x >= 0 for x in t), "invalid 32-word query record")
    own = t[20]
    require(own == t[21] and 0 <= own <= 2048, "invalid final token counter")
    chain = [0, 1, 2, 3, 6, 11, 12, 7] + ([13] if own else []) + [14, 15, 4, 5, 8, 26, 16, 24, 17, 9, 10]
    require(t[0] > 0 and all(t[i] <= t[j] for i, j in zip(chain, chain[1:])), "nonmonotonic or missing terminal clocks")
    require(bool(t[13]) == bool(own) and t[10] > t[0], "WQE start and final token counter disagree")
    # Only Q2 joins the row's prefetch CQ. Post-1 overhead on an empty Q belongs to control.
    phases = dict(init=t[1]-t[0], topk=t[2]-t[1], rank_publish=t[3]-t[2], dedup_slots_routes=t[11]-t[3],
                  output_publish=t[7]-t[11], WQE=t[14]-t[13] if own else 0, first_Post=t[4]-t[15] if own else 0,
                  snapshot=t[26]-t[8], Index_join=t[16]-t[26], final_join=t[24]-t[16],
                  Q2_prefetch_join=t[17]-t[24] if q == 1 else 0, retire=t[10]-t[17])
    phases["control"] = t[10]-t[0]-sum(phases.values())
    require(phases["control"] >= 0, "stage accounting exceeds the query wall span")
    return phases


def window_totals(events, batch):
    if events is None:
        return None
    require(len(events) == 4 and all(len(epoch) == batch for epoch in events), "wrong window layout")
    totals = []
    for epoch, rows in enumerate(events):
        for row, w in enumerate(rows):
            require(len(w) == 16 and w[2] == (epoch << 32) + row and (w[3] & 0xffffffff) == 1,
                    f"invalid window completion/identity: row={row} epoch={epoch}")
            require(0 < w[0] <= w[1] and 0 <= w[10] <= 256, f"invalid window clocks/count: row={row} epoch={epoch}")
        totals.append(sum(w[10] for w in rows))
    return totals


def report(root, group, batch):
    data, path, selected = find_record(root, group, batch)
    ident = data["identity"]
    t = data["separate_terminal_query_stamps"]
    hz = float(data["separate_LI_diagnostic"]["clock_mhz"])
    require(math.isfinite(hz) and hz > 0, "invalid device clock calibration")
    require(len(t) == batch and all(len(row) == 4 for row in t), "wrong row/Q shape")
    spans = [[query_spans(stamp, q) for q, stamp in enumerate(row)] for row in t]
    origin = min(s[0] for row in t for s in row)
    end = max(s[10] for row in t for s in row)
    lasts = [(r, q) for r, row in enumerate(t) for q, s in enumerate(row) if s[10] == end]
    r, q = lasts[0]
    teams = data["separate_terminal_teams"]
    require((r, q+1) == (teams["last_row"], teams["last_q"]) and
            math.isclose((end-origin)/hz, teams["observed_span_us"], rel_tol=1e-8),
            "raw last row/Q or calibrated span differs from stored team report")
    pref = window_totals(data.get("separate_window_event_u64"), batch)
    if pref is not None:
        for row, queries in enumerate(t):
            start = sum(epoch[row][10] for epoch in data["separate_window_event_u64"])
            for slot, count in sorted((s[22], s[20]) for s in queries if s[20]):
                require(slot == start, f"reserved final slots disagree with observed row/Q counts: row={row}")
                start += count
    entries = [(i, j, s) for i, row in enumerate(t) for j, s in enumerate(row)]
    fin = sum(s[20] for _, _, s in entries)
    tag = f"G{group} BS{batch}"
    print(f"{tag} SOURCE={path} origin={selected} native={ident['native'][:8]} input={ident['sequence_sha256'][:12]} "
          f"clock_MHz={hz:.4f} span_us={(end-origin)/hz:.2f} last_row={r} donor={ident['donor_rows'][r]} "
          f"last_Q={q+1} tied_finishes={len(lasts)} final_tokens={fin} "
          f"prefetch_by_window={','.join(map(str, pref)) if pref is not None else 'UNAVAILABLE'}")
    last = t[r][q]
    fields = " ".join(f"{name}={ticks/hz:.2f}" for name, ticks in spans[r][q].items())
    print(f"{tag} LAST_Q_PATH_us entry_from_first={(last[0]-origin)/hz:.2f} {fields}")
    if group == 7:
        for sibling, stamp in enumerate(t[r]):
            stage = spans[r][sibling]
            print(f"{tag} LAST_ROW_Q={sibling+1} final_tokens={stamp[20]} entry_us={(stamp[0]-origin)/hz:.2f} "
                  f"WQE_us={stage['WQE']/hz:.2f} first_Post_us={stage['first_Post']/hz:.2f} "
                  f"Index_join_us={stage['Index_join']/hz:.2f} final_join_us={stage['final_join']/hz:.2f} "
                  f"Q2_prefetch_join_us={stage['Q2_prefetch_join']/hz:.2f} retire_us={stage['retire']/hz:.2f} "
                  f"exit_us={(stamp[10]-origin)/hz:.2f}")
    # The last retiree can have no GET of its own. Expose the latest observations
    # by active owners separately; never add overlapping Q durations.
    for label, until, selection in (
        ("final", 24, [(i,j,s) for i,j,s in entries if s[20]]),
        ("prefetch", 17, [(i,j,s) for i,j,s in entries if j == 1 and pref is not None
                           and any(e[i][10] for e in data["separate_window_event_u64"])]),
    ):
        if not selection:
            print(f"{tag} LATEST_{label}_JOIN {'UNAVAILABLE missing window counters' if label == 'prefetch' and pref is None else 'no_nonempty_GET'}")
            continue
        i, j, s = max(selection, key=lambda x:x[2][until])
        p = spans[i][j]
        print(f"{tag} LATEST_{label}_JOIN row={i} donor={ident['donor_rows'][i]} Q={j+1} "
              f"final_tokens={s[20]} WQE_us={p['WQE']/hz:.2f} Index_join_us={p['Index_join']/hz:.2f} "
              f"final_join_us={p['final_join']/hz:.2f} Q2_prefetch_join_us={p['Q2_prefetch_join']/hz:.2f} "
              f"join_to_last_retire_us={(end-s[until])/hz:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=ARCHIVE)
    parser.add_argument("--groups", nargs="+", type=int, choices=GROUPS, default=GROUPS)
    parser.add_argument("--batches", nargs="+", type=int, choices=BATCHES, default=BATCHES)
    args = parser.parse_args()
    root = args.archive.resolve()
    if not root.is_dir() or marker(root, "physical_npu.txt") != "3":
        parser.error(f"expected a physical NPU3 archive at {root}")
    print("SCOPE=one_separate_instrumented_replay unit=us row=zero_based Q=one_based")
    print("dedup_slots_routes includes output routing; Index_join includes metadata work and synchronization; "
          "final_join includes later paced Posts, status writes and CQ waits. "
          "Q2 joins prefetch. Snapshot is diagnostic overhead. Raw CQE arrival, per-prefix times, "
          "physical AIV IDs and ten-call-graph internal times are unavailable.")
    count = 0
    for group in args.groups:
        for batch in args.batches:
            try:
                report(root, group, batch)
                count += 1
            except (ValueError, KeyError, TypeError, OSError) as exc:
                print(f"G{group} BS{batch} UNAVAILABLE archive={root} reason={exc}")
    print(f"DIAGNOSTIC_READ cells={count}/{len(args.batches)*len(args.groups)}")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
