# SPDX-License-Identifier: Apache-2.0
"""Read the four query timelines of the last G7 row to retire on physical NPU7."""
import json
import math
import os
import re
import sys
from pathlib import Path


SINGLE = "one_restored_graph_per_device_event_with_profiler"
NATIVE = "af52a02b97daddadd5d93b9d100f2ff1b03b499d"

BATCHES = (20, 24, 28, 32)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def marker(root, name):
    path = root / name
    return path.read_text().strip() if path.is_file() else None


def local_marker(filename):
    path = Path(filename)
    return path.read_text().strip() if path.is_file() else None


def load_record(path, group, batch):
    data = json.loads(path.read_text())
    identity = data["identity"]
    expected = ("fused", group, batch, 7, SINGLE)
    actual = (data["arm"], identity["group"], identity["batch"], identity["physical_npu"], identity["protocol"])
    require(expected == actual, f"wrong arm, group, batch, physical card or recording protocol: {actual}")
    donors = list(range(16)) + [i * 16 // (batch - 16) for i in range(batch - 16)]
    require(identity["donor_rows"] == donors, "unexpected donor row assignment")
    root = path.parents[3]
    for name, key in (("host.txt", "host"), ("native.txt", "native"), ("physical_npu.txt", "physical_npu")):
        value = marker(root, name)
        require(value in (None, str(identity[key])), f"archive marker disagrees with arm: {name}")
    require(re.fullmatch(r"[0-9a-f]{64}", identity["sequence_sha256"]), "invalid input fingerprint")
    require(identity["host"] == os.uname().nodename, "different host")
    require(identity["calls"] == 10 and identity["warmup"] == 4 and data["get_workers"] == 32
            and data["checked_queries"] == 40 * batch and data["checked_rows"] == 10 * batch
            and data["full_KV_Index_checked_rows"] == batch, "incomplete measurement")
    require(marker(root, "physical_npu.txt") == "7", "missing or wrong physical card")
    proof = path.parent / "path/path_result.json"
    if proof.is_file():
        result = json.loads(proof.read_text())
        require(result["path"] == "fused" and result["transport"] == "urma" and result["correctness"] == "PASS"
                and not result["failures"] and result["runtime_workers"] == 32, "failed or different replay")
    machine = local_marker("/etc/machine-id")
    archived = marker(root, "machine_id.txt")
    if machine and archived:
        require(machine == archived, "different machine ID")
    else:
        boot = local_marker("/proc/sys/kernel/random/boot_id")
        require(boot and boot == marker(root, "boot_id.txt"), "different boot or missing machine identity")
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


def check_query(t):
    require(len(t) == 32 and all(isinstance(x, int) and x >= 0 for x in t), "invalid 32-word query record")
    own = t[20]
    require(own == t[21] and 0 <= own <= 2048, "invalid final token counter")
    chain = [0, 1, 2, 3, 6, 11, 12, 7] + ([13] if own else []) + [14, 15, 4, 5, 8, 26, 16, 24, 17, 9, 10]
    require(t[0] > 0 and all(t[i] <= t[j] for i, j in zip(chain, chain[1:])), "nonmonotonic or missing terminal clocks")
    require(bool(t[13]) == bool(own) and t[10] > t[0], "WQE start and final token counter disagree")


def check_windows(events, batch):
    require(len(events) == 4 and all(len(epoch) == batch for epoch in events), "wrong window layout")
    for epoch, rows in enumerate(events):
        for row, w in enumerate(rows):
            require(len(w) == 16 and w[2] == (epoch << 32) + row and (w[3] & 0xffffffff) == 1,
                    f"invalid window completion/identity: row={row} epoch={epoch}")
            require(0 < w[0] <= w[1] and 0 <= w[10] <= 256, f"invalid window clocks/count: row={row} epoch={epoch}")


def intervals(stamp, start, mhz, has_prefetch):
    def at(i):
        return f"{(stamp[i] - start) / mhz:.2f}"

    def interval(begin, end, present=True):
        return f"{at(begin)}..{at(end)}" if present else "-"

    own = stamp[20] > 0
    return (f"final_GET={stamp[20]} start={at(0)} WQE={interval(13, 14, own)} "
            f"first_Post={interval(15, 4, own)} Index={interval(26, 16)} "
            f"final_CQ_join={interval(16, 24, own)} Q2_prefetch_CQ_join={interval(24, 17, has_prefetch)} "
            f"exit={at(10)}")


def timeline(root, batch, fingerprint):
    data, path, selection = find_record(root, 7, batch)
    ident = data["identity"]
    if fingerprint:
        require(ident["sequence_sha256"] == fingerprint, "G7 input differs from the fixed workload archive")
    require(data["ordered_Index_diagnostic_rows"] == batch, "incomplete instrumented Index check")
    ticks = data["separate_terminal_query_stamps"]
    mhz = float(data["separate_LI_diagnostic"]["clock_mhz"])
    require(math.isfinite(mhz) and mhz > 0, "invalid device clock calibration")
    require(len(ticks) == batch and all(len(row) == 4 for row in ticks), "wrong row/Q shape")
    for row in ticks:
        for stamp in row:
            check_query(stamp)
    events = data["separate_window_event_u64"]
    check_windows(events, batch)
    for r, row in enumerate(ticks):
        slot = sum(w[r][10] for w in events)
        for offset, count in sorted((s[22], s[20]) for s in row if s[20]):
            require(offset == slot, "final reservations disagree with window counters")
            slot += count

    start = min(s[0] for row in ticks for s in row)
    last_row, last_q = max(((r, q) for r in range(batch) for q in range(4)),
                           key=lambda rq: ticks[rq[0]][rq[1]][10])
    span = (ticks[last_row][last_q][10] - start) / mhz
    team = data["separate_terminal_teams"]
    require((last_row, last_q + 1) == (team["last_row"], team["last_q"])
            and math.isclose(span, team["observed_span_us"], rel_tol=1e-8), "team clocks disagree")
    tag = f"G7 BS{batch}"
    lines = [f"{tag} PHYSICAL_NPU=7 last_row={last_row} last_Q={last_q+1} span_us={span:.2f} "
             f"input={ident['sequence_sha256']} SOURCE={path} source_type={selection}"]
    for q, stamp in enumerate(ticks[last_row], 1):
        has_prefetch = q == 2 and any(w[last_row][10] for w in events)
        lines.append(f"{tag} Q{q} " + intervals(stamp, start, mhz, has_prefetch))
    return lines


def main(area):
    roots = sorted((p.parent for p in area.glob("*/physical_npu.txt")),
                   key=lambda p: p.stat().st_mtime_ns, reverse=True)
    roots = [p for p in roots if marker(p, "physical_npu.txt") == "7" and marker(p, "native.txt") == NATIVE]
    print("TAIL_CLOCKS unit=us origin=first_terminal_entry last_row=last_to_retire row_base=0 Q_base=1 "
          "interval=start..end source=one_instrumented_replay")
    complete = 0
    for batch in BATCHES:
        # The NPU3 reference supplies only an input digest. Never use its clocks.
        ref = area / "fixed-q4.ygIfRl" / f"g7/bs{batch}/fused/ARM_RESULT.json"
        fingerprint = None
        if ref.is_file():
            refid = json.loads(ref.read_text())["identity"]
            require((refid["group"], refid["batch"], refid["native"]) == (7, batch, NATIVE),
                    f"unexpected fixed workload reference at {ref}")
            fingerprint = refid["sequence_sha256"]
        for root in roots:
            if not (root / f"g7/bs{batch}/fused/ARM_RESULT.json").is_file():
                continue
            try:
                lines = timeline(root, batch, fingerprint)
            except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
                print(f"ARCHIVE_SKIPPED BS{batch} {root}: {exc}", file=sys.stderr)
                continue
            print(*lines, sep="\n")
            complete += 1
            break
        else:
            print(f"MISSING {batch}")
    print(f"DIAGNOSTIC_READ cells={complete}/4")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
