# SPDX-License-Identifier: Apache-2.0
"""Survey real physical Index permutations and cold-demand discovery on CPU.

Eight cold windows, plus the resident publication. Discovery is a bound: no
budget, threshold, transport completion or counterfactual cache evolution is
simulated. Read only ID/state members of the score archives, not Q/K or scores.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import gzip
import json
from pathlib import Path
import re
import time

import numpy as np

from kvip.diagnostics.archive import checked_path, source_commit
from kvip.diagnostics.threshold_survey import survey_tasks
from kvip.diagnostics.trajectory import inspect_series

BLOCK = 2048
WINDOWS = 8
FIELDS = ("pre.logical_ids", "pre.hot_valid", "pre.query_ends", "pre.tail_bounds",
          "pre.lru_slots", "pre.address_valid", "pre.metadata", "post.topk",
          "post.logical_output", "post.logical_ids", "post.hot_valid")


def read_ids(path):
    """NPZ member CRC and declared dtype/shape checked; do not read score members."""
    path = Path(path)
    m = json.loads(path.read_text())
    if (m.get("status") != "complete" or m.get("content") != "scores"
            or m.get("validation", {}).get("passed") is not True):
        raise ValueError(f"Incomplete score archive: {path}")
    with np.load(checked_path(path.parent, m["archive"]["file"]), allow_pickle=False) as z:
        a = {name: z[name] for name in FIELDS}
    for name, value in a.items():
        spec = m["arrays"][name]
        if value.dtype.hasobject or value.dtype.name != spec["dtype"] or list(value.shape) != spec["shape"]:
            raise ValueError(f"ID member differs from manifest: {path}: {name}")
    return m, a


def correlation(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def scan_windows(history, resident, ends, reverse):
    """Physical -> native scan coordinates; identical involution to cold_scan.h."""
    if history % 256 or resident % 256 or (history - resident) // 128 <= WINDOWS:
        raise ValueError("Expected aligned history/resident with at least nine cold tiles")
    physical = np.arange(history, dtype=np.int64)
    scan = physical.copy()
    begin = (resident + BLOCK - 1) // BLOCK * BLOCK
    end = int(min(ends)) // BLOCK * BLOCK
    if reverse and end > begin:
        mask = (physical >= begin) & (physical < end)
        p = physical[mask]
        scan[mask] = begin + end - BLOCK - p // BLOCK * BLOCK + p % BLOCK
    boundaries = resident + ((history - resident) // 128 * np.arange(WINDOWS + 1) // WINDOWS) * 128
    # Slot boundary belongs to the following publication. Resident is phase 0.
    windows = np.searchsorted(boundaries, scan, side="right")
    if np.any(windows > WINDOWS):
        raise AssertionError("A physical position escaped the scan schedule")
    return windows


def request_metrics(m, a, row, previous):
    g = m["geometry"]
    h, r = g["history"], g["resident"]
    ids = a["pre.logical_ids"][row].astype(np.int64)
    post_ids = a["post.logical_ids"][row]
    if (len(ids) != h or np.any(ids < 0) or np.any(ids >= h)
            or not np.all(np.bincount(ids, minlength=h) == 1)
            or np.any(post_ids < 0) or np.any(post_ids >= h)
            or not np.all(np.bincount(post_ids, minlength=h) == 1)):
        raise ValueError("Index IDs must be complete permutations")
    positions = np.arange(h)
    ends = a["pre.query_ends"][row]
    stop = int(ends.max())
    if not r <= int(ends.min()) <= stop <= h:
        raise ValueError("Expected full resident history and causal Q4")
    topk = a["post.topk"][row]
    if np.any(topk < 0) or np.any(topk >= ends[:, None]):
        raise ValueError("TopK physical coordinates violate causal ends")
    if np.any(ids[topk] >= ends[:, None]):
        raise ValueError("TopK logical IDs violate causal ends")
    selected = np.zeros(h, bool)
    selected[topk.reshape(-1)] = True
    unpack = a["pre.metadata"][2] // 4
    logical = np.unique(a["post.logical_output"][unpack == row])
    if not np.array_equal(np.sort(ids[selected]), logical):
        raise ValueError("Physical TopK does not map to recorded logical output")
    hot = np.zeros(h, bool)
    hot[:r] = a["pre.hot_valid"][row] != 0
    lo, hi = a["pre.tail_bounds"][row]
    abc = (ids >= lo) & (ids < hi)
    cold = ~hot & ~abc
    need = selected & cold
    addresses = a["pre.address_valid"][row] != 0
    if np.any(need & ~addresses):
        raise ValueError("Final cold demand lacks a published DDR address")
    eligible = positions >= r
    eligible[a["pre.lru_slots"][row]] = True
    eligible &= addresses & ~abc
    eligible_need = need & eligible
    cold_suffix = (positions >= r) & (positions < stop) & cold & addresses
    domain = positions[cold_suffix]
    moved = ids[domain] != domain
    block_moved = ids[domain] // BLOCK != domain // BLOCK
    adjacent = cold_suffix[:-1] & cold_suffix[1:]
    breaks = (np.diff(ids) != 1) & adjacent
    before = previous["ids"][row]
    previous_stop = min(stop, int(previous["ends"][row].max()))
    compare_positions = np.arange(r, previous_stop)
    fwd = scan_windows(h, r, ends, False)
    rev = scan_windows(h, r, ends, True)
    f_counts = np.bincount(fwd[need], minlength=9)
    r_counts = np.bincount(rev[need], minlength=9)
    metrics = dict(
        selected=int(selected.sum()), cold=int(need.sum()), cold_suffix_count=len(domain),
        displaced=int(moved.sum()), displaced_block=int(block_moved.sum()),
        adjacency_pairs=int(adjacent.sum()), adjacency_breaks=int(breaks.sum()),
        position_token_correlation=correlation(domain, ids[domain]),
        previous_span=len(compare_positions),
        changed_since_previous_pre=int(np.count_nonzero(ids[compare_positions] != before[compare_positions])),
        previous_post_to_current_pre_changed=int(np.count_nonzero(
            ids[:previous_stop] != previous["post_ids"][row, :previous_stop])),
        current_pre_to_post_changed=int(np.count_nonzero(ids[:stop] != post_ids[:stop])),
        final_only_cold=int((need & ~eligible).sum()),
        cold_in_resident=int((need & (positions < r)).sum()),
        forward_early=int(f_counts[:8].sum()), reverse_early=int(r_counts[:8].sum()),
        forward_lead=float(np.dot(f_counts, (8 - np.arange(9)) / 8)),
        reverse_lead=float(np.dot(r_counts, (8 - np.arange(9)) / 8)),
        eligible_cold=int(eligible_need.sum()),
        forward_eligible_early=int((eligible_need & (fwd < 8)).sum()),
        reverse_eligible_early=int((eligible_need & (rev < 8)).sum()),
    )
    metrics.update({f"{order}_w{i}": int(counts[i]) for order, counts in
                    (("forward", f_counts), ("reverse", r_counts)) for i in range(9)})
    blocks = []
    for start in range(r, stop, BLOCK):
        end = min(start + BLOCK, stop)
        p = positions[start:end]
        local = cold_suffix[start:end]
        blocks.append(dict(
            physical_start=start, physical_end=end, physical_block=start // BLOCK,
            cold_demand=int(need[start:end].sum()), eligible_cold=int(eligible_need[start:end].sum()),
            cold_index_count=int(local.sum()), displaced=int(((ids[start:end] != p) & local).sum()),
            displaced_block=int(((ids[start:end] // BLOCK != p // BLOCK) & local).sum()),
            logical_id_min=int(ids[start:end].min()), logical_id_max=int(ids[start:end].max()),
            logical_id_mean=float(ids[start:end].mean()),
        ))
    return metrics, blocks


def task_run(task):
    rows, blocks, previous = [], [], None
    for capture_round, scheduler_round, path in task.snapshots:
        m, a = read_ids(path)
        meta = m["metadata"]
        signature = tuple((x["request_id"], x["owner"]) for x in meta["requests"])
        if (meta["dp_rank"] != task.dp_rank or meta["group"] != task.group
                or meta["context"]["round"] != scheduler_round
                or meta["context"].get("global_decode_confirmed") is not True
                or tuple(meta["layer_names"]) != task.layer_names or signature != task.signature):
            raise ValueError(f"Trace identity changed: {path}")
        layer_matches = [re.search(r"layers\.(\d+)(?:\.|$)", name) for name in task.layer_names]
        if not all(layer_matches):
            raise ValueError(f"Cannot resolve layer numbers: {task.layer_names}")
        layers = [int(match.group(1)) for match in layer_matches]
        if previous is not None:
            for row in range(m["geometry"]["batch"]):
                identity = dict(dp_rank=task.dp_rank, capture_round=capture_round,
                                scheduler_round=scheduler_round, group=task.group, packed_row=row,
                                layers="|".join(map(str, layers)))
                values, local_blocks = request_metrics(m, a, row, previous)
                rows.append(dict(identity, **values))
                blocks.extend(dict(identity, **b) for b in local_blocks)
        previous = dict(ids=a["pre.logical_ids"], post_ids=a["post.logical_ids"], ends=a["pre.query_ends"])
    return rows, blocks


def pct(n, d):
    return 100 * n / d if d else 0.0


def aggregate(rows):
    counts = {key: sum(row[key] for row in rows) for key in (
        "selected", "cold", "cold_suffix_count", "displaced", "displaced_block", "adjacency_pairs",
        "adjacency_breaks", "forward_early", "reverse_early", "forward_lead", "reverse_lead",
        "previous_span", "changed_since_previous_pre", "final_only_cold", "eligible_cold",
        "forward_eligible_early", "reverse_eligible_early", "previous_post_to_current_pre_changed")}
    return dict(
        samples=len(rows), **counts,
        miss_pct=pct(counts["cold"], counts["selected"]),
        displaced_pct=pct(counts["displaced"], counts["cold_suffix_count"]),
        displaced_block_pct=pct(counts["displaced_block"], counts["cold_suffix_count"]),
        fragmentation_pct=pct(counts["adjacency_breaks"], counts["adjacency_pairs"]),
        forward_early_pct=pct(counts["forward_early"], counts["cold"]),
        reverse_early_pct=pct(counts["reverse_early"], counts["cold"]),
        early_gain_pp=pct(counts["reverse_early"] - counts["forward_early"], counts["cold"]),
        forward_lead_pct=pct(counts["forward_lead"], counts["cold"]),
        reverse_lead_pct=pct(counts["reverse_lead"], counts["cold"]),
        lead_gain_pp=pct(counts["reverse_lead"] - counts["forward_lead"], counts["cold"]),
    )


def write_rows(writer, rows):
    if not rows:
        return writer
    if writer[1] is None:
        writer[1] = csv.DictWriter(writer[0], fieldnames=list(rows[0]))
        writer[1].writeheader()
    writer[1].writerows(rows)
    return writer


def finish(rows, output, expected_rounds):
    by_group, by_round = {}, {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
        by_round.setdefault((row["capture_round"], row["group"]), []).append(row)
    round_rows = []
    group_records = []
    for group, samples in sorted(by_group.items()):
        rounds = []
        for (capture, g), values in sorted(by_round.items()):
            if g != group:
                continue
            summary = dict(capture_round=capture, group=group, **aggregate(values))
            rounds.append(summary)
            for layer in samples[0]["layers"].split("|"):
                round_rows.append(dict(layer=int(layer), **summary))
        width = min(20, max(1, (expected_rounds - 1) // 5))
        early = aggregate([r for r in samples if r["capture_round"] <= width + 1])
        late = aggregate([r for r in samples if r["capture_round"] > expected_rounds - width])
        delta = late["early_gain_pp"] - early["early_gain_pp"]
        row = dict(group=group, layers=samples[0]["layers"], **aggregate(samples),
                   first_rounds=early, last_rounds=late, segment_rounds=width,
                   early_gain_change_pp=delta,
                   round_disorder_correlation=correlation([v["capture_round"] for v in rounds], [v["displaced_block_pct"] for v in rounds]),
                   disorder_gain_correlation=correlation([v["displaced_block_pct"] for v in rounds], [v["early_gain_pp"] for v in rounds]),
                   gain_p10_pp=float(np.percentile([v["early_gain_pp"] for v in rounds], 10)),
                   gain_p50_pp=float(np.median([v["early_gain_pp"] for v in rounds])),
                   gain_min_pp=min(v["early_gain_pp"] for v in rounds),
                   round_win_count=sum(v["early_gain_pp"] > 0 for v in rounds),
                   round_loss_count=sum(v["early_gain_pp"] < 0 for v in rounds),
                   reading="decline_observed" if delta < -1 else "increase_observed" if delta > 1 else "within_1pp")
        group_records.append(row)
    with (output / "round_layers.csv").open("x", newline="") as f:
        write_rows([f, None], round_rows)
    weighted = [row for row in rows for _ in row["layers"].split("|")]
    summary = dict(status="complete", reader_revision=source_commit(), cold_windows=8,
                   resident_publications=1, physical_block=BLOCK,
                   request_rows=len(rows), round_layer_rows=len(round_rows),
                   overall_layer_weighted=aggregate(weighted), groups=group_records,
                   scope="Recorded physical permutations; Q4-union cold demand; no synthetic ID sorting",
                   validation="Series/manifests + selected NPZ members CRC/dtype/shape/permutation/TopK-ID consistency; full score archive checksum not reread",
                   limits=["scanEarly/scanLead are discovery/publication opportunities without threshold or capacity gates, not actual prefetch or transfer completion",
                           "Recent scoring does not advance native candidate publication in this fixed-threshold baseline",
                           "Cold demand excludes hot and ABC; selected-hot-victim refetch is not counted as cold",
                           "Each round uses captured original cache state, not a counterfactual rollout",
                           "First/last and correlations show associations; miss/query/content changes confound causal attribution",
                           "These 100 rounds may already begin with a scrambled Index; they do not represent request start"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("LAYOUT_SURVEY_BEGIN cold_windows=8 resident_publications=1")
    print("G layers    miss% movedBlock% FscanEarly% RscanEarly% gain_first->last_pp R-Flead_pp win/loss")
    for row in group_records:
        nums = row["layers"].split("|")
        label = nums[0] if len(nums) == 1 else nums[0] + "-" + nums[-1]
        print(f"{row['group']:2} {label:7} {row['miss_pct']:6.2f} {row['displaced_block_pct']:11.2f} "
              f"{row['forward_early_pct']:11.2f} {row['reverse_early_pct']:11.2f} "
              f"{row['first_rounds']['early_gain_pp']:+7.2f}->{row['last_rounds']['early_gain_pp']:+7.2f} "
              f"{row['lead_gain_pp']:+10.2f} {row['round_win_count']}/{row['round_loss_count']}")
    for row in group_records:
        if row["reading"] == "decline_observed":
            def fmt(value):
                return "NA" if value is None else f"{value:+.3f}"
            print(f"DECAY G{row['group']} gain_change={row['early_gain_change_pp']:+.2f}pp "
                  f"movedBlock={row['first_rounds']['displaced_block_pct']:.2f}->{row['last_rounds']['displaced_block_pct']:.2f}% "
                  f"corr(disorder,gain)={fmt(row['disorder_gain_correlation'])} association_not_causation")
    all_rows = summary["overall_layer_weighted"]
    print(f"TOTAL manifests={len(rows)//2} request_rows={len(rows)} round_layer_rows={len(round_rows)} "
          f"FscanEarly={all_rows['forward_early_pct']:.2f}% RscanEarly={all_rows['reverse_early_pct']:.2f}%")
    print("DISCOVERY_ONLY: no claim of achieved prefetch/bandwidth/speedup")
    print(f"LAYOUT_SURVEY_PASS RESULT_DIR={output}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--cold-windows", type=int, choices=(8,), default=8)
    p.add_argument("--expect-dp", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--expect-rounds", type=int, default=101)
    p.add_argument("--expect-layers", type=int, default=78)
    args = p.parse_args(argv)
    if args.workers < 1 or args.result_dir.exists():
        p.error("Require positive workers and a new result directory")
    reports = inspect_series(args.root, expected_dp=args.expect_dp, expected_rounds=args.expect_rounds,
                             expected_layers=args.expect_layers, expected_content="scores")
    tasks = survey_tasks(args.root, reports)
    args.result_dir.mkdir(parents=True)
    (args.result_dir / "INCOMPLETE").touch()
    print(f"LAYOUT_SURVEY_START tasks={len(tasks)} workers={args.workers} cold_windows=8 ID_members_only", flush=True)
    rows, done, tick = [], 0, time.monotonic()
    with gzip.open(args.result_dir / "requests.csv.gz", "xt", newline="") as req, gzip.open(
            args.result_dir / "physical_blocks.csv.gz", "xt", newline="") as block_file:
        rw, bw = [req, None], [block_file, None]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            it = iter(tasks)
            pending = {pool.submit(task_run, task) for task in list(tasks)[:args.workers]}
            for _ in range(min(args.workers, len(tasks))):
                next(it)
            while pending:
                ready, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in ready:
                    local, blocks = future.result()
                    rows.extend(local)
                    write_rows(rw, local)
                    write_rows(bw, blocks)
                    done += 1
                    task = next(it, None)
                    if task is not None:
                        pending.add(pool.submit(task_run, task))
                if time.monotonic() - tick >= 60:
                    print(f"LAYOUT_PROGRESS groups={done}/{len(tasks)}", flush=True)
                    tick = time.monotonic()
    finish(rows, args.result_dir, args.expect_rounds)
    (args.result_dir / "INCOMPLETE").unlink()


if __name__ == "__main__":
    main()
