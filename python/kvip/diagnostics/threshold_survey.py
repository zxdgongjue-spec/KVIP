# SPDX-License-Identifier: Apache-2.0
"""Survey captured score thresholds on CPU, without replaying an operator."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time

from kvip.diagnostics.archive import checked_path, source_commit
from kvip.diagnostics.token_trace import load_token_trace
from kvip.diagnostics.trajectory import inspect_series

POLICIES = ("hot", "recent", "seed_hot", "rolling", "final")
POLICY_COUNTS = (
    "candidates",
    "selected_candidates",
    "extra_candidates",
    "extra_ties",
    "extra_below",
    "cold_extra",
    "candidate_hot_extra",
    "extra_bytes",
)
COUNTS = (
    "selected_unique",
    "cold_demand",
    "candidate_hot_refetch",
    "hot_hit",
    "abc_hit",
    "actual_prefetch",
    "actual_final",
    "actual_unused_prefetch",
    "actual_unused_cold",
    "actual_unused_candidate_hot",
    "actual_unused_other_hot",
    "actual_unused_abc",
    "actual_unused_final_ties",
    "actual_unused_below_final",
    "actual_read_bytes",
    "actual_unused_bytes",
    "gap_query_count",
    "gap_equal_queries",
    "gap_finite_count",
    "recent_equal_queries",
    "recent_finite_count",
    "retired_abc_positions",
    "retired_abc_included",
    "previous_topk_positions",
    "previous_topk_covered_by_recent",
    "recent_added_positions",
) + tuple(f"{policy}_{name}" for policy in POLICIES for name in POLICY_COUNTS)
MISS_EDGES = (0, 1, 2, 5, 10, 15, 20, 100)
MIB = 1 << 20


@dataclass(frozen=True)
class SurveyTask:
    dp_rank: int
    group: int
    layer_names: tuple
    signature: tuple
    snapshots: tuple  # (capture index, coordinated scheduler round, manifest path)


def survey_tasks(root, reports):
    """Resolve the already inspected series once; never guess snapshot names."""
    tasks = []
    for report in sorted(reports, key=lambda item: item["dp_rank"]):
        directory = Path(root) / f"dp{report['dp_rank']}"
        series = json.loads((directory / "series.json").read_text())
        signature = tuple((item["request_id"], item["owner"]) for item in series["request_owner_signature"])
        for group in sorted(series["groups"], key=lambda item: item["group"]):
            snapshots = []
            for round_entry in series["rounds"]:
                entries = [entry for entry in round_entry["groups"] if entry["group"] == group["group"]]
                if len(entries) != 1:
                    raise ValueError("Inspected series changed while planning the survey")
                path = checked_path(directory, entries[0]["manifest"])
                snapshots.append((round_entry["round_index"], round_entry["scheduler_round"], str(path)))
            tasks.append(
                SurveyTask(report["dp_rank"], group["group"], tuple(group["layer_names"]), signature, tuple(snapshots))
            )
    return tasks


def _run_task(task):
    """Keep only the current archive and the preceding small TopK seed."""
    from kvip.diagnostics.threshold_stats import analyze_trace, seed_state

    requests, queries, previous = [], [], None
    for capture_round, scheduler_round, path in task.snapshots:
        trace = load_token_trace(Path(path).parent)
        try:
            manifest = trace.manifest
            metadata = manifest["metadata"]
            signature = tuple((item["request_id"], item["owner"]) for item in metadata["requests"])
            if (
                manifest.get("content") != "scores"
                or manifest.get("source_transport") != "ubmem"
                or metadata["dp_rank"] != task.dp_rank
                or metadata["group"] != task.group
                or metadata["context"]["round"] != scheduler_round
                or metadata["context"].get("global_decode_confirmed") is not True
                or tuple(metadata["layer_names"]) != task.layer_names
                or signature != task.signature
            ):
                raise ValueError(f"Snapshot differs from its inspected series: {path}")
            if previous is not None:
                current_requests, current_queries = analyze_trace(trace, previous)
                if not current_requests or not current_queries:
                    raise ValueError(f"Analysis returned no request/query rows: {path}")
                for rows in (current_requests, current_queries):
                    for row in rows:
                        if (
                            row["dp_rank"] != task.dp_rank
                            or row["group"] != task.group
                            or row["scheduler_round"] != scheduler_round
                        ):
                            raise ValueError("Analysis identity differs from the selected snapshot")
                        row["capture_round"] = capture_round
                requests.extend(current_requests)
                queries.extend(current_queries)
            previous = seed_state(trace)
        finally:
            trace.close()
    return requests, queries


class CsvRows:
    """Incremental CSV output, rejecting a changing row schema."""

    def __init__(self, path):
        self.file = Path(path).open("x", newline="")
        self.writer = None
        self.fields = None
        self.count = 0

    def write(self, rows):
        for row in rows:
            if self.writer is None:
                preferred = ("dp_rank", "capture_round", "scheduler_round", "group", "packed_row", "owner")
                self.fields = [name for name in preferred if name in row]
                self.fields += sorted(set(row) - set(self.fields))
                self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
                self.writer.writeheader()
            if set(row) != set(self.fields):
                raise ValueError("Survey CSV rows have inconsistent fields")
            self.writer.writerow(row)
            self.count += 1
        self.file.flush()

    def close(self):
        self.file.close()


def _ratio(numerator, denominator, multiplier=1.0):
    return numerator / denominator * multiplier if denominator else None


def miss_bin(cold, selected):
    if type(cold) is not int or type(selected) is not int or not 0 <= cold <= selected or selected <= 0:
        raise ValueError("Miss bins require positive selected demand and cold within that demand")
    for lower, upper in zip(MISS_EDGES, MISS_EDGES[1:]):
        if cold * 100 < upper * selected or upper == 100:
            return f"[{lower},{upper}{']' if upper == 100 else ')'}%"
    raise AssertionError("Unreachable miss bin")


class Totals:
    """Sum request-level counts; score gaps are query-weighted, not ratio-averaged."""

    def __init__(self):
        self.requests = 0
        self.samples = set()
        self.widths = set()
        self.counts = {name: 0 for name in COUNTS}
        self.gap_sum = 0.0
        self.gap_max = None
        self.recent_gap_sum = 0.0
        self.recent_gap_max = None
        self.recent_range_max = 0

    def add(self, row, *, layer=False):
        width, layers = row["token_bytes"], row["layer_count"]
        if type(width) is not int or type(layers) is not int or width <= 0 or layers <= 0 or width % layers:
            raise ValueError("Group bytes must divide exactly across its declared layers")
        divisor = layers if layer else 1
        for name in COUNTS:
            value = row[name]
            if type(value) is not int or value < 0:
                raise ValueError(f"Expected a nonnegative integer survey count: {name}")
            if name.endswith("_bytes"):
                if value % divisor:
                    raise ValueError(f"Group byte count cannot be split across layers: {name}")
                value //= divisor
            self.counts[name] += value
        if not 0 <= row["cold_demand"] <= row["selected_unique"] or row["selected_unique"] == 0:
            raise ValueError("Invalid selected/cold request counts")
        gap_sum, gap_max = row["gap_sum"], row["gap_max"]
        if not isinstance(gap_sum, (int, float)) or not math.isfinite(gap_sum):
            raise ValueError("gap_sum must contain only finite query gaps")
        self.gap_sum += gap_sum
        recent_sum, recent_max = row["recent_gap_sum"], row["recent_gap_max"]
        if not isinstance(recent_sum, (int, float)) or not math.isfinite(recent_sum):
            raise ValueError("recent_gap_sum must contain only finite query gaps")
        self.recent_gap_sum += recent_sum
        self.recent_range_max = max(self.recent_range_max, row["recent_range_positions_max"])
        if row["recent_finite_count"]:
            if not isinstance(recent_max, (int, float)) or not math.isfinite(recent_max):
                raise ValueError("recent_gap_max must be finite when finite gaps exist")
            self.recent_gap_max = (
                max(recent_max, self.recent_gap_max) if self.recent_gap_max is not None else recent_max
            )
        if row["gap_finite_count"]:
            if not isinstance(gap_max, (int, float)) or not math.isfinite(gap_max):
                raise ValueError("gap_max must be finite when finite gaps exist")
            self.gap_max = max(gap_max, self.gap_max) if self.gap_max is not None else gap_max
        self.requests += 1
        self.samples.add((row["dp_rank"], row["capture_round"], row["group"]))
        self.widths.add(width // divisor)

    def record(self):
        counts = self.counts
        result = dict(
            requests=self.requests,
            samples=len(self.samples),
            token_bytes="|".join(map(str, sorted(self.widths))),
            **counts,
            cold_miss_percent=_ratio(counts["cold_demand"], counts["selected_unique"], 100),
            gap_mean=_ratio(self.gap_sum, counts["gap_finite_count"]),
            gap_max=self.gap_max,
            gap_equal_percent=_ratio(counts["gap_equal_queries"], counts["gap_query_count"], 100),
            recent_gap_mean=_ratio(self.recent_gap_sum, counts["recent_finite_count"]),
            recent_gap_max=self.recent_gap_max,
            recent_equal_percent=_ratio(counts["recent_equal_queries"], counts["gap_query_count"], 100),
            retired_abc_coverage_percent=_ratio(counts["retired_abc_included"], counts["retired_abc_positions"], 100),
            previous_topk_recent_coverage_percent=_ratio(
                counts["previous_topk_covered_by_recent"], counts["previous_topk_positions"], 100
            ),
            recent_added_positions_per_query=_ratio(counts["recent_added_positions"], counts["gap_query_count"]),
            recent_range_positions_max=self.recent_range_max,
            actual_prefetch_precision_percent=_ratio(
                counts["actual_prefetch"] - counts["actual_unused_prefetch"], counts["actual_prefetch"], 100
            ),
            actual_prefetch_waste_percent=_ratio(counts["actual_unused_prefetch"], counts["actual_prefetch"], 100),
            actual_unused_vs_cold_percent=_ratio(counts["actual_unused_prefetch"], counts["cold_demand"], 100),
            actual_unused_mib_per_request=_ratio(counts["actual_unused_bytes"], self.requests * MIB),
            actual_unused_656_mib_per_request=_ratio(counts["actual_unused_prefetch"] * 656, self.requests * MIB),
        )
        for policy in POLICIES:
            result[f"{policy}_precision_percent"] = _ratio(
                counts[f"{policy}_selected_candidates"], counts[f"{policy}_candidates"], 100
            )
            result[f"{policy}_extra_mib_per_request"] = _ratio(counts[f"{policy}_extra_bytes"], self.requests * MIB)
            result[f"{policy}_extra_656_mib_per_request"] = _ratio(
                counts[f"{policy}_extra_candidates"] * 656, self.requests * MIB
            )
            if policy != "hot":
                result[f"{policy}_extra_reduction_percent"] = _ratio(
                    counts["hot_extra_candidates"] - counts[f"{policy}_extra_candidates"],
                    counts["hot_extra_candidates"],
                    100,
                )
        return result


class SurveyTotals:
    def __init__(self):
        self.overall = Totals()
        self.groups, self.round_layers, self.layers, self.bins = {}, {}, {}, {}
        self.group_names = {}

    def add(self, row):
        names = tuple(row["layer_names"].split("|"))
        if len(names) != row["layer_count"] or len(set(names)) != len(names) or not all(names):
            raise ValueError("Request layer names differ from its layer count")
        group = row["group"]
        if self.group_names.setdefault(group, names) != names:
            raise ValueError("A group maps to different layers across requests")
        self.overall.add(row)
        self.groups.setdefault(group, Totals()).add(row)
        bucket = miss_bin(row["cold_demand"], row["selected_unique"])
        self.bins.setdefault(bucket, Totals()).add(row)
        for name in names:
            key = (row["capture_round"], row["scheduler_round"], group, name)
            self.round_layers.setdefault(key, Totals()).add(row, layer=True)
            self.layers.setdefault((group, name), Totals()).add(row, layer=True)

    def group_rows(self):
        return [
            dict(group=group, layer_names="|".join(self.group_names[group]), **total.record())
            for group, total in sorted(self.groups.items())
        ]

    def layer_rows(self):
        return [
            dict(group=group, layer_name=name, **total.record()) for (group, name), total in sorted(self.layers.items())
        ]

    def round_layer_rows(self):
        return [
            dict(capture_round=capture, scheduler_round=scheduler, group=group, layer_name=name, **total.record())
            for (capture, scheduler, group, name), total in sorted(self.round_layers.items())
        ]

    def bin_rows(self):
        rows = []
        for lower, upper in zip(MISS_EDGES, MISS_EDGES[1:]):
            label = f"[{lower},{upper}{']' if upper == 100 else ')'}%"
            rows.append(dict(miss_bin=label, **self.bins.get(label, Totals()).record()))
        return rows


def _write_csv(path, rows):
    output = CsvRows(path)
    try:
        output.write(rows)
    finally:
        output.close()


def _number(value):
    return "--" if value is None else f"{value:.3f}"


def print_summary(summary):
    print("THRESHOLD_TABLE units=MiB/request normalized_to_656B; policy=uncapped_prediction actual=observed_UBmem")
    columns = (
        "cold_miss_percent",
        "gap_equal_percent",
        "gap_mean",
        "recent_gap_mean",
        "hot_extra_656_mib_per_request",
        "actual_unused_656_mib_per_request",
        "actual_prefetch_waste_percent",
        "seed_hot_extra_reduction_percent",
        "rolling_extra_reduction_percent",
        "recent_extra_reduction_percent",
        "recent_equal_percent",
        "retired_abc_coverage_percent",
        "previous_topk_recent_coverage_percent",
    )
    header = "miss% gap_equal% old_gap recent_gap hot_extra_MiB actual_unused_MiB actual_waste% seed_reduce% rolling_reduce% recent_reduce% recent_equal% retired_covered% old_topk_covered%"

    def line(label, row, *, mixed_scales=False):
        values = dict(row)
        if mixed_scales:
            values["gap_mean"] = values["gap_max"] = None
            values["recent_gap_mean"] = None
        print(f"{label} " + " ".join(_number(values[name]) for name in columns))

    print(f"SAMPLES requests={summary['request_rows']} queries={summary['query_rows']}")
    print(
        "RECENT_RANGE "
        f"mean_added_positions_per_query={_number(summary['overall']['recent_added_positions_per_query'])} "
        f"max_full_range_positions={summary['overall']['recent_range_positions_max']} no_truncation=true"
    )
    print(
        "MISS_SCOPE OVERALL=unique_request_groups "
        f"all_layers_weighted_miss%={_number(summary['all_layers_weighted_cold_miss_percent'])}"
    )
    print("GROUP " + header)
    line("OVERALL", summary["overall"], mixed_scales=True)
    ordered = sorted(summary["groups"], key=lambda row: row["hot_extra_656_mib_per_request"] or 0, reverse=True)
    for row in ordered:
        line(f"G{row['group']}", row)
    print("MISS_BIN " + header)
    for row in summary["miss_bins"]:
        line(row["miss_bin"], row, mixed_scales=True)
    print("GAP_NOTE numeric gap scales differ by group; pooled/bin numeric gaps are not used to judge accuracy.")


def run_survey(args):
    root, result = args.root.resolve(), args.result_dir.resolve()
    reports = inspect_series(
        root,
        expected_dp=args.expect_dp,
        expected_rounds=args.expect_rounds,
        expected_layers=args.expect_layers,
        expected_content="scores",
    )
    tasks = survey_tasks(root, reports)
    result.mkdir(parents=True, exist_ok=False)
    marker = result / "INCOMPLETE"
    marker.write_text("Survey output is incomplete until summary.json has status=complete.\n")
    requests, queries = CsvRows(result / "requests.csv"), CsvRows(result / "queries.csv")
    totals = SurveyTotals()
    started = last_progress = time.monotonic()
    completed = 0
    print(f"THRESHOLD_BEGIN tasks={len(tasks)} workers={args.workers} source={root} output={result}", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            iterator = iter(tasks)
            pending = {pool.submit(_run_task, task): task for task in list(tasks)[: args.workers]}
            for _ in pending:
                next(iterator)
            while pending:
                ready, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in ready:
                    task = pending.pop(future)
                    try:
                        request_rows, query_rows = future.result()
                    except Exception as error:
                        for waiting in pending:
                            waiting.cancel()
                        raise RuntimeError(f"DP{task.dp_rank} group{task.group}: {error}") from error
                    requests.write(request_rows)
                    queries.write(query_rows)
                    for row in request_rows:
                        totals.add(row)
                    completed += 1
                    next_task = next(iterator, None)
                    if next_task is not None:
                        pending[pool.submit(_run_task, next_task)] = next_task
                now = time.monotonic()
                if now - last_progress >= 30:
                    print(
                        f"THRESHOLD_PROGRESS tasks={completed}/{len(tasks)} requests={requests.count} "
                        f"elapsed_s={now - started:.0f}",
                        flush=True,
                    )
                    last_progress = now
        _write_csv(result / "round_layers.csv", totals.round_layer_rows())
        _write_csv(result / "layers.csv", totals.layer_rows())
        _write_csv(result / "miss_bins.csv", totals.bin_rows())
        requests.close()
        queries.close()
        summary = dict(
            schema_version=2,
            kind="kvaf_threshold_survey",
            status="complete",
            root=str(root),
            kvip_commit=source_commit(),
            source_series=reports,
            seed_rounds=1,
            analyzed_capture_rounds=[2, reports[0]["rounds"]],
            request_rows=requests.count,
            query_rows=queries.count,
            all_layers_weighted_cold_miss_percent=_ratio(
                sum(item.counts["cold_demand"] for item in totals.layers.values()),
                sum(item.counts["selected_unique"] for item in totals.layers.values()),
                100,
            ),
            scope={
                "actual": "Observed UBmem service prefetch/final transfers; not a URMA measurement.",
                "ties": (
                    "Unused tokens touching any causal query's final BF16 boundary are separated from "
                    "tokens below every final query boundary. This does not identify which query caused a transfer."
                ),
                "predicted": "Uncapped threshold-gate candidate model; candidate bytes are not observed I/O.",
                "candidate_domain": (
                    "Fixed counterfactual domain: LRU candidate slots union cold physical suffix, "
                    "with readable addresses, causal physical bounds and ABC exclusion. "
                    "The capture does not record cold_only_prefetch; this does not reconstruct its "
                    "switch, native window timing, budget or admission order."
                ),
                "hot": "Actual resident-prefix initial threshold; do not filter that prefix by hot_valid.",
                "seed_hot": (
                    "Current resident physical prefix union previous-round Q4 TopK logical-ID union; "
                    "rescored separately by each current query, then its Kth score is used."
                ),
                "recent": (
                    "Resident physical prefix union [previous-round ABC beginning,current valid query end), "
                    "clipped separately for each current query; exact Kth current score, overlap counted once. "
                    "Includes retired ABC before the current ABC beginning. No silent length cap; "
                    "logical-to-physical identity in the recent interval is verified."
                ),
                "recent_coverage": (
                    "Retired ABC and predecessor TopK coverage counts are query-weighted positions. "
                    "They describe score-seed coverage, not cache misses or observed KV transfers."
                ),
                "bounded_vs_full": (
                    "Recent uses the full predecessor-to-current interval. Policy candidates apply "
                    "causality/residency/address/ABC gates but no window or per-round transfer cap; "
                    "these are uncapped unique hypothetical candidates. Actual transfer columns retain "
                    "the recorded bounded kernel's events, including repeats. Neither predicts latency."
                ),
                "rolling": (
                    "Begin with resident threshold; raise it after each additional 16K physical prefix "
                    "and apply only to later segments. Previously admitted candidates are not removed."
                ),
                "timing": "CPU score analysis only; no policy, LI, Fetch or service latency is measured.",
                "layer_rows": "Shared Index-group observations expanded to layers; bytes divided by layer_count.",
                "overall": "Each request/group counted once; layer expansions are not summed into these totals.",
                "miss_bins": "Per-request cold/selected_unique; left-closed/right-open percent bins, final bin includes 100%.",
                "normalized_bytes": "656-byte token equivalents; actual_read_bytes retains the real group width.",
                "reductions": "Reduction of predicted extra candidate counts relative to hot, not runtime speedup.",
                "gap": (
                    "Numeric gap scales differ across groups. Per-group mean/max and pooled equality rate "
                    "are reported; cross-group numeric averages must not be used as an accuracy criterion."
                ),
                "precision": (
                    "Policy precision=selected candidates/candidates; actual prefetch precision=1-unused/prefetch. "
                    "Unused/cold is a traffic-to-demand ratio and can exceed 100%; zero denominators are null."
                ),
            },
            overall=totals.overall.record(),
            groups=totals.group_rows(),
            miss_bins=totals.bin_rows(),
            files={name: f"{name}.csv" for name in ("requests", "queries", "round_layers", "layers", "miss_bins")},
            elapsed_seconds=time.monotonic() - started,
        )
        temporary = result / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        temporary.replace(result / "summary.json")
        marker.unlink()
        print_summary(summary)
        print(f"THRESHOLD_SURVEY_PASS summary={result / 'summary.json'}", flush=True)
        return summary
    finally:
        requests.close()
        queries.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--result-dir", type=Path, required=True, help="New output directory; existing directories are refused"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expect-dp", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--expect-rounds", type=int, default=101, help="Includes seed capture round 1")
    parser.add_argument("--expect-layers", type=int, default=78)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 64:
        parser.error("--workers must be in 1..64")
    if args.expect_rounds < 2 or args.expect_layers < 1:
        parser.error("Scores require at least two captured rounds and one layer")
    if (
        not args.expect_dp
        or any(rank < 0 for rank in args.expect_dp)
        or len(set(args.expect_dp)) != len(args.expect_dp)
    ):
        parser.error("--expect-dp requires distinct nonnegative ranks")
    try:
        run_survey(args)
    except (ValueError, KeyError, OSError, AssertionError, RuntimeError) as error:
        print(f"THRESHOLD_SURVEY_FAILED: {error}; incomplete_output={args.result_dir}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
