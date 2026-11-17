# SPDX-License-Identifier: Apache-2.0
"""CPU-only traffic audit for low rounds that retain hot KV without promotion.

This is a counterfactual against captured TopK IDs, not a timing benchmark or
an autonomous-route simulation. Each span starts at the captured hot state;
that hot state is held fixed for the requested span while ABC advances normally.
Only ID/state NPZ members are decompressed, never scores or real LI inputs.
"""

import argparse
from contextlib import ExitStack
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from kvip.diagnostics.archive import checked_path
from kvip.diagnostics.threshold_survey import survey_tasks
from kvip.diagnostics.trajectory import inspect_series


MEMBERS = (
    "pre.logical_ids", "pre.hot_valid", "pre.tail_bounds", "pre.metadata",
    "post.logical_output", "post.topk",
)
READ_CLASSES = ("first_other", "first_abc_retired", "repeat_other", "repeat_abc_retired")
BREAKDOWN = tuple(f"{name}_reads" for name in READ_CLASSES) + tuple(
    f"new_{name}_reads" for name in READ_CLASSES
)
COUNTERS = (
    "samples", "selected", "baseline_cold_reads", "retained_cold_reads",
    "newly_missing_reads", "saved_reads", "recorded_total_gets",
) + BREAKDOWN
CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64, 100)


@dataclass(frozen=True)
class RequestState:
    identity: tuple
    hot: frozenset
    selected: frozenset
    abc_begin: int
    abc_end: int
    recorded_gets: int

    def cold(self, hot):
        return {token for token in self.selected
                if token not in hot and not self.abc_begin <= token < self.abc_end}


def read_requests(path):
    """Read validated trace IDs lazily; do not hash/read the whole score ZIP."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    g, metadata = manifest["geometry"], manifest["metadata"]
    b, h, resident = (g[key] for key in ("batch", "history", "resident"))
    expected = {
        "pre.logical_ids": (np.dtype("int32"), (b, h)),
        "pre.hot_valid": (np.dtype("uint8"), (b, resident)),
        "pre.tail_bounds": (np.dtype("int32"), (b, 2)),
        "pre.metadata": (np.dtype("int64"), (5, b * 4)),
        "post.logical_output": (np.dtype("int32"), (b * 4, 2048)),
        "post.topk": (np.dtype("int32"), (b, 4, 2048)),
    }
    archive = checked_path(path.parent, manifest["archive"]["file"])
    # NpzFile is lazy: accessing a member verifies that member's ZIP CRC.
    # Whole-file SHA verification would read the intentionally skipped scores.
    with np.load(archive, allow_pickle=False) as source:
        arrays = {name: source[name] for name in MEMBERS}
    for name, (dtype, shape) in expected.items():
        value = arrays[name]
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(f"Unexpected ID/state member dtype or shape: {path}: {name}")
    unpack = arrays["pre.metadata"][2]
    if not np.array_equal(np.sort(unpack), np.arange(b * 4)):
        raise ValueError(f"Invalid packed/TND query mapping: {path}")
    summaries = {row["packed_row"]: row for row in manifest["summary"]["requests"]}
    result = []
    for row, request in enumerate(metadata["requests"]):
        ids, valid = arrays["pre.logical_ids"][row], arrays["pre.hot_valid"][row]
        begin, end = map(int, arrays["pre.tail_bounds"][row])
        if not 0 <= begin <= end <= h or end - begin > g["output_stride"] - resident:
            raise ValueError(f"Invalid ABC interval: {path}")
        if np.any(valid > 1) or len(np.unique(ids)) != h or np.any(ids < 0) or np.any(ids >= h):
            raise ValueError(f"Invalid Index permutation/hot validity: {path}")
        if not np.array_equal(ids[begin:end], np.arange(begin, end)):
            raise ValueError(f"ABC logical and physical coordinates differ: {path}")
        packed = arrays["post.topk"][row]
        if np.any(packed < 0) or np.any(packed >= h):
            raise ValueError(f"TopK position outside Index history: {path}")
        logical = ids[packed]
        for tnd in np.flatnonzero(unpack // 4 == row):
            if not np.array_equal(logical[unpack[tnd] % 4], arrays["post.logical_output"][tnd]):
                raise ValueError(f"Physical TopK and logical output differ: {path}")
        state = RequestState(
            (str(request["request_id"]), int(request["owner"])),
            frozenset(map(int, ids[:resident][valid != 0])),
            frozenset(map(int, logical.reshape(-1))), begin, end,
            int(summaries[row]["reads"]["total"]),
        )
        if (len(state.selected) != summaries[row]["selected_unique"]
                or len(state.cold(state.hot)) != summaries[row]["cold_demand"]):
            raise ValueError(f"Recomputed demand differs from captured manifest: {path}")
        result.append(state)
    return manifest, result


@dataclass
class TrafficTotals:
    group: int
    span: int
    samples: int = 0
    selected: int = 0
    baseline_cold_reads: int = 0
    retained_cold_reads: int = 0
    newly_missing_reads: int = 0
    saved_reads: int = 0
    recorded_total_gets: int = 0
    worst: dict = field(default_factory=dict)
    breakdown: dict = field(default_factory=lambda: dict.fromkeys(BREAKDOWN, 0))
    by_age: dict = field(default_factory=dict)

    def add(self, state, retained_hot, *, rank, capture_round, scheduler_round,
            age, seen_reads, seen_abc):
        baseline, retained = state.cold(state.hot), state.cold(retained_hot)
        newly, saved = retained - baseline, baseline - retained
        if age == 1:
            assert not newly and not saved  # First-state identity, not zero cold demand.
        # Every retained read is outside current ABC by construction. "Retired"
        # means this token was in ABC earlier in this same low span.
        retired, repeated = retained & seen_abc, retained & seen_reads
        classes = {
            "first_other": retained - retired - repeated,
            "first_abc_retired": retired - repeated,
            "repeat_other": repeated - retired,
            "repeat_abc_retired": repeated & retired,
        }
        counts = dict(samples=1, selected=len(state.selected),
                      baseline_cold_reads=len(baseline), retained_cold_reads=len(retained),
                      newly_missing_reads=len(newly), saved_reads=len(saved),
                      recorded_total_gets=state.recorded_gets)
        counts.update({f"{name}_reads": len(tokens) for name, tokens in classes.items()})
        counts.update({f"new_{name}_reads": len(tokens & newly) for name, tokens in classes.items()})
        assert sum(len(tokens) for tokens in classes.values()) == len(retained)
        assert sum(len(tokens & newly) for tokens in classes.values()) == len(newly)
        age_counts = self.by_age.setdefault(age, dict.fromkeys(COUNTERS, 0))
        for name, value in counts.items():
            age_counts[name] += value
            if name in self.breakdown:
                self.breakdown[name] += value
            else:
                setattr(self, name, getattr(self, name) + value)
        extra = len(retained) - len(baseline)
        if not self.worst or extra > self.worst["net_extra_reads"]:
            self.worst = dict(dp_rank=rank, capture_round=capture_round,
                              request_id=state.identity[0], owner=state.identity[1],
                              baseline_cold_reads=len(baseline),
                              retained_cold_reads=len(retained), net_extra_reads=extra)
        seen_reads.update(retained)
        seen_abc.update(range(state.abc_begin, state.abc_end))
        return dict(group=self.group, span=self.span, age=age, dp_rank=rank,
                    capture_round=capture_round, scheduler_round=scheduler_round,
                    request_id=state.identity[0], owner=state.identity[1],
                    **counts, net_extra_reads=extra,
                    baseline_miss_percent=100 * len(baseline) / len(state.selected),
                    retained_miss_percent=100 * len(retained) / len(state.selected),
                    net_extra_percent=100 * extra / len(baseline) if baseline else None)

    def report(self, token_bytes):
        values = dict(vars(self))
        values.update(values.pop("breakdown"))
        values["by_age"] = [summarize_counts(dict(age=age, **counts), token_bytes)
                            for age, counts in sorted(self.by_age.items())]
        return summarize_counts(values, token_bytes)


def summarize_counts(values, token_bytes):
    extra = values["retained_cold_reads"] - values["baseline_cold_reads"]
    assert extra == values["newly_missing_reads"] - values["saved_reads"]
    values.update(
        net_extra_reads=extra,
        net_extra_percent=(100 * extra / values["baseline_cold_reads"]
                           if values["baseline_cold_reads"] else None),
        baseline_miss_percent=100 * values["baseline_cold_reads"] / values["selected"],
        retained_miss_percent=100 * values["retained_cold_reads"] / values["selected"],
        token_bytes=token_bytes, net_extra_mib=extra * token_bytes / (1 << 20),
    )
    return values


def audit_sequence(frames, *, group, rank, spans, totals=None, write_detail=None):
    """Each non-overlapping span resets to recorded hot, including a last partial span."""
    totals = totals if totals is not None else {span: TrafficTotals(group, span) for span in spans}
    retained, read_history, abc_history = {}, {}, {}
    previous_round, identities = None, None
    for index, (capture_round, scheduler_round, states) in enumerate(frames):
        current_ids = tuple(state.identity for state in states)
        if previous_round is not None and scheduler_round != previous_round + 1:
            raise ValueError("Traffic audit requires consecutive scheduler rounds")
        if identities is not None and current_ids != identities:
            raise ValueError("Request/owner lifetime changed inside a low span")
        for span in spans:
            if index % span == 0:
                retained[span] = tuple(state.hot for state in states)
                read_history[span] = [set() for _ in states]
                abc_history[span] = [set() for _ in states]
            for row, (state, hot) in enumerate(zip(states, retained[span])):
                detail = totals[span].add(
                    state, hot, rank=rank, capture_round=capture_round,
                    scheduler_round=scheduler_round, age=index % span + 1,
                    seen_reads=read_history[span][row], seen_abc=abc_history[span][row],
                )
                if write_detail is not None:
                    write_detail(detail)
        previous_round, identities = scheduler_round, current_ids
    return totals


def run(args):
    reports = inspect_series(args.root, expected_dp=args.expect_dp,
                             expected_rounds=args.expect_rounds, expected_content="scores")
    groups = list(dict.fromkeys(args.groups))
    spans = sorted(set(args.spans))
    if not groups or min(groups) < 0 or not spans or min(spans) < 1:
        raise ValueError("Groups must be nonnegative; spans must be positive")
    if args.start_round < 2 or args.token_bytes <= 0:
        raise ValueError("Start round must exclude seed; token bytes must be positive")
    for report in reports:
        if not set(groups).issubset(report["selected_groups"]):
            raise ValueError(f"DP{report['dp_rank']} does not contain every requested group")
        if args.start_round > report["rounds"]:
            raise ValueError("Start round exceeds the capture")
    totals = {group: {span: TrafficTotals(group, span) for span in spans} for group in groups}
    tasks = [task for task in survey_tasks(args.root, reports) if task.group in groups]
    with ExitStack() as stack:
        write_detail = None
        if args.details_dir is not None:
            args.details_dir.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context((args.details_dir / "request_rounds.csv").open("w", newline=""))
            columns = ("group", "span", "age", "dp_rank", "capture_round", "scheduler_round",
                       "request_id", "owner") + COUNTERS + (
                           "net_extra_reads", "baseline_miss_percent", "retained_miss_percent", "net_extra_percent",
                       )
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            write_detail = writer.writerow
        for task in tasks:
            def frames():
                for capture_round, scheduler_round, path in task.snapshots:
                    if capture_round < args.start_round:
                        continue
                    manifest, states = read_requests(path)
                    metadata = manifest["metadata"]
                    if (metadata["dp_rank"] != task.dp_rank or metadata["group"] != task.group
                            or metadata["context"]["round"] != scheduler_round
                            or tuple(state.identity for state in states) != task.signature):
                        raise ValueError(f"Trace identity changed after inspection: {path}")
                    yield capture_round, scheduler_round, states
            audit_sequence(frames(), group=task.group, rank=task.dp_rank, spans=spans,
                           totals=totals[task.group], write_detail=write_detail)
    return dict(
        schema_version=2, kind="read_through_traffic_audit", status="complete",
        trace_root=str(Path(args.root).resolve()), start_round=args.start_round,
        dp_ranks=[report["dp_rank"] for report in reports],
        analysis_rounds=reports[0]["rounds"] - args.start_round + 1,
        scope={
            "selection": "Captured logical Q4 TopK union; exact deduplication within each request/round.",
            "policy": "Non-overlapping low spans; freeze first-round hot IDs, advance recorded ABC, never promote scratch KV.",
            "reset": "Each span starts with an oracle reset to recorded pre-state; last partial span is included. This is not an implemented refresh policy or a performance bound; no refresh cost is modeled.",
            "full_span": "A span covering all analysis rounds freezes the first hot state continuously with no intermediate reset.",
            "age": "One-based position within a low span; counts are aggregated at the same age, not cumulatively.",
            "read_classes": "first/repeat means not/previously actually read within this span, excluding any reads before the span. abc_retired means observed in ABC within this span but outside current ABC; other means no such retirement observed, not never historically in ABC. Four classes are mutually exclusive.",
            "new_read_classes": "The same four classes restricted to retained reads absent from baseline cold demand. Saved baseline reads are separate, never counted as retained reads.",
            "retirement_limit": "ABC residency before the span is unknown to this counterfactual; such tokens are classified other, not abc_retired. Span boundaries reset read and ABC histories.",
            "age_one": "At age 1 retained and baseline hot states are identical: newly_missing_reads=saved_reads=0, but cold reads need not be zero.",
            "baseline": "Necessary cold reads under recorded hot/ABC, not total serial or fused Get submissions.",
            "recorded_total_gets": "Observed fusion transfers, including prefetch and hot refetch; reference only.",
            "limits": "Counterfactual traffic only, not elapsed time, autonomous route behavior or future numerical output.",
            "integrity": "Series manifests plus selected NPZ member CRCs and demand checks; whole score archive SHA is not reread.",
            "excluded_payloads": "Scores, real Q/Index, and KV payload are never loaded.",
        },
        details_dir=None if args.details_dir is None else str(args.details_dir.resolve()),
        cases=[totals[group][span].report(args.token_bytes) for group in groups for span in spans],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--groups", type=int, nargs="+", default=[2, 7, 10, 17, 4])
    parser.add_argument("--spans", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--start-round", type=int, default=2)
    parser.add_argument("--token-bytes", type=int, default=656)
    parser.add_argument("--expect-dp", type=int, nargs="+")
    parser.add_argument("--expect-rounds", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path,
                        help="Write request_rounds.csv and by_age.csv; no tensor payloads")
    args = parser.parse_args(argv)
    try:
        result = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.details_dir is not None:
            rows = [dict(group=case["group"], span=case["span"], **age)
                    for case in result["cases"] for age in case["by_age"]]
            with (args.details_dir / "by_age.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        for row in result["cases"]:
            delta = row["net_extra_percent"]
            percent = "n/a" if delta is None else f"{delta:+.2f}%"
            worst = row["worst"]
            print(f"TRAFFIC G{row['group']} span={row['span']} samples={row['samples']} "
                  f"cold={row['baseline_cold_reads']}->{row['retained_cold_reads']} "
                  f"net={row['net_extra_reads']:+d}({percent}) "
                  f"new={row['newly_missing_reads']} saved={row['saved_reads']} "
                  f"worst=DP{worst['dp_rank']}/R{worst['capture_round']}/owner{worst['owner']}"
                  f":{worst['net_extra_reads']:+d}")
            if row["span"] == max(args.spans):
                for age in row["by_age"]:
                    if age["age"] not in CHECKPOINTS:
                        continue
                    print(f"AGE G{row['group']} span={row['span']} age={age['age']} "
                          f"miss={age['baseline_miss_percent']:.3f}%->{age['retained_miss_percent']:.3f}% "
                          f"net={age['net_extra_reads']:+d} "
                          f"first_other={age['first_other_reads']} first_abc_retired={age['first_abc_retired_reads']} "
                          f"repeat_other={age['repeat_other_reads']} repeat_abc_retired={age['repeat_abc_retired_reads']}")
        print(f"READ_THROUGH_TRAFFIC_PASS cpu_only=true scores_loaded=false RESULT={args.output}")
        print("SCOPE: captured TopK traffic only; span boundaries use oracle resets without refresh cost. "
              "A full-length span has no intermediate reset. No latency or autonomous route claim.")
        print("CLASSES: repeat=read earlier in this span; abc_retired=observed in ABC earlier in this span "
              "and now outside ABC; other=no such retirement observed. Pre-span history is excluded.")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"READ_THROUGH_TRAFFIC_FAILED: {error}\n")


if __name__ == "__main__":
    main()
