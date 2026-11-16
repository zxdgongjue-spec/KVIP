# SPDX-License-Identifier: Apache-2.0
"""CPU-only traffic audit for low rounds that retain hot KV without promotion.

This is a counterfactual against captured TopK IDs, not a timing benchmark or
an autonomous-route simulation. Each span starts at the captured hot state;
that hot state is held fixed for 1/2/4/8 rounds while ABC advances normally.
Only ID/state NPZ members are decompressed, never scores or real LI inputs.
"""

import argparse
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

    def add(self, state, retained_hot, *, rank, capture_round):
        baseline, retained = state.cold(state.hot), state.cold(retained_hot)
        newly, saved = len(retained - baseline), len(baseline - retained)
        self.samples += 1
        self.selected += len(state.selected)
        self.baseline_cold_reads += len(baseline)
        self.retained_cold_reads += len(retained)
        self.newly_missing_reads += newly
        self.saved_reads += saved
        self.recorded_total_gets += state.recorded_gets
        extra = len(retained) - len(baseline)
        if not self.worst or extra > self.worst["net_extra_reads"]:
            self.worst = dict(dp_rank=rank, capture_round=capture_round,
                              request_id=state.identity[0], owner=state.identity[1],
                              baseline_cold_reads=len(baseline),
                              retained_cold_reads=len(retained), net_extra_reads=extra)

    def report(self, token_bytes):
        values = dict(vars(self))
        extra = self.retained_cold_reads - self.baseline_cold_reads
        assert extra == self.newly_missing_reads - self.saved_reads
        values.update(
            net_extra_reads=extra,
            net_extra_percent=(100 * extra / self.baseline_cold_reads
                               if self.baseline_cold_reads else None),
            baseline_miss_percent=100 * self.baseline_cold_reads / self.selected,
            retained_miss_percent=100 * self.retained_cold_reads / self.selected,
            token_bytes=token_bytes,
            net_extra_mib=extra * token_bytes / (1 << 20),
        )
        return values


def audit_sequence(frames, *, group, rank, spans, totals=None):
    """Each non-overlapping span resets to recorded hot, including a last partial span."""
    totals = totals if totals is not None else {span: TrafficTotals(group, span) for span in spans}
    retained = {}
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
            for state, hot in zip(states, retained[span]):
                totals[span].add(state, hot, rank=rank, capture_round=capture_round)
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
                       totals=totals[task.group])
    return dict(
        schema_version=1, kind="read_through_traffic_audit", status="complete",
        trace_root=str(Path(args.root).resolve()), start_round=args.start_round,
        dp_ranks=[report["dp_rank"] for report in reports],
        scope={
            "selection": "Captured logical Q4 TopK union; exact deduplication within each request/round.",
            "policy": "Non-overlapping low spans; freeze first-round hot IDs, advance recorded ABC, never promote scratch KV.",
            "reset": "Each span resets to recorded pre-state; last partial span is included. No refresh cost is modeled.",
            "baseline": "Necessary cold reads under recorded hot/ABC, not total serial or fused Get submissions.",
            "recorded_total_gets": "Observed fusion transfers, including prefetch and hot refetch; reference only.",
            "limits": "Counterfactual traffic only, not elapsed time, autonomous route behavior or future numerical output.",
            "integrity": "Series manifests plus selected NPZ member CRCs and demand checks; whole score archive SHA is not reread.",
            "excluded_payloads": "Scores, real Q/Index, and KV payload are never loaded.",
        },
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
    args = parser.parse_args(argv)
    try:
        result = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
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
        print(f"READ_THROUGH_TRAFFIC_PASS cpu_only=true scores_loaded=false RESULT={args.output}")
        print("SCOPE: captured TopK traffic counterfactual; not latency or autonomous route performance.")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"READ_THROUGH_TRAFFIC_FAILED: {error}\n")


if __name__ == "__main__":
    main()
