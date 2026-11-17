# SPDX-License-Identifier: Apache-2.0
"""Offline profiler attribution and real-window inputs; no device or latency model."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from kvip.diagnostics.archive import TensorArchive, source_commit
from kvip.diagnostics.incremental_topk import (
    TOP, TieWindowTopK, WindowTopK, eligibility, inputs, scan_windows,
)
from kvip.diagnostics.recent_threshold import recent_positions
from kvip.diagnostics.scan_report import load_report, traffic
from kvip.diagnostics.scan_economy_report import control, useful_prefetch
from kvip.diagnostics.system_profile import summarize_timeline

VARIANTS = ("reverse", "reverse-inc-register", "reverse-inc-inplace")


class CachedArchive(TensorArchive):
    def __init__(self, directory, manifest):
        super().__init__(directory, manifest)
        self.cached = {}

    def array(self, name, *, verify=True):
        if name not in self.cached:
            self.cached[name] = super().array(name, verify=verify)
        return self.cached[name]


def query_work(scores, seed, windows, eligible, *, k=TOP):
    """Exact ordered-prefix oracle and logical work, not modeled device time.

    Eligibility is the frozen snapshot mask. Published IDs may already have
    been posted by the asynchronous service; do not call them new DDR reads.
    """
    dense = WindowTopK(scores[seed], seed, k=k)
    sparse = TieWindowTopK(scores[seed], seed, k=k, chunk=k)
    seen = np.zeros(len(eligible), dtype=bool)
    seen[seed] = True
    previous = set(map(int, dense.ids[eligible[dense.ids]]))
    rows, selected, values, thresholds = [], [dense.ids.copy()], [dense.scores.copy()], [dense.threshold]
    for epoch, window in enumerate(windows, 1):
        valid = window[window < len(scores)]
        valid = valid[~seen[valid]]
        old_ids, old_threshold = dense.ids.copy(), dense.threshold
        strict = int(np.count_nonzero(dense.scores > old_threshold))
        admitted = int(np.count_nonzero(scores[valid] > old_threshold))
        dense.append(scores[valid], valid)
        sparse.append(scores[valid], valid)
        if (dense.threshold != sparse.threshold or not np.array_equal(dense.ids, sparse.ids)
                or not np.array_equal(dense.scores, sparse.scores)):
            raise AssertionError(f"Dense/sparse ordered prefix differs at cold window {epoch}")
        seen[valid] = True
        pending = set(map(int, dense.ids[eligible[dense.ids]]))
        ties = k - strict
        branch = ("empty" if not admitted else "dense" if admitted > k
                  else "inplace" if admitted < ties <= 256
                  else "tie" if admitted < ties else "sparse_kth")
        rows.append(dict(
            window=epoch, physical_positions=len(window), valid_new_positions=len(valid),
            first_filter_admitted=admitted, old_threshold=old_threshold, threshold=dense.threshold,
            old_strict=strict, old_ties=ties, inplace_branch=branch,
            retained_old_ids=len(set(map(int, old_ids)) & set(map(int, dense.ids))),
            publication_ids=len(pending), repeated_publication_ids=len(pending & previous),
            introduced_publication_ids=len(pending - previous), removed_publication_ids=len(previous - pending),
            # Source-level loop work. No cost is assigned to individual loops.
            register_selection_positions=(k + len(window) + 255) // 256 * 256,
            sparse_pack_passes=1 + int(0 < admitted <= k),
            sparse_pack_positions=len(window) * (1 + int(0 < admitted <= k)),
            sparse_selection_positions=0 if not admitted or admitted < ties else
                (k + (admitted if admitted <= k else len(window)) + 255) // 256 * 256,
        ))
        selected.append(dense.ids.copy())
        values.append(dense.scores.copy())
        thresholds.append(dense.threshold)
        previous = pending
    return rows, dict(expected_ids=np.asarray(selected, dtype=np.int32),
                      expected_scores=np.asarray(values, dtype=np.uint16),
                      expected_thresholds=np.asarray(thresholds, dtype=np.uint16))


def audit_snapshot(path, output, *, samples=4):
    raw = (path / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    archive = CachedArchive(path, manifest)
    geometry, scores, ends, bounds = inputs(archive)
    rows, queries, expected_queries, eligible_rows = [], [], [], []
    for row in range(geometry["batch"]):
        eligible = eligibility(archive, row)
        eligible_rows.append(eligible)
        windows = scan_windows(geometry["history"], geometry["resident"], ends[row], True)
        for q in range(4):
            values = scores[row, q, :int(ends[row, q])]
            seed = recent_positions(geometry["resident"], bounds[row], len(values))
            work, expected = query_work(values, seed, windows, eligible)
            expected_queries.append(expected)
            rows.extend(dict(row=row, query=q, **item) for item in work)
            queries.append((sum(item["first_filter_admitted"] for item in work), row, q))
    # Export all 64 query sequences. Quantiles are only suggested starting
    # points for component probes, not substitutes for full parallel work.
    queries.sort()
    chosen = [queries[i] for i in sorted(set(np.linspace(0, len(queries)-1, min(samples, len(queries)), dtype=int)))]
    descriptions = [dict(row=row, query=q, first_filter_admitted=admitted,
                         query_end=int(ends[row, q])) for admitted, row, q in chosen]
    arrays = dict(scores=scores, query_ends=ends, threshold_bounds=bounds,
                  tail_bounds=archive.array("pre.tail_bounds"), eligible=np.asarray(eligible_rows))
    for name in expected_queries[0]:
        stacked = np.asarray([item[name] for item in expected_queries])
        arrays[name] = stacked.reshape(geometry["batch"], 4, *stacked.shape[1:])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "windows.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "probe_inputs.npz", **arrays)
    counts = [item["first_filter_admitted"] for item in rows]
    summary = dict(
        snapshot=str(path), snapshot_sha256=hashlib.sha256(raw).hexdigest(), geometry=geometry,
        queries=len(queries), cold_windows=8, query_windows=len(rows),
        candidates_mean=float(np.mean(counts)), candidates_p95=float(np.percentile(counts, 95)),
        candidates_max=max(counts), branches={name: sum(r["inplace_branch"] == name for r in rows)
            for name in ("empty", "inplace", "tie", "sparse_kth", "dense")},
        publication_ids=sum(r["publication_ids"] for r in rows),
        repeated_publication_ids=sum(r["repeated_publication_ids"] for r in rows),
        register_selection_positions=sum(r["register_selection_positions"] for r in rows),
        sparse_pack_positions=sum(r["sparse_pack_positions"] for r in rows),
        sparse_selection_positions=sum(r["sparse_selection_positions"] for r in rows),
        probe_samples=descriptions,
        probe_queries=len(queries), probe_bytes=(output / "probe_inputs.npz").stat().st_size,
        correctness="dense/sparse/inplace exact ordered prefix scores and IDs",
        timing="NOT_SIMULATED",
        limits=["Source-level logical work; items are not cycles or bandwidth",
                "Frozen eligibility; repeated publications are not repeated DDR reads",
                "No CPU wall time reported as NPU time; no URMA queue/completion model",
                "Probe inputs retain all real eight-window query sequences, not URMA service/completion timing"],
    )
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    archive.close()
    return summary


def profile_attribution(path):
    data = json.loads(path.read_text())
    result = {}
    for variant in VARIANTS:
        reports = [r for r in data["cases"][variant]["system_profiles"]["passes"] if r["metric"] == "none"]
        if len(reports) != 1 or reports[0].get("graph_replays") != 1:
            raise ValueError(f"{variant}: expected one independent single-graph profile")
        report = reports[0]
        timeline = summarize_timeline(Path(report["directory"]), report["artifacts"])
        if timeline["status"] != "available":
            raise ValueError(f"{variant}: {timeline}")
        stages = {}
        for stage in ("init", "li", "streaming"):
            item = timeline.get(stage)
            if not item or item["count"] != 1:
                raise ValueError(f"{variant}: ambiguous {stage} kernel count")
            stages[stage + "_us"] = item["sum_us"]
        kernels = [r for r in timeline["kernels"] if "v9_" in r["name"]]
        stages["kernel_span_us"] = (max(r["start_us"] + r["duration_us"] for r in kernels)
                                    - min(r["start_us"] for r in kernels))
        result[variant] = dict(stages=stages, timeline=timeline)
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, analysis_commit=source_commit(), groups={}, profiles={})
    print("AUDIT_BEGIN CPU_only; saved_headline_and_profiler; no_NPU no_build", flush=True)
    for group in args.groups:
        snapshot = args.inputs / f"g{group}-input"
        data = load_report(args.results / f"g{group}.json", require_incremental_inplace=True,
                           expected_native=args.expected_native, snapshot=snapshot)
        cases = data["cases"]
        reference, reverse_drift = control(cases, "reverse")
        original, original_drift = control(cases, "forward")
        drift = dict(forward=original_drift, reverse=reverse_drift)
        print(f"CONTROL G{group} forward_drift_mean/p50={drift['forward']['mean_us']:+.2f}/"
              f"{drift['forward']['p50_us']:+.2f}% reverse_drift_mean/p50={drift['reverse']['mean_us']:+.2f}/"
              f"{drift['reverse']['p50_us']:+.2f}%", flush=True)
        headline = {}
        for name in ("forward", *VARIANTS):
            case = cases[name]
            timing, transfers = case["timing"], traffic(case)
            headline[name] = dict(timing=timing, traffic=transfers, useful_prefetch=useful_prefetch(case))
            print(f"HEADLINE G{group} {name} mean/p50={timing['mean_us']:.3f}/{timing['p50_us']:.3f}us "
                  f"vs_best_reverse_delta={timing['mean_us']-reference['mean_us']:+.3f}/"
                  f"{timing['p50_us']-reference['p50_us']:+.3f}us "
                  f"vs_original_gain={100*(1-timing['mean_us']/original['mean_us']):+.2f}/"
                  f"{100*(1-timing['p50_us']/original['p50_us']):+.2f}% "
                  f"useful_pre={useful_prefetch(case):.1f} waste/repair/total_MiB={transfers['waste']/2**20:.4f}/"
                  f"{transfers['repair']/2**20:.4f}/{transfers['total']/2**20:.4f}", flush=True)
        work = audit_snapshot(snapshot, args.output / f"g{group}")
        result["groups"][str(group)] = dict(cpu=work, headline=headline, reverse_reference=reference,
                                          original_reference=original, drift=drift)
        branches = "/".join(str(work["branches"][k]) for k in ("empty", "inplace", "tie", "sparse_kth", "dense"))
        print(f"CPU_WORK G{group} PASS windows={work['query_windows']} "
              f"admitted_mean/p95/max={work['candidates_mean']:.1f}/{work['candidates_p95']:.1f}/{work['candidates_max']} "
              f"empty/inplace/tie/sparse/dense={branches} "
              f"published/repeated_IDs={work['publication_ids']}/{work['repeated_publication_ids']} "
              f"probe_queries={work['probe_queries']} probe_MiB={work['probe_bytes']/2**20:.2f} "
              "timing=NOT_SIMULATED", flush=True)
    for group in args.profile_groups:
        profile = profile_attribution(args.results / f"g{group}-diagnostic.json")
        result["profiles"][str(group)] = profile
        baseline = profile["reverse"]["stages"]
        for variant, report in profile.items():
            t = report["stages"]
            print(f"PROFILE G{group} {variant} init/LI/resolve={t['init_us']:.3f}/"
                  f"{t['li_us']:.3f}/{t['streaming_us']:.3f}us span={t['kernel_span_us']:.3f}us "
                  f"delta_LI/resolve={t['li_us']-baseline['li_us']:+.3f}/"
                  f"{t['streaming_us']-baseline['streaming_us']:+.3f}us single_profile_not_headline", flush=True)
    result["status"] = "pass"
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"AUDIT_PASS\nAUDIT_DIR={args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-native", required=True)
    parser.add_argument("--groups", type=int, nargs="+", default=[2, 7, 10, 17, 4])
    parser.add_argument("--profile-groups", type=int, nargs="+", default=[7, 2])
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, AssertionError) as error:
        parser.exit(1, f"AUDIT_FAILED {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
