#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize independently profiled baseline-safe resolver worker runs."""
import json
import statistics
import sys
from pathlib import Path


def summarize(root):
    root = Path(root)
    lines = [
        'baseline-safe: H131072 R16384 Q4 heads32 miss10%, graph mode; checks after timing.',
        'Resolver device median is from 3 separately profiled full-graph replays; graph p50/p99 use 20 replays.',
        'Window count may be clipped for small BS; compare worker counts within the same BS.',
        'batch | workers/request | actual windows | resolver us | full graph p50 us | p99 us | final reads range | SFA max abs',
    ]
    for workers in (1, 2, 4):
        reports = list((root / f'w{workers}').rglob('report.json'))
        by_batch = {}
        for path in reports:
            report = json.loads(path.read_text())
            batch = int(report['resource_geometry']['logical_aic_tasks'] // report['resource_geometry']['cores_per_batch'])
            if batch in by_batch:
                raise ValueError(f'duplicate BS{batch} for workers={workers}')
            by_batch[batch] = (path, report)
        for batch in (1, 2, 4, 8, 16):
            if batch not in by_batch:
                raise ValueError(f'missing report BS{batch} workers={workers}')
            path, report = by_batch[batch]
            if report['status'] != 'pass' or report['final_workers'] != workers:
                raise ValueError(f'correctness/worker mismatch: {path}')
            trace = report['full_trace']['samples']
            if len(trace) < 3 or any('error' in row['accounting'] for row in trace):
                raise ValueError(f'missing valid profiler samples: {path}')
            resolver = statistics.median(x['accounting']['stages_us']['resolve'] for x in trace)
            timing = report['timing']['prototype_with_update']
            counts = [x['final_reads'] for x in report['split_samples']['prototype_with_update']]
            lines.append(f"{batch} | {workers} | {report['resource_geometry']['intermediate_fetches']} | "
                         f"{resolver:.3f} | {timing['p50_us']:.3f} | {timing['p99_us']:.3f} | "
                         f"{min(counts)}..{max(counts)} | {report['sfa']['max_abs']:.6g}")
    result = '\n'.join(lines) + '\n'
    (root / 'summary.log').write_text(result)
    return result


if __name__ == '__main__':
    root = Path(sys.argv[1])
    print(summarize(root), end='')
    print(f'Summary: {root / "summary.log"}')
