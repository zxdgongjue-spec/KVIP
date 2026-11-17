#!/usr/bin/env python3
"""Execute actual host scheduling for B1/Q4/H128K at the requested A5 budget.

This records dispatch, exact task coverage and payload counts. It never
estimates microseconds, A5 ISA, bandwidth, or device concurrency.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import runpy
import subprocess
import tempfile

from simulate_schedule import FILES, SOURCE, execute, snapshot
from verify_cpu import compile_host

FIELDS = ('dispatch_count', 'prepared_q_weight_qscale_bytes_each_direction',
          'score_critical_128key_iterations', 'prepare_bytes_before_hot_service',
          'direct_theta_padded_dma_bytes')
SHORTLIST = {
    2: [(4, 8), (8, 16), (16, 24), (26, 28), (32, 32)],
    8: [(4, 8), (8, 16), (16, 16), (26, 26), (32, 32)],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frontier(rows):
    # Equal structural vectors are represented once. This is not a speed rank.
    representatives = {}
    for row in rows:
        representatives.setdefault(tuple(row[f] for f in FIELDS), row)
    result = []
    for vector, row in sorted(representatives.items()):
        if any(all(x[f] <= row[f] for f in FIELDS) for x in result):
            continue
        result = [x for x in result if not all(row[f] <= x[f] for f in FIELDS)]
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aics', type=int, default=32)
    parser.add_argument('--aivs', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capacity = min(args.aics, args.aivs // 2)
    if not (1 <= args.aics <= 32 and 16 <= args.aivs <= 64):
        parser.error('AIC 1..32 and AIV 16..64 are required')
    output = args.output.resolve()
    if output.exists():
        parser.error('choose a fresh output directory')
    output.mkdir(parents=True)
    recorded = list(FILES) + [
        'tests/operators/lifetch/cold_parallel/simulate_schedule.py',
        'tests/operators/lifetch/cold_parallel/verify_cpu.py',
        'tests/operators/lifetch/cold_parallel/explore_a5_budget.py',
    ]
    source_hashes = {p: digest(SOURCE / p) for p in recorded}
    rows, controls, selected, fetch_rows = [], [], [], []
    split = runpy.run_path(str(SOURCE / FILES[-1]))['split_score_tasks']
    with tempfile.TemporaryDirectory(prefix='kvip-a5-budget-') as folder:
        folder = Path(folder)
        before = folder / 'reference'
        reference_commit = snapshot('04ddfaa', before)
        old_build = folder / 'old-build'
        new_build = folder / 'new-build'
        old_build.mkdir()
        new_build.mkdir()
        old_lib, old_tags = compile_host(before, old_build)
        lib, tags = compile_host(SOURCE, new_build)

        def count(w, post, prepare, service, ph=0, pc=0, **kw):
            row = execute(SOURCE, lib, tags, w, post, prepare, service,
                          ph, pc, args.aics, args.aivs, splitter=split,
                          compact=True, **kw)
            assert row['logical_qk_macs'] == 2147483648
            assert row['cold_predicate_comparisons'] == 458752
            return row

        for windows in (2, 8):
            old = execute(before, old_lib, old_tags, windows, 0, 0, 0,
                          0, 0, args.aics, args.aivs, compact=True)
            controls.append(dict(label='04ddfaa_separate_wave', **old))
            for post, prepare, service in ((0, 0, 0), (2, 0, 0),
                                           (2, 2, 0), (2, 2, 1), (2, 1, 1)):
                controls.append(dict(label='current', **count(windows, post, prepare, service)))
            for ph, pc, prepare, service in itertools.product(
                    range(1, capacity + 1), range(1, capacity + 1), range(3), range(2)):
                rows.append(count(windows, 2, prepare, service, ph, pc))
            for ph, pc in SHORTLIST[windows]:
                if max(ph, pc) > capacity:
                    continue
                row = count(windows, 2, 2, 1, ph, pc)
                selected.append(row)
                for workers in (1, 2, 4):
                    if args.aivs <= 4 * workers:
                        continue
                    fetch_rows.append(count(windows, 2, 2, 1, ph, pc, workers=workers))

    for p, sha in source_hashes.items():
        if digest(SOURCE / p) != sha:
            raise RuntimeError('source changed during scheduling experiment: ' + p)
    for name, data in (('primary_grid.csv', rows), ('controls.csv', controls),
                       ('shortlist.csv', selected), ('fetch_width.csv', fetch_rows)):
        if not data:
            continue
        with (output / name).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            for row in data:
                writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                                 for k, v in row.items()})
    pareto = []
    for windows in (2, 8):
        pareto.extend(frontier([r for r in rows if r['windows'] == windows
                               and r['prepare'] == 2 and r['service'] == 1]))
    result = dict(
        status='SOURCE_COUNTS_PASSED', aics=args.aics, aivs=args.aivs,
        reference_commit=reference_commit,
        candidate_base_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
        candidate_state='working tree bound by hashes', source_sha256=source_hashes,
        grid_cases=len(rows), control_cases=len(controls), fetch_cases=len(fetch_rows),
        structural_frontier_fields=FIELDS, structural_frontier=pareto, shortlist=selected,
        notes=[
            'Critical iterations sum the longest task per score wave; they are not device cycles.',
            'Independent UB permits local work, but does not establish cross-kernel overlap.',
            'Fetch worker sweep changes tail capacity/dispatch only; no URMA latency is modeled.',
            'Resident and Final TopK remain one H partition per real query.',
            'Legacy P2 device timing is not executed by this host scheduling experiment.',
        ],
        excludes=['A5 timing', 'CANN Model cycles', 'ISA counts', 'full GM transactions',
                  'first Fetch', 'uncovered join', 'ConsumerReady', 'best device configuration'],
        artifacts={p.name: dict(path=str(p.relative_to(SOURCE)) if p.is_relative_to(SOURCE) else str(p), sha256=digest(p))
                   for p in output.glob('*.csv')},
    )
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in
                      ('status', 'aics', 'aivs', 'grid_cases', 'control_cases', 'fetch_cases')}, indent=2))


if __name__ == '__main__':
    main()
