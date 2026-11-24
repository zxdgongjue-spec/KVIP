#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare only the Hot Score launch on the same MR28 library."""

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile

from run_mr25_handoff import KVAF, ROOT, bench_arguments, collect, git, read, save


MODES = ('p2', 'uniform')
MR28 = 'f29180a01867758c353724437c668e2c302ad45f'
PAIRED_KEYS = ('fixture_sha256', 'library_sha256', 'soc', 'effective_aic', 'windows',
               'score_parts', 'score_blocks', 'service_lanes', 'terminal_tasks', 'local_counts')


def completed_reads(output, mode, batch):
    result = read(output / mode / 'miss10' / f'case-b{batch}' / 'fused/path_result.json')
    return {
        (row['source_round'], row['invocation'], row['request']):
            (row['prefetch_by_window'], row['final_get_by_query'])
        for row in result['pipeline_counts']
    }


def summarize(output, batch, completed):
    rows = [collect(output, mode, batch) for mode in MODES]
    base, uniform = rows
    paired = all(completed.get(mode) == 0 for mode in MODES) and all(row['accepted'] for row in rows)
    if paired:
        paired = (all(base[key] == uniform[key] for key in PAIRED_KEYS)
                  and base['local_counts'] is True
                  and completed_reads(output, 'p2', batch) == completed_reads(output, 'uniform', batch))
        expected_parts = (2, min(base['effective_aic'] // batch, 16384 // 256))
        for expected, row in zip(expected_parts, rows):
            paired &= row['hot_score_parts'] == expected and row['hot_score_blocks'] == batch * expected
    for row in rows:
        if row['accepted']:
            parts = row['hot_score_parts']
            row['hot_tokens_min'] = (16384 // 256 // parts) * 256
            row['hot_tokens_max'] = ((16384 // 256 + parts - 1) // parts) * 256
        row.update(paired=paired,
                   hot_over_p2=row['hot_us'] / base['hot_us'] if paired else None,
                   kv_ready_over_p2=row['kv_ready_us'] / base['kv_ready_us'] if paired else None)
    save(output / 'summary.json', rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
    if paired:
        print(f"BS{batch} Hot Score {base['hot_score_blocks']} -> {uniform['hot_score_blocks']} MIX blocks, "
              f"{base['hot_us']:.2f} -> {uniform['hot_us']:.2f} us "
              f"({uniform['hot_over_p2']:.4f}x)", flush=True)
        print(f"BS{batch} KV ready {base['kv_ready_us']:.2f} -> {uniform['kv_ready_us']:.2f} us "
              f"({uniform['kv_ready_over_p2']:.4f}x)", flush=True)
    else:
        print('Pair rejected: check completion, route, shared library, fixture and per-call Get counts.', flush=True)
    print('Summary: ' + str(output / 'summary.csv'), flush=True)
    return paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', action='append', type=int, choices=(1, 2, 4, 8, 16), required=True,
                        help='batch size; repeat to run independent cases with one shared build')
    parser.add_argument('--comparison-dry-run', action='store_true')
    parser.add_argument('--reverse-order', action='store_true')
    args = parser.parse_args()
    if len(set(args.batch)) != len(args.batch):
        parser.error('Each batch can appear only once')
    order = MODES[::-1] if args.reverse_order else MODES
    base = Path(os.environ.get('KVIP_AB_OUTPUT', 'results/mr25_hot_score')).resolve()
    base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='mr25_hot_score.', dir=base))
    kvaf = Path(os.environ.get('KVIP_KVAF_ROOT', ROOT / 'third_party/KVAF')).resolve()
    cases = {batch: dict(arguments=bench_arguments(kvaf, (batch,)), completed={}) for batch in args.batch}
    manifest = dict(status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), mr28_base=MR28,
                    expected_kvaf=KVAF, cases=cases, variants=order, local_counts=True,
                    shared_build=str(output / 'build'))
    save(output / 'manifest.json', manifest)
    print('MR25 Hot Score comparison: ' + str(output), flush=True)
    if args.comparison_dry_run:
        print(f'Planned batches {args.batch}, 2 independent workers per batch, local_counts ON.', flush=True)
        return 0

    from execution import launch
    from workloads import configuration

    try:
        if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
            raise ValueError(f'Use the built KVAF checkout at {KVAF}')
        manifest['status'] = 'RUNNING'
        save(output / 'manifest.json', manifest)
        passed = True
        for batch, case in cases.items():
            directory = output / f'bs{batch}'
            directory.mkdir()
            completed = case['completed']
            for mode in order:
                config = configuration(case['arguments'])
                config.mr25_handoff = mode
                config.mr25_hot_partition = mode == 'uniform'
                config.handoff_build_dir = output / 'build'
                print(f'Running BS{batch} {mode}, local_counts ON', flush=True)
                completed[mode] = launch(config, directory / mode)
                save(output / 'manifest.json', manifest)
                if completed[mode]:
                    break
            passed = summarize(directory, batch, completed) and passed
        manifest['status'] = 'PASS' if passed else 'FAILED'
        return int(not passed)
    except BaseException:
        manifest['status'] = 'FAILED'
        raise
    finally:
        save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    sys.exit(main())
