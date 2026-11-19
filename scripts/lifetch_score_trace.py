#!/usr/bin/env python3
"""Inspect all score captures, export real replay inputs, and run a complete suite."""
from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import json
from pathlib import Path
import shutil
import sys
import subprocess

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests/operators/lifetch/support'))
from workloads import (ReplayPagePlan, ReplaySequence, checked_path, load_score_capture,
                       select_series)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def discover(root):
    paths = sorted(root.rglob('series.json'))
    paths = [p for p in paths if p.parent.name.startswith('dp') and p.parent.name[2:].isdigit()]
    if not paths:
        raise ValueError(f'No dp*/series.json found under {root}')
    cases, series_rows = {}, []
    for path in paths:
        series = json.loads(path.read_text())
        entries = series.get('rounds', [])
        manifests = list(path.parent.rglob('manifest.json'))
        series_rows.append(dict(path=str(path), status=series.get('status'),
            completed=series.get('completed_rounds'), indexed_rounds=len(entries),
            manifests=len(manifests), reason=series.get('last_reason'),
            empty=not entries and not manifests))
        indexed = set()
        for entry in entries:
            for group in entry['groups']:
                key = (path.parent.parent, entry['round_index'], group['group'])
                relative = group['manifest']
                indexed.add((path.parent / relative).resolve())
                try:
                    member = checked_path(path.parent, relative)
                    manifest = json.loads(member.read_text())
                    info = dict(path=str(member), real=manifest.get('real_inputs_saved'),
                        batch=manifest.get('geometry', {}).get('batch'), error=None)
                except (OSError, ValueError) as error:
                    info = dict(path=str(path.parent / relative), real=None, batch=None, error=str(error))
                cases.setdefault(key, []).append(info)
        series_rows[-1]['unindexed'] = [str(p) for p in manifests if p.resolve() not in indexed]
    return cases, series_rows


def inventory(root):
    cases, rows = discover(root)
    for row in rows:
        print(f"SERIES {row['path']}: status={row['status']} completed={row['completed']} "
              f"indexed_rounds={row['indexed_rounds']} manifests={row['manifests']} "
              f"unindexed={len(row['unindexed'])} reason={row['reason']}")
    if not cases:
        print('EMPTY: no indexed captures; collection settings are not captured data.' if all(r['empty'] for r in rows)
              else 'BLOCKED: files exist but no indexed captures; inspect series and manifest consistency.')
        return
    print('Root | Round | Group | Records | Real Q/Index | Score-only | Unknown/broken')
    for (source, number, group), members in sorted(cases.items()):
        print(f"{source} | {number} | {group} | {len(members)} | "
              f"{sum(m['real'] is True for m in members)} | "
              f"{sum(m['real'] is False for m in members)} | "
              f"{sum(m['real'] is None or m['error'] is not None for m in members)}")


def request_counts(capture):
    a, geometry = capture.arrays, capture.manifest['geometry']
    rows = []
    resident = geometry['resident']
    for row in range(geometry['batch']):
        selected = np.unique(a['post.topk'][row])
        logical = a['pre.logical_ids'][row, selected]
        hot = selected < resident
        hot[hot] = a['pre.hot_valid'][row, selected[hot]] != 0
        begin, end = map(int, a['pre.tail_bounds'][row])
        if not 0 <= begin <= end <= geometry['history']:
            raise ValueError('Invalid ABC tail bounds')
        abc = (logical >= begin) & (logical < end)
        cold = ~(hot | abc)
        missing_ddr = cold & (a['pre.address_valid'][row, selected] == 0)
        candidate_hot = hot & np.isin(selected, a['pre.lru_slots'][row])
        rows.append(dict(dp=capture.manifest['metadata']['dp_rank'], request=row,
                         demand=len(selected), hot=int(hot.sum()), abc_only=int((abc & ~hot).sum()),
                         cold=int(cold.sum()), candidate_hot=int(candidate_hot.sum()),
                         cold_without_ddr=int(missing_ddr.sum())))
    return rows


def display(rows):
    print('Captured Q4 union; cold = demand absent from valid hot AND ABC. Counts are not Get counts.')
    print('DP | Request | Demand | Hot | ABC-only | Cold | Cold miss % | Candidate hot | Cold w/o DDR')
    for row in rows:
        pct = 100 * row['cold'] / row['demand'] if row['demand'] else 0
        print(f"{row['dp']:2} | {row['request']:7} | {row['demand']:6} | {row['hot']:4} | "
              f"{row['abc_only']:8} | {row['cold']:4} | {pct:11.2f} | {row['candidate_hot']:13} | "
              f"{row['cold_without_ddr']:12}")
    total = {key: sum(row[key] for row in rows) for key in
             ('demand', 'hot', 'abc_only', 'cold', 'candidate_hot', 'cold_without_ddr')}
    total['cold_miss_percent'] = 100 * total['cold'] / total['demand'] if total['demand'] else None
    print(f"TOTAL BS={len(rows)} demand={total['demand']} cold={total['cold']} "
          f"cold_miss={total['cold_miss_percent']:.2f}% candidate_hot={total['candidate_hot']}")
    return total


def run(args):
    root = args.root.resolve()
    if args.group is None or args.round is None:
        if args.group is not None or args.round is not None or args.output is not None:
            raise ValueError('--round and --group must be supplied together')
        inventory(root)
        return
    for series_path in sorted(root.glob('dp[0-9]*/series.json')):
        series = json.loads(series_path.read_text())
        if series.get('status') != 'complete':
            raise ValueError(f"Series not complete: {series_path}; status={series.get('status')}, "
                             f"completed={series.get('completed_rounds')}, indexed_rounds={len(series.get('rounds', []))}; "
                             f"reason={series.get('last_reason')}. Capture configuration is not recorded data.")
    references = select_series(root, args.group, (args.round, args.round))
    if args.dp_ranks is not None:
        available = {ref.dp_rank for ref in references}
        if len(set(args.dp_ranks)) != len(args.dp_ranks) or not set(args.dp_ranks) <= available:
            raise ValueError('--dp-ranks contains duplicates or unavailable ranks')
        references = [ref for ref in references if ref.dp_rank in args.dp_ranks]
    output = args.output.resolve() if args.output else None
    if output and output.exists():
        raise ValueError(f'Output already exists: {output}; use a new directory')
    if output:
        output.mkdir(parents=True)
        write_json(output / 'conversion.json', dict(status='RUNNING'))
    rows, sources, widths = [], [], set()
    try:
        for ref in references:
            capture = load_score_capture(ref.manifest_path, require_real=output is not None)
            rows.extend(request_counts(capture))
            sources.append(str(ref.manifest_path))
            widths.add(capture.manifest['geometry']['token_bytes'])
            if not output:
                continue
            manifest = copy.deepcopy(capture.manifest)
            width = manifest['geometry']['token_bytes']
            if width not in (656, 2624):
                raise ValueError(f'Unsupported source token width: {width}')
            if args.layer_index is not None:
                if not 0 <= args.layer_index < width // 656:
                    raise ValueError('--layer-index outside captured layer group')
                manifest['geometry']['token_bytes'] = 656
            target = output / f'dp{ref.dp_rank}' / 'round0001' / f'group{args.group:03d}'
            target.mkdir(parents=True)
            archive = checked_path(capture.path.parent, manifest['archive']['file'])
            shutil.copyfile(archive, target / 'scores.npz')
            manifest['archive']['file'] = 'scores.npz'
            # Arrays, scores, owners, slots and their checksum are unchanged.
            # Only synthetic KV width is projected; captured transfer records
            # remain source evidence, not projected replay byte measurements.
            manifest['replay_export'] = dict(source=str(capture.path), source_round=args.round,
                source_token_bytes=width, layer_index=args.layer_index,
                kv='synthetic C8; real Q/Index and captured residency',
                captured_transfer_records='unchanged; source group units')
            write_json(target / 'manifest.json', manifest)
            series = json.loads((root / f'dp{ref.dp_rank}' / 'series.json').read_text())
            entry = copy.deepcopy(series['rounds'][args.round - 1])
            entry['round_index'] = 1
            member = next(copy.deepcopy(g) for g in entry['groups'] if g['group'] == args.group)
            member['manifest'] = f'round0001/group{args.group:03d}/manifest.json'
            entry['groups'] = [member]
            series.update(rounds=[entry], completed_rounds=1, expected_rounds=1,
                          dp_ranks=[r.dp_rank for r in references])
            series['replay_export'] = dict(source=str(root), source_round=args.round,
                                          independent_snapshot=True)
            write_json(target.parents[1] / 'series.json', series)
        totals = display(rows)
        result = dict(status='ANALYZED', source=str(root), source_round=args.round,
                      group=args.group, source_token_bytes=sorted(widths),
                      layer_index=args.layer_index, requests=rows, totals=totals, sources=sources)
        if output:
            sequence = ReplaySequence(output, args.group, (1, 1))
            batch = next(iter(sequence))
            # Exercise the same assembly and page-alias plan as device replay.
            pages = ReplayPagePlan(sequence)
            if totals['cold_without_ddr']:
                raise ValueError('Selected cold tokens lack DDR addresses')
            result.update(status='COMPLETE', batch=batch.geometry['batch'],
                          replay_token_bytes=batch.geometry['token_bytes'], pages=pages.page_count,
                          validation='CPU integrity, captured TopK, replay assembly and page plan PASS')
            write_json(output / 'conversion.json', result)
            print(f"READY --input real --trace-root {output} --group {args.group} --round-range 1:1")
            print('Independent captured round; device correctness and performance still require NPU replay.')
        return result
    except Exception as error:
        if output:
            write_json(output / 'conversion.json', dict(status='FAILED', error=str(error), sources=sources))
        raise


def export_all(args):
    root, output = args.root.resolve(), args.output.resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError('Use an output directory outside the scanned source tree')
    cases, series_rows = discover(root)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', source=str(root), series=series_rows, cases=[])
    write_json(output / 'suite.json', report)
    for i, ((source, number, group), members) in enumerate(sorted(cases.items())):
        name = f'case{i:05d}-r{number}-g{group}'
        row = dict(name=name, source=str(source), source_round=number, group=group, members=members)
        executable = all(m['real'] is True and not m['error'] for m in members)
        selected = argparse.Namespace(root=source, round=number, group=group, dp_ranks=None,
            layer_index=0, output=output / name if executable else None)
        try:
            with (output / f'{name}.log').open('w') as log, contextlib.redirect_stdout(log):
                result = run(selected)
            row.update(status='READY' if executable else 'SCORE_ONLY', analysis=result)
            if executable:
                row['trace_root'] = name
        except Exception as error:
            row.update(status='FAILED', error=str(error))
        report['cases'].append(row)
        write_json(output / 'suite.json', report)
        print(f"[{i+1}/{len(cases)}] {name}: {row['status']} {row.get('error', '')}", flush=True)
    failed = any(r['status'] == 'FAILED' for r in report['cases']) or any(r['unindexed'] for r in series_rows)
    report['status'] = 'PARTIAL' if failed else 'COMPLETE' if cases else 'EMPTY'
    write_json(output / 'suite.json', report)
    counts = {s: sum(r['status'] == s for r in report['cases']) for s in ('READY', 'SCORE_ONLY', 'FAILED')}
    print(f"SUITE {report['status']}: {counts}; empty series={sum(r['empty'] for r in series_rows)}")
    print(f"Report: {output / 'suite.json'}")
    if not counts['READY']:
        print('No executable Q/Index inputs; no NPU tests can be scheduled from this dataset.')
    return 1 if failed else 2 if not counts['READY'] else 0


def run_suite(args):
    suite_root = args.run_suite.resolve()
    suite = json.loads((suite_root / 'suite.json').read_text())
    ready = [r for r in suite['cases'] if r['status'] == 'READY']
    if not ready:
        raise ValueError('No READY inputs in suite.json; cannot run an empty test suite')
    extra = args.test_args
    if extra and extra[0] == '--':
        extra = extra[1:]
    reserved = {'--input', '--trace-root', '--group', '--round-range', '--output'}
    if any(word.split('=')[0] in reserved for word in extra):
        raise ValueError('Input selection and output are controlled by the suite runner')
    if args.output is None:
        raise ValueError('--run-suite requires --output for test results')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    entry = Path(__file__).resolve().parents[1] / 'tests/operators/lifetch/test_lifetch.py'
    for index, case in enumerate(ready):
        directory = output / case['name']
        command = [sys.executable, str(entry), args.mode, '--input', 'real',
            '--trace-root', str(suite_root / case['trace_root']), '--group', str(case['group']),
            '--round-range', '1:1', '--output', str(directory), *extra]
        print(f"RUN {index+1}/{len(ready)} {case['name']}", flush=True)
        with (output / f"{case['name']}.log").open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        rows = []
        for summary in directory.glob('*/summary.json'):
            rows.extend(json.loads(summary.read_text()).get('cases', []))
        passed = completed.returncode == 0 and bool(rows) and all(r.get('accepted') is True for r in rows)
        results.append(dict(case=case['name'], source_round=case['source_round'], group=case['group'],
            status='PASS' if passed else 'FAIL', exit_code=completed.returncode, command=command, results=rows))
        write_json(output / 'suite_results.json', dict(cases=results))
        flat = [dict(dataset=r['case'], source_round=r['source_round'], group=r['group'],
                     suite_status=r['status'], **sample) for r in results for sample in (r['results'] or [{}])]
        with (output / 'results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in flat for k in row)))
            writer.writeheader()
            writer.writerows(flat)
        print(f"{case['name']}: {results[-1]['status']} exit={completed.returncode}", flush=True)
    print(f"PASS {sum(r['status'] == 'PASS' for r in results)}/{len(results)}; {output / 'results.csv'}")
    return int(any(r['status'] != 'PASS' for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='Trace directory or data parent; scan all records by default')
    parser.add_argument('--round', type=int, help='Optional single capture round')
    parser.add_argument('--group', type=int, help='Optional single group')
    parser.add_argument('--dp-ranks', type=int, nargs='+')
    parser.add_argument('--output', type=Path, help='New export or test-results directory')
    parser.add_argument('--layer-index', type=int, help='Single-case projection; bulk exports use layer 0 (656B)')
    parser.add_argument('--run-suite', type=Path, help='Run every READY input in an exported suite')
    parser.add_argument('--mode', choices=('bench', 'diagnose'), default='bench')
    parser.add_argument('test_args', nargs=argparse.REMAINDER, help='After --: normal bench/diagnose options')
    args = parser.parse_args()
    if args.run_suite:
        return run_suite(args)
    if args.root is None or args.test_args:
        parser.error('--root is required; trailing test arguments require --run-suite')
    if args.round is None and args.group is None and args.output:
        if args.layer_index is not None or args.dp_ranks is not None:
            parser.error('Bulk export selects all DP records and uses single-layer 656B; omit selection flags')
        return export_all(args)
    try:
        run(args)
    except ValueError as error:
        parser.exit(1, f'ERROR: {error}\n')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        raise SystemExit(f'ERROR: {error}')
