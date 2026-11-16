#!/usr/bin/env python3
"""Summarize saved incremental LIFetch A/B results without importing NPU libraries."""
import argparse
import json
import statistics
from pathlib import Path


LABEL = 'prototype_with_update'


def read(path):
    return json.loads(path.read_text())


def number(value):
    return '--' if value is None else f'{value:.2f}'


def table(headers, rows):
    rows = [[str(v) for v in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    return [' | '.join(v.ljust(w) for v, w in zip(row, widths))
            for row in [headers, ['-' * w for w in widths], *rows]]


def load_cases(root):
    result = {}
    for path in sorted(root.rglob('report.json')):
        report = read(path)
        if 'batch' not in report or 'timing' not in report:
            continue
        key = (int(report['batch']), report.get('miss_percent'))
        if key in result:
            raise ValueError(f'Multiple reports for BS/miss={key} under {root}; select one A/B run')
        result[key] = (report, path)
    return result


def stages(report, path):
    trace_path = path.with_name('full_attribution.json')
    trace = read(trace_path) if trace_path.exists() else report.get('full_trace', {})
    samples = trace.get('samples', [])
    valid = [s for s in samples if 'error' not in s.get('accounting', {})
             and s.get('accounting', {}).get('stages_us')]
    values = {}
    for s in valid:
        account = s['accounting']
        row = dict(account['stages_us'])
        row.update(gap=account.get('inter_kernel_gap_us'), boundary=account.get('boundary_residual_us'))
        for name, value in row.items():
            if value is not None:
                values.setdefault(name, []).append(value)
    return {k: statistics.mean(v) for k, v in values.items()}, len(valid), len(samples)


def counts(samples, name):
    values = [s[name] for s in samples if name in s]
    if not values:
        return '--'
    lo, hi = min(values), max(values)
    return str(lo) if lo == hi else f'{lo}..{hi}'


def summarize(run):
    cases = {v: load_cases(run / v) for v in ('baseline', 'incremental')}
    lines = [f'Run: {run}', 'A=baseline; B=incremental. Time: us; positive reduction means faster.']
    keys = sorted(set(cases['baseline']) | set(cases['incremental']), key=lambda k: (k[0], str(k[1])))
    for key in keys:
        lines += ['', f'BS={key[0]} miss={key[1]}%']
        graph, io, profiles = [], [], {}
        means = {}
        for variant, short in (('baseline', 'A'), ('incremental', 'B')):
            if key not in cases[variant]:
                graph.append([short, 'NO_REPORT', '--', '--', '--', '--'])
                continue
            report, path = cases[variant][key]
            timing = report['timing']
            fused, native = timing.get(LABEL, {}), timing.get('native_chain', {})
            status = report.get('correctness', report.get('status', 'UNKNOWN')).upper()
            graph.append([short, status, number(native.get('mean_us')), number(fused.get('mean_us')),
                          number(fused.get('p50_us')), number(fused.get('p99_us'))])
            if status == 'PASS' and fused.get('mean_us'):
                means[short] = fused['mean_us']
            profiles[short] = stages(report, path)
            samples = report.get('read_samples', {}).get(LABEL, [])
            split = report.get('split_samples', {}).get(LABEL, [])
            native_reads = report.get('native_misses_per_request')
            io.append([short, '--' if native_reads is None else sum(native_reads),
                       *[counts(samples, n) for n in ('prefetch_hot', 'prefetch_cold', 'final', 'unused', 'total')],
                       counts(split, 'conflicting_reads')])
        lines += table(['Path', 'Check', 'Serial mean', 'Fused mean', 'Fused p50', 'Fused p99'], graph)
        if len(means) == 2:
            lines += [f"B vs A: {100*(1-means['B']/means['A']):+.2f}% reduction; delta={means['B']-means['A']:+.2f} us"]
        if profiles:
            lines += ['Profiled stages: mean of all valid separate replays, including round 0.']
            names = list(dict.fromkeys(n for p, _, _ in profiles.values() for n in p))
            rows = []
            for name in names:
                a = profiles.get('A', ({}, 0, 0))[0].get(name)
                b = profiles.get('B', ({}, 0, 0))[0].get(name)
                rows.append([name, number(a), number(b), number(b-a if a is not None and b is not None else None)])
            if rows:
                lines += table(['Stage', 'A mean', 'B mean', 'B-A'], rows)
            lines += ['Trace samples: ' + '; '.join(f'{k}={v[1]}/{v[2]} valid' for k, v in profiles.items())]
        if key in cases['incremental']:
            report, path = cases['incremental'][key]
            trace_path = path.with_name('full_attribution.json')
            trace = read(trace_path) if trace_path.exists() else report.get('full_trace', {})
            windows = [r['incremental_windows'] for r in trace.get('samples', []) if 'incremental_windows' in r]
            if windows:
                mhz = report.get('counter_clock_mhz')
                rows = []
                for epoch in sorted({w['epoch'] for sample in windows for w in sample}):
                    groups = [[w for w in sample if w['epoch'] == epoch] for sample in windows]
                    rows.append([epoch,
                                 number(statistics.mean(sum(not w['skipped'] for w in g) for g in groups)),
                                 number(statistics.mean(sum(w['skipped'] for w in g) for g in groups)),
                                 number(statistics.mean(sum(w['topk_calls'] for w in g) for g in groups)),
                                 number(statistics.mean(max((w['cycles'] for w in g), default=0) for g in groups)/(mhz or 1))])
                lines += ['B incremental windows: averages across profiled replays; calls sum all service lanes.']
                lines += table(['Epoch', 'Active lanes', 'Skipped lanes', 'TopK calls', 'Slowest lane us' if mhz else 'Slowest lane cycles'], rows)
        if io:
            lines += ['DDR entries: whole batch, per-replay min..max; unused = fetched but absent from final TopK.']
            lines += table(['Path', 'Native', 'Prefetch hot', 'Prefetch cold', 'Final', 'Unused', 'Total', 'Conflict'], io)
    for variant in ('baseline', 'incremental'):
        if not cases[variant]:
            lines += [f'{variant}: no completed report.']
        errors = sorted((run / variant).rglob('error.log'))
        if errors:
            error = errors[-1]
            tail = [s for s in error.read_text(errors='replace').splitlines() if s.strip()]
            lines += [f'{variant} failure: {tail[-1] if tail else "empty error log"}', f'  {error}']
    if not keys:
        lines += ['No completed A/B cases yet; original logs remain in the run directory.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', type=Path, help='A/B result directory; default: latest run')
    args = parser.parse_args()
    run = args.run
    if run is None:
        root = Path(__file__).resolve().parents[3] / 'tests/results/incremental-lifetch'
        runs = sorted(p for p in root.glob('*') if p.is_dir())
        if not runs:
            parser.error(f'No runs under {root}')
        run = runs[-1]
    run = run.resolve()
    if not run.is_dir():
        parser.error(f'Not a directory: {run}')
    text = summarize(run)
    output = run / 'key_results.log'
    output.write_text(text)
    print(text, end='')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
