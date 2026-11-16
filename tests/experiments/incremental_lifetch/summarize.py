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


def detailed_trace(report, path, label, required=False):
    trace_path = path.with_name('full_attribution.json')
    trace = read(trace_path) if trace_path.exists() else report.get('full_trace', {})
    samples = trace.get('samples', [])
    if not samples or not any('final_topk' in s for s in samples):
        if required:
            raise ValueError(f'{path}: detailed TopK records missing')
        return [f'{label}: detailed TopK records absent (old run).']
    mhz = report.get('counter_clock_mhz')
    if not mhz or mhz <= 0:
        raise ValueError(f'{path}: missing device counter clock')
    final, windows = [], {}
    for sample in samples:
        records = sample.get('final_topk', [])
        if len(records) != report['batch'] * 4 or any(not r['present'] or r['end'] < r['begin'] or r['compute_cycles'] < 0 for r in records):
            raise ValueError(f'{path}: incomplete/invalid final TopK trace')
        final.extend(records)
        for w in sample['incremental_windows']:
            windows.setdefault(w['epoch'], []).append(w)
    mean = lambda rows, name: statistics.mean(r[name] for r in rows)
    lines = [f'{label} final TopK: compute includes input/compact/merge/ID recovery; excludes explicit wait and final classification.',
             'Per-query durations; block totals below sum sequential queries, not parallel cores.']
    lines += table(['Path', 'Final Q calls/replay', 'TopK calls/replay', 'Wait mean/max us per Q', 'Compute mean/max us per Q'], [[label, report['batch']*4, number(sum(r['topk_calls'] for r in final)/len(samples)), f"{mean(final,'wait_cycles')/mhz:.2f}/{max(r['wait_cycles'] for r in final)/mhz:.2f}", f"{mean(final,'compute_cycles')/mhz:.2f}/{max(r['compute_cycles'] for r in final)/mhz:.2f}"]])
    rows = []
    for q in sorted({r['query'] for r in final}):
        rr = [r for r in final if r['query'] == q]
        rows.append([q//4, q%4+1, rr[0]['block'], number(mean(rr,'wait_cycles')/mhz), number(mean(rr,'compute_cycles')/mhz), number(mean(rr,'topk_calls'))])
    lines += table(['Request','Q','Block','Wait us','Compute us','TopK calls'], rows)
    rows=[]
    for block in sorted({r['block'] for r in final}):
        rr=[r for r in final if r['block']==block]
        rows.append([block, number(sum(r['wait_cycles'] for r in rr)/len(samples)/mhz), number(sum(r['compute_cycles'] for r in rr)/len(samples)/mhz), number(sum(r['topk_calls'] for r in rr)/len(samples))])
    lines += table(['Final block','Wait us/replay','Compute us/replay','TopK calls/replay'], rows)
    lines += ['Window costs: mean/max over executed lane events; counts averaged per replay. Incremental and service are separate intervals.']
    rows=[]
    for epoch, rr in sorted(windows.items()):
        active=[r for r in rr if r['service_calls']]
        def duration(name):
            return '--' if not active else f'{mean(active,name)/mhz:.2f}/{max(r[name] for r in active)/mhz:.2f}'
        rows.append([epoch, number(len(rr)/len(samples)), number(sum(r['skipped'] for r in rr)/len(samples)), number(len(active)/len(samples)), number(sum(r['topk_calls'] for r in rr)/len(samples)), number(sum(r.get('scanned',0) for r in rr)/len(samples)), number(sum(r.get('candidates',0) for r in rr)/len(samples)), duration('cycles'), duration('service_wait_cycles'), duration('service_cycles'), duration('scan_cycles'), duration('candidate_cycles'), duration('post_cycles')])
    lines += table(['Epoch','Attempts','Skipped','Service calls','Incremental TopK calls','Scanned scores','Candidates','Incremental mean/max us','Service wait mean/max us','Service mean/max us','Scan mean/max us','Candidate mean/max us','Post mean/max us'],rows)
    return lines


def summarize(run, required=False):
    cases = {v: load_cases(run / v) for v in ('baseline', 'incremental')}
    lines = [f'Run: {run}', 'A=baseline; B=incremental. Time: us; positive reduction means faster.']
    if (run / 'experiment.txt').exists() and (run / 'experiment.txt').read_text().strip() == 'ranked-floor':
        lines[1] = 'A=ranked prefetch without floor; B=same selection with quota/2 floor before dedup/mapping.'
        lines += ['With 32/Q cap, select 16..32 eligible cold candidates; fewer than 16 => all. Actual Fetch can be lower after dedup/mapping.']
    if (run / 'experiment.txt').exists() and (run / 'experiment.txt').read_text().strip() == 'ranked-prefetch':
        lines[1] = 'A=accepted ABC baseline; B=A + per-window score-rank threshold and per-Q highest-score quota.'
        lines += ['Four-window ranks=1024,1365,1707,2048; 128/request/window gives 32/Q before cross-Q dedup.',
                  'All windows including window 0 use ranked cold candidates; no redistribution after dedup/mapping filters.']
    if (run / 'experiment.txt').exists() and (run / 'experiment.txt').read_text().strip() == 'recent-index':
        manifest = json.loads((run / 'incremental-source/recent_experiment.json').read_text())
        lines[1] = 'A=current TopK-prefetch B; B=A + PR15 recent Index seed. Time: us.'
        lines += [f"Configured ABC tail ({manifest['abc_tail_tokens']} tokens) range=[{manifest['recent_begin']},{manifest['recent_end']}); per-Q causal clipping.",
                  'Identical KV residency/input; Index-only early scoring, exact seed merge, no duplicate admission on later scan.',
                  'Incremental windows/final TopK, original correctness checks and level3 timing retained.']
    if (run / 'experiment.txt').exists() and (run / 'experiment.txt').read_text().strip() == 'window-topk-only':
        lines[1] = 'A=original LIFetch; B=original LIFetch + private window TopK only. Time: us.'
        lines += ['Same seed initialization and profiling in A/B; intermediate results do not affect prefetch or final TopK.']
    if (run / 'experiment.txt').exists() and (run / 'experiment.txt').read_text().strip() == 'final-topk-only':
        lines[1] = 'A=window TopK + original final TopK; B=same windows + incremental final TopK. Time: us.'
        lines += ['Original prefetch decisions retained in both paths; final TopK compute/wait reported separately.']
    if (run / 'experiment.txt').exists() and (run / 'experiment.txt').read_text().strip() == 'topk-prefetch':
        lines[1] = 'A=bitmap prefetch; B=latest TopK union prefetch. Both use incremental window/final TopK.'
        lines += ['B middle scan = compact cold-ID load + cross-Q dedup; cold-ID export is included in Incremental. Hot window uses original bitmap.',
                  'B middle candidates are unique cold TopK tokens capped at 2048; original map filtering/budget/post remain. Compare reads and full graph.']
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
        for variant, label in (('baseline', 'A'), ('incremental', 'B')):
            if key in cases[variant]:
                report, path = cases[variant][key]
                lines += detailed_trace(report, path, label, required)
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
    parser.add_argument('run', nargs='?', type=Path, help='A/B result directory; default: latest window-topk-only run')
    parser.add_argument('--experiment', default='window-topk-only',
                        choices=('window-topk-only', 'final-topk-only', 'topk-prefetch', 'recent-index', 'ranked-prefetch', 'ranked-floor'),
                        help='Result family to search when run is omitted')
    parser.add_argument('--require-detailed', action='store_true', help='Fail if new A/B trace records are absent')
    args = parser.parse_args()
    run = args.run
    if run is None:
        root = Path(__file__).resolve().parents[3] / 'tests/results' / args.experiment
        runs = sorted(p for p in root.glob('*') if p.is_dir())
        if not runs:
            parser.error(f'No runs under {root}')
        run = runs[-1]
    run = run.resolve()
    if not run.is_dir():
        parser.error(f'Not a directory: {run}')
    text = summarize(run, args.require_detailed)
    output = run / 'key_results.log'
    output.write_text(text)
    print(text, end='')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
