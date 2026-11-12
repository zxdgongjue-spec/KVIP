#!/usr/bin/env python3
"""Offline read-volume and same-replay Fetch critical-path comparison."""
import argparse
import json
from pathlib import Path
import statistics

from summarize_core_timeline import decode


def read_range(samples, key):
    values = [s[key] for s in samples]
    return f'{min(values)}..{max(values)}'


def analyze(root, batch, misses):
    lines = ['OFFLINE: no device execution; counts cover the entire batch.',
             'Graph p50 is unprofiled. All per-core times are same-replay calibrated us.',
             'CQ counters include polling/processing; they are NOT pure network latency.',
             'No prefetch bandwidth inferred: these timelines start after LI, not at first prefetch post.']
    details, records = [], []
    for miss in misses:
        folder = root / f'current-b{batch}-m{miss}'
        reports = list(folder.glob('case*/report.json'))
        if len(reports) != 1:
            raise ValueError(f'Expected one report: {folder}')
        path = reports[0]
        report = json.loads(path.read_text())
        if report['status'] != 'pass':
            raise ValueError(f'Unchecked result: {path}')
        reads = report['read_samples']['prototype_with_update']
        if not reads:
            raise ValueError('Missing read samples')
        native = sum(report['native_misses_per_request'])
        for s in reads:
            if s['total'] != s['prefetch_hot'] + s['prefetch_cold'] + s['final']:
                raise ValueError('Read accounting mismatch')
        timing = report['timing']
        lines += [f'\nBS={batch} miss={miss}% serial_graph={timing["native_chain"]["p50_us"]:.3f} current_graph={timing["prototype_with_update"]["p50_us"]:.3f} us',
                  f'native_reads={native}; independent read samples (min..max, do not add range endpoints):',
                  ' | '.join(f'{k}={read_range(reads,k)}' for k in
                             ('prefetch_hot','prefetch_cold','unused_hot','unused_cold','final','total'))]
        main_reports = list((root / f'main-b{batch}-m{miss}').glob('case*/report.json'))
        if len(main_reports) == 1:
            main = json.loads(main_reports[0].read_text())
            lines.append(f'main_graph={main["timing"]["prototype_with_update"]["p50_us"]:.3f} us; main_total_reads=' +
                         read_range(main['read_samples']['prototype_with_update'], 'total'))
        samples = json.loads(path.with_name('full_attribution.json').read_text())['samples']
        if not samples:
            raise ValueError('Missing full pipeline profiling samples')
        for sample in samples:
            if 'error' in sample['accounting']:
                raise ValueError(f'Invalid full attribution: {path}')
            raw = sample['streaming_cycles'][0]['core_timeline_raw']
            if raw['batch'] != batch:
                raise ValueError('Trace batch differs')
            tl = decode(raw)
            cores = tl['cores']
            last = next(c for c in cores if c['block'] == tl['last_block'])
            fetch = [c for c in cores if c['role'].startswith('fetch ')]
            updates = [c for c in cores if c['role'].startswith('index_update ')]
            end_fetch = max(c['end_us'] for c in fetch)
            record = dict(miss=miss, round=sample['round'], timeline=tl,
                          profiled_stages=sample['accounting']['stages_us'],
                          prefetch=sample['prefetch'], final=sample['final'])
            records.append(record)
            lines += [f'ROUND {sample["round"]}: stages_us={json.dumps(record["profiled_stages"], sort_keys=True)}',
                      f'  reads prefetch={sample["prefetch"]} final={sample["final"]}; update_end={max(c["end_us"] for c in updates):.3f} fetch_end={end_fetch:.3f} kernel_last={last["block"]}({last["role"]}) end={last["end_us"]:.3f} tail_after_fetch={last["end_us"]-end_fetch:.3f}',
                      '  Fetch block | Q/lane | start..end | published_entries | Get intervals | post_us | CQ_us']
            for c in fetch:
                stages = c['stages']
                gets = [s for s in stages if s['name'].endswith('.KVAF_Get')]
                counters = c.get('KVAF_counters', {})
                post = sum(counters.get(k,0) for k in ('safe_post_cycles','conflict_post_cycles')) / tl['clock_mhz']
                cq = sum(counters.get(k,0) for k in ('safe_CQ_cycles','conflict_CQ_cycles')) / tl['clock_mhz']
                intervals = ','.join(f'{s["name"]}:{s["start_us"]:.2f}..{s["end_us"]:.2f}' for s in gets)
                lines.append(f'  {c["block"]} | {c["role"]} | {c["start_us"]:.2f}..{c["end_us"]:.2f} | {sum(c["published_misses"])} | {intervals or "empty"} | {post:.2f} | {cq:.2f}')
            lines.append('  Last core waits/finalization: stage start..end (duration)')
            for s in last['stages']:
                if s['name'] in ('prefetch_CQ_join','residency','status') or s['name'].startswith('wait.'):
                    lines.append(f'  {s["name"]} {s["start_us"]:.2f}..{s["end_us"]:.2f} ({s["duration_us"]:.2f})')
            details.append(f'BS={batch} miss={miss} ROUND={sample["round"]}')
            for c in cores:
                details.append(f'block={c["block"]} {c["role"]} {c["start_us"]:.3f}..{c["end_us"]:.3f}')
                for s in c['stages']:
                    details.append(f'  {s["name"]}: {s["start_us"]:.3f}..{s["end_us"]:.3f} duration={s["duration_us"]:.3f}')
    lines += ['\nMEDIANS across profiled rounds (not additive to unprofiled graph p50):',
              'miss | prefetch | final | fused_kernel_us | update_end | fetch_end | kernel_end']
    for miss in misses:
        group = [r for r in records if r['miss'] == miss]
        values = [[r['prefetch'], r['final'], r['profiled_stages']['resolve_update'],
                   max(c['end_us'] for c in r['timeline']['cores'] if c['role'].startswith('index_update ')),
                   max(c['end_us'] for c in r['timeline']['cores'] if c['role'].startswith('fetch ')),
                   r['timeline']['observed_span_us']] for r in group]
        lines.append(str(miss) + ' | ' + ' | '.join(f'{statistics.median(v):.3f}' for v in zip(*values)))
    return '\n'.join(lines)+'\n', '\n'.join(details)+'\n', records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--misses', type=int, nargs='+', default=[10,15])
    args = parser.parse_args()
    text, details, records = analyze(args.root, args.batch, args.misses)
    output = args.root / ('prefetch-analysis-b' + str(args.batch))
    output.mkdir(exist_ok=True)
    (output/'summary.log').write_text(text)
    (output/'details.log').write_text(details)
    (output/'analysis.json').write_text(json.dumps(records, indent=2))
    print(text, end='')
    print(f'Summary: {output / "summary.log"}\nAll core details: {output / "details.log"}')


if __name__ == '__main__':
    main()
