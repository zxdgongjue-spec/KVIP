"""Compare per-batch metadata preparation with the fixed original Update path."""
import argparse
import json
import statistics
from pathlib import Path

from summarize_core_timeline import decode


def summarize(root, description=None):
    lines = [description or 'A: original Update; B: per-batch metadata preparation. Parallel Fetch finalization disabled.',
             'Graph times are unprofiled; core stages are separate profiled replays in calibrated us.',
             'version | BS | full_graph_p50 | full_graph_p99 | fused_profiled_median']
    results = {}
    for version in ('A', 'B'):
        reports = sorted((root / version).rglob('report.json'))
        if not reports:
            raise ValueError(f'{version}: no checked reports')
        for path in reports:
            report = json.loads(path.read_text())
            samples = json.loads(path.with_name('full_attribution.json').read_text())['samples']
            if not samples or any('error' in s['accounting'] for s in samples):
                raise ValueError(f'invalid profiler results: {path}')
            batch = samples[0]['streaming_cycles'][0]['core_timeline_raw']['batch']
            timing = report['timing']['prototype_with_update']
            fused = statistics.median(s['accounting']['stages_us']['resolve_update'] for s in samples)
            results[batch, version] = timing['p50_us']
            lines.append(f"{version} | {batch} | {timing['p50_us']:.3f} | {timing['p99_us']:.3f} | {fused:.3f}")
            lines.append('Profiled stage | median_us (separate replay, not additive to graph p50)')
            names = sorted(set.intersection(*(set(s['accounting']['stages_us']) for s in samples)))
            for name in names:
                lines.append(f"{name} | {statistics.median(s['accounting']['stages_us'][name] for s in samples):.3f}")
            for sample in samples:
                timeline = decode(sample['streaming_cycles'][0]['core_timeline_raw'])
                updates = [c for c in timeline['cores'] if c['role'].startswith('index_update ')]
                last = max(updates, key=lambda c: c['end_us'])
                lines.append(f"round={sample['round']} last_Update={last['block']} last_kernel_block={timeline['last_block']} span={timeline['observed_span_us']:.3f}")
                lines.append('Update block | prefetch_apply | Q1_apply | Q2_apply | Q3_apply | Q4_apply | Q_wait_sum | metadata | end')
                for core in updates:
                    stages = {s['name']: s for s in core['stages']}
                    values = [stages.get(name, {}).get('duration_us', 0.) for name in
                              ('apply.prefetch', 'apply.Q1', 'apply.Q2', 'apply.Q3', 'apply.Q4')]
                    values += [sum(s['duration_us'] for s in core['stages'] if s['name'].startswith('wait.Q')),
                               stages['metadata']['duration_us'], core['end_us']]
                    lines.append(str(core['block']) + ' | ' + ' | '.join(f'{v:.3f}' for v in values))
                lines.append('Last Update: stage | start | end | duration')
                for stage in last['stages']:
                    lines.append(f"{stage['name']} | {stage['start_us']:.3f} | {stage['end_us']:.3f} | {stage['duration_us']:.3f}")
    for batch in sorted({b for b, _ in results}):
        a, b = results[batch, 'A'], results[batch, 'B']
        lines.append(f'AB BS={batch}: graph {a:.3f} -> {b:.3f} us; change={(b/a-1)*100:+.2f}% (negative is faster)')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--description')
    args = parser.parse_args()
    text = summarize(args.root, args.description)
    (args.root / 'summary.log').write_text(text)
    print(text, end='')
    print(f'Summary: {args.root / "summary.log"}')
