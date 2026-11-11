"""Compare graph timing and same-replay finalization, including every Fetch core."""
import argparse
import json
import statistics
from pathlib import Path

from summarize_core_timeline import decode


def summarize(root):
    lines = [
        'Graph timings are unprofiled. All core times below are separately profiled, calibrated us.',
        'Tail is Get-return to core-exit (includes diagnostic writes), not extra network latency.',
        'version | BS | graph_p50_us | graph_p99_us | profiled_fused_median_us',
    ]
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
            lines.append(f"{version} | {batch} | {timing['p50_us']:.3f} | {timing['p99_us']:.3f} | {fused:.3f}")
            results[batch, version] = timing['p50_us']
            for sample in samples:
                raw = sample['streaming_cycles'][0]['core_timeline_raw']
                timeline = decode(raw)
                lines.append(f"  round={sample['round']} last_block={timeline['last_block']} span_us={timeline['observed_span_us']:.3f}")
                last = timeline['cores'][timeline['last_block']]
                # Complete stage list for whichever core actually finishes last.
                lines.append('  LAST CORE: stage | start | end | duration')
                for s in last['stages']:
                    lines.append(f"    {s['name']} | {s['start_us']:.3f} | {s['end_us']:.3f} | {s['duration_us']:.3f}")
                lines.append('  Fetch block | Q/lane | Get_end | index_wait | residency_ack | diagnostic_write | core_end | tail')
                for c in timeline['cores'][2*batch:]:
                    stages = {s['name']: s for s in c['stages']}
                    get = stages.get('conflict.KVAF_Get', stages.get('safe.KVAF_Get'))
                    if get is None:
                        continue  # inactive lane when BS < workers
                    def duration(key):
                        return f"{stages[key]['duration_us']:.3f}" if key in stages else '--'
                    lines.append(f"    {c['block']} | {c['role']} | {get['end_us']:.3f} | "
                                 + ' | '.join(duration(k) for k in ('finalize.wait_index', 'finalize.residency_and_ack', 'finalize.diagnostic_write'))
                                 + f" | {c['end_us']:.3f} | {c['end_us']-get['end_us']:.3f}")
    batches = {b for b, _ in results}
    for batch in sorted(batches):
        before, after = results[batch, 'A'], results[batch, 'B']
        lines.append(f'AB BS={batch}: full graph {before:.3f} -> {after:.3f} us; time change={(after/before-1)*100:+.2f}% (negative is faster)')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    text = summarize(args.root)
    (args.root / 'summary.log').write_text(text)
    print(text, end='')
    print(f'Summary: {args.root / "summary.log"}')


if __name__ == '__main__':
    main()
