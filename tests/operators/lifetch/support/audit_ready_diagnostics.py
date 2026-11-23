"""Compare standalone LI diagnostics with the same B24 candidate's prior diagnostics."""
import argparse
import json
from pathlib import Path
from statistics import mean, median

GROUPS = (10, 4, 17, 7, 2)
BASELINE = 'aa6889e773c57ff737fd5ddec75307613d8949dc'


def sample(detail):
    head = detail['streaming_cycles'][0]
    mhz = head['terminal_timeline']['clock_mhz']
    events = [e for e in head['prefetch_services'] if e['epoch'] == 3]
    if mhz <= 0 or len(events) != 16 or {e['lane'] for e in events} != set(range(16)):
        raise ValueError('Expected one final-window event per row')
    critical = max(events, key=lambda e: (e['end_cycles'], -e['lane']))
    coverage = {(r, q): covered for r, q, w, ready, covered, rank in detail['rank_progress']
                if w == 3 and r >= 8}
    if len(coverage) != 32:
        raise ValueError('Missing final-window progress for the eight high-numbered rows')
    return (critical['incremental_cycles'] / mhz, mean(coverage.values()),
            sum(e['posted_count'] for e in events))


def compare(group, old, new):
    if len(old) != 20 or not new:
        raise ValueError(f'G{group}: missing diagnostic calls')
    before = list(zip(*(sample(d) for d in old)))
    after = list(zip(*(sample(d) for d in new)))
    parts = [f'G{group} DIAGNOSTIC_COMPARISON archive_calls={len(old)} diagnostic_calls={len(new)}']
    outside = []
    for name, a, b in zip(('last_row_four_Q_us', 'rows8_to15_coverage_pct', 'fourth_window_posts'), before, after):
        parts.append(f'{name}_means={mean(a):.2f}/{mean(b):.2f}')
        if not min(a) <= median(b) <= max(a):
            outside.append(name)
    parts.append('diagnostic_median_outside_archive_range=' + (','.join(outside) or 'none'))
    return ' '.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('baseline', type=Path)
    args = parser.parse_args()
    if ((args.baseline/'candidate.txt').read_text().strip() != BASELINE or
            (args.baseline/'physical_npu.txt').read_text().strip() != '4'):
        raise ValueError('Expected the dual-window candidate archive from physical NPU4')
    for g in GROUPS:
        cases = args.baseline / f'g{g}/li-window/results'
        path, = cases.glob('*/cases/diagnose/fused/diagnose/round-1/detail_samples.json')
        old = json.loads(path.read_text())
        new = json.loads((args.root/f'g{g}/detail_samples.json').read_text())
        print(compare(g, old, new))


if __name__ == '__main__':
    main()
