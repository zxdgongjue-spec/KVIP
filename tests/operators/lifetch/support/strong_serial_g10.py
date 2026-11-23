"""Measure and reuse the local two-Get-worker G10 reference for the same capture."""

import argparse
import hashlib
import os
import platform
import sys
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median

from li_restore_archive import check_snapshot, dp_capture_id
from li_restore_report import saved, stats
from peer_delta import pair_placement
from query_owner_report import load as load_candidate
from ranked_auto_report import load as load_serial
from ranked_integration_sets import INPUT_AND_RESULTS

GROUP = 10
OWNER_TOPK = '888db6b1603d22f4142bd2ccffe0518611856a31'
MARKER = 'host24-npu4-g10-serial2.json'


def from_this_host(root):
    host = platform.node()
    if not (root / 'host.txt').is_file() or (root / 'host.txt').read_text().strip() != host:
        return False
    for marker, current in (('machine_id.txt', '/etc/machine-id'),
                            ('boot_id.txt', '/proc/sys/kernel/random/boot_id')):
        recorded = root / marker
        if recorded.is_file() and Path(current).is_file():
            actual = Path(current).read_text().strip()
            if actual and recorded.read_text().strip() == actual:
                return True
    # Earlier ranked runs folded host, boot and physical device into the run key.
    parts = ('candidate.txt', 'reference.txt', 'harness.txt', 'run.key')
    boot = Path('/proc/sys/kernel/random/boot_id')
    if not boot.is_file() or not all((root / name).is_file() for name in parts):
        return False
    candidate, reference, harness, key = ((root / name).read_text().strip() for name in parts)
    record = [candidate, reference, host, boot.read_text().strip(), harness, '4']
    return hashlib.sha256(('\n'.join(record) + '\n').encode()).hexdigest()[:20] == key


def serial_measurement(root):
    if not from_this_host(root) or (root / 'physical_npu.txt').read_text().strip() != '4':
        raise ValueError('Expected the serial G10 replay on physical NPU4')
    arm = load_serial(root, GROUP, 'serial2')
    cfg, timing = arm['config'], arm['timing']
    if (cfg['device'] != 0 or cfg['round_range'] != [1, 1] or cfg['mode'] != 'diagnose'
            or cfg['input'] != 'real' or cfg['iterations'] != 20 or cfg.get('serial_workers', 2) != 2
            or cfg['prefetch_tokens'] != 1024
            or cfg['prefetch_rank_start'] != 1024 or cfg['intermediate_fetches'] != 4):
        raise ValueError('Expected G10 real input round 1, four warmups and 20 calls')
    graph = timing['event_mean_us'], timing['p50_us']
    if (not all(isfinite(v) and v > 0 for v in graph)
            or not isclose(median(timing['samples_us']), graph[1], abs_tol=.005)):
        raise ValueError('Invalid G10 graph event mean or invocation P50')
    durations = {key: [] for key in ('li', 'lidu_prepare', 'fetch')}
    for item in arm['samples']:
        groups = item['intervals_us']
        if set(groups) != set(durations) or any(len(groups[key]) != 1 for key in durations):
            raise ValueError('Expected native LI, LIDU prepare, then BatchGet in every serial call')
        previous = None
        for key in durations:
            start, stop = groups[key][0]
            if (not isfinite(start) or not isfinite(stop) or start >= stop
                    or (previous is not None and start < previous - 1e-5)
                    or not isclose(item['stages_us'][key], stop - start, abs_tol=.005)):
                raise ValueError(f'Invalid serial interval for {key}')
            durations[key].append(stop - start)
            previous = stop
    durations['tail'] = [a + b for a, b in zip(durations['lidu_prepare'], durations['fetch'])]
    stages = {key: (mean(values), median(values)) for key, values in durations.items()}
    return arm, graph, stages


def same_capture(serial_inputs, inputs):
    expected = INPUT_AND_RESULTS[GROUP][0]
    # Either side can be a single merged export of the eight DP ranks.
    if dp_capture_id(GROUP, serial_inputs) == expected:
        check_snapshot(GROUP, expected, serial_inputs, inputs)
    else:
        if dp_capture_id(GROUP, inputs) != expected:
            raise ValueError('Cannot bind G10 to the eight-rank source-round-51 reference')
        check_snapshot(GROUP, expected, inputs, serial_inputs)


def reference_root(area):
    area = area.resolve()
    marker = area / MARKER
    raw = marker.resolve(strict=True)
    root = area / raw.relative_to(area).parts[0]
    paths = list((root / 'g10/serial2/results').glob('*/cases/diagnose/serial/path_result.json'))
    if paths != [raw]:
        raise ValueError(f'Invalid G10 strong serial reference target: {raw}')
    return root


def gain_line(reference, graph, root):
    gain = '/'.join(f'{100 * (a - b) / a:+.2f}' for a, b in zip(reference, graph))
    return (f'G10 vs_serial2_archive historical_unpaired_host24_npu4 '
            f'graph_saved_mean/P50_us={saved(reference, graph)} gain_mean/P50_pct={gain} '
            f'gap_to_35_mean/P50_us={saved(graph, [0.65*v for v in reference])} source={root.name}')


def comparison_line(area, inputs, graph):
    if not os.path.lexists(area / MARKER):
        return 'G10 vs_serial2_archive UNAVAILABLE missing_local_reference'
    root = reference_root(area)
    serial, reference, _ = serial_measurement(root)
    same_capture(serial['inputs'], inputs)
    return gain_line(reference, graph, root)


def report(root, candidate, *, publish):
    if ((candidate / 'logical-ties.txt').read_text().strip() != OWNER_TOPK
            or (candidate / 'physical_npu.txt').read_text().strip() != '4'):
        raise ValueError('Expected the completed B24 NPU4 per-query final TopK archive')
    current = load_candidate(candidate, GROUP, 'logical-ties', row_dependencies=True)
    serial, reference, stages = serial_measurement(root)
    same_capture(serial['inputs'], current[0])
    if publish:
        name = root.parent / MARKER
        raw, = (root / 'g10/serial2/results').glob('*/cases/diagnose/serial/path_result.json')
        try:
            name.symlink_to(raw.relative_to(root.parent))
        except FileExistsError:
            # Two completed runs must not silently replace each other's benchmark.
            other = reference_root(root.parent)
            if other != root:
                return report(other, candidate, publish=False)
    print(f'G10 strong_serial_2get CHECK correctness=PASS calls=20/20 get_workers=2 '
          f'input=capture_round51 physical_NPU=4 source={root}')
    print(f'G10 strong_serial_2get graph_mean/P50_us={stats(reference)}',
          *(f'{key}_mean/P50_us={stats(pair)}' for key, pair in stages.items()))
    graph = current[3:5]
    placement = pair_placement(serial['placement'], current[2], local_baseline=True)
    print(f'G10 archived_owner_topk graph_mean/P50_us={stats(graph)} source={candidate.name}')
    print(gain_line(reference, graph, root), f'placement={placement}')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode', choices=('publish', 'cached'))
    cli.add_argument('root', type=Path, help='new result root for publish, peer-baseline directory for cached')
    cli.add_argument('candidate', type=Path, help='completed per-query final TopK result root')
    args = cli.parse_args()
    where = args.root.resolve()
    if args.mode == 'cached':
        if not os.path.lexists(where / MARKER):
            for path in sorted(where.glob('*/g10/serial2'), key=lambda p: p.stat().st_mtime, reverse=True):
                root = path.parents[1]
                if from_this_host(root):
                    try:
                        report(root, args.candidate, publish=True)
                        return 0
                    except (FileNotFoundError, KeyError, ValueError, IndexError, TypeError) as error:
                        print(f'G10 skipped_serial_archive={root} reason={error}', file=sys.stderr)
            return 3
        where = reference_root(where)
    report(where, args.candidate, publish=args.mode == 'publish')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
