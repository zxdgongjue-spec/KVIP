# SPDX-License-Identifier: Apache-2.0
"""Report matched G2 graph timings and account for the measured candidate delta."""

import argparse
import hashlib
import json
from pathlib import Path

ARMS = ('serial', 'main', 'before', 'after', 'before-end', 'serial-end')
CONTROL = 'cdfc4da4b53b80255740183c7a0a8db8bf350dcb'
CANDIDATE = '4262f8e6be86c7da65cc3771b97d30467bb14e4e'
PEER_SERIAL = (298.718547821, 292.6975)
CLEAN_SERIAL = (300.992, 294.344)
PEER_FUSED = (325.835752487, 313.7135)


def read(path):
    return json.loads(path.read_text())


def load_arm(root, label):
    path = 'serial' if label.startswith('serial') else 'fused'
    files = list((root / label / 'results').glob(f'**/{path}/path_result.json'))
    if len(files) != 1:
        raise ValueError(f'{label}: found {len(files)} path results')
    file = files[0]
    result = read(file)
    inp = read(file.parents[1] / 'replay_input.json')
    ids = json.dumps(inp['identities'], separators=(',', ':')).encode()
    geometry = dict(history=131328, resident=16384, heads=32, token_bytes=656,
                    output_stride=16896, batch=16)
    if (hashlib.sha256(ids).hexdigest() !=
            '83bef8d67299723cbd9e04eaf14b9e270bbfdb2609bca771a8aee9c1eabd92bc'
            or inp['page_count'] != 16416 or inp['rounds'] != 1
            or any(inp['geometry'].get(k) != v for k, v in geometry.items())):
        raise ValueError(f'{label}: G2 request order or page layout changed')
    timing = result['all_rounds']
    if (result['correctness'] != 'PASS' or result['failures'] or result['path'] != path
            or result['transport'] != 'urma' or result['runtime_workers'] != (2 if path == 'serial' else 40)
            or timing['timing_protocol'] != 'one_profiled_graph' or timing['measured_graphs'] != 1
            or timing['graph_calls'] != 20 or timing['samples'] != 20):
        raise ValueError(f'{label}: device correctness or timing protocol mismatch')
    stages = timing['stage_mean_us']
    if path == 'serial' and set(stages) != {'li', 'lidu_prepare', 'fetch'}:
        raise ValueError(f'{label}: serial stage mismatch')
    profile = read(file.parent / 'diagnose/round-1/graph_profile.json')
    if profile['accounting'].get('unclassified') or len(profile['samples']) != 20:
        raise ValueError(f'{label}: unmatched kernel intervals')
    accounted = (sum(stages.values()) + timing['inter_kernel_gap_mean_us'] - timing['overlap_mean_us']
                 + timing['outside_invocations_mean_us'])
    if abs(accounted - timing['event_mean_us']) > .01:
        raise ValueError(f'{label}: timing accounting mismatch {accounted}')
    thp = read(root / label / 'thp.json')
    if thp['status'] != 'measured' or thp['page_mode'] != 'thp' or thp['bytes'] != 1378586624:
        raise ValueError(f'{label}: THP or region size mismatch')
    counts = None
    if label in ('before', 'after', 'before-end'):
        revision = CANDIDATE if label == 'after' else CONTROL
        route = result['auto_route']
        measured = route['measured']
        if (result['candidate_revision'] != revision or route['source'] != revision
                or len(result['binary_sha256']) != 64
                or [(w['low'], w['high']) for w in route['warmup']] != [(0,20), (20,0), (20,0), (20,0)]
                or (measured['low'], measured['high']) != (20,0)):
            raise ValueError(f'{label}: revision or route mismatch')
        counts = measured['necessary_get_per_call']
        if len(counts) != 20 or min(counts) < 0:
            raise ValueError(f'{label}: incomplete necessary Get counts')
    return timing, thp, counts


def scores(timing):
    return timing['event_mean_us'], timing['p50_us']



def bridge_matches(timing):
    return all(abs(v / ref - 1) <= .05 for refs in (CLEAN_SERIAL, PEER_SERIAL)
               for v, ref in zip(scores(timing), refs))


def placement_matches(thp):
    return (thp['coverage_min_pct'] >= 99 and set(thp['numa_pages_overlapping_vmas']) == {'0'}
            and thp['ub_device'] == 'udmac0d1e2')


def compare(base, new):
    pairs = list(zip(base, new))
    return ('saved_us=' + '/'.join(f'{a-b:+.3f}' for a, b in pairs)
            + ' gain_pct=' + '/'.join(f'{100*(a-b)/a:+.2f}' for a, b in pairs))


def arm_line(label, row):
    timing, thp, counts = row
    fields = ' '.join(f'{k}={v:.3f}' for k, v in timing['stage_mean_us'].items())
    traffic = '' if counts is None else f' route_LOW=20/20 necessary_Get={min(counts)}..{max(counts)}'
    print(f"ARM {label} event_mean={timing['event_mean_us']:.3f} span_mean={timing['profiler_mean_us']:.3f} "
          f"span_p50={timing['p50_us']:.3f} {fields} THP_min_pct={thp['coverage_min_pct']:.3f} "
          f"NUMA_pages={thp['numa_pages_overlapping_vmas']} ub_device={thp['ub_device']}{traffic}")


def report(root):
    rows = {name: load_arm(root, name) for name in ARMS}
    if not rows['before'][2] == rows['after'][2] == rows['before-end'][2]:
        raise ValueError('necessary Get counts changed across the three candidate runs')
    metrics = {k: scores(v[0]) for k, v in rows.items()}
    bridge_ok = all(bridge_matches(rows[key][0]) for key in ('serial', 'serial-end'))
    thp_ok = all(placement_matches(row[1]) for row in rows.values())
    print(f'CHECK serial_within_5pct_of_clean_and_peer={bridge_ok} placement_matches_clean={thp_ok} '
          'same_learned_route_and_necessary_Get=True units=us pairs=event_mean/span_p50')
    for name in ('serial', 'serial-end', 'main', 'before', 'before-end'):
        print(f'GAIN after_vs_{name} {compare(metrics[name], metrics["after"])}')
    print(f'REFERENCE after_vs_external_optimized {compare(PEER_FUSED, metrics["after"])}')
    for name in ('serial', 'before'):
        drifts = [b-a for a, b in zip(metrics[name], metrics[name+'-end'])]
        print(f'DRIFT {name} end_minus_start_us={drifts[0]:+.3f}/{drifts[1]:+.3f}')
    # Historical device spans came from a different replay protocol, with the
    # same candidate kernel source. Do not subtract its unprofiled event time.
    for label in ('before', 'before-end'):
        timing = rows[label][0]
        stages = timing['stage_mean_us']
        print(f'MIGRATION old_protocol_minus_{label} device_span_us={356.816-timing["profiler_mean_us"]:+.3f} '
              f'init={11.378-stages["init"]:+.3f} li={239.067-stages["li"]:+.3f} '
              f'tail={106.327-stages["resolve_update"]:+.3f} cross_protocol=true')
    for label in ('before', 'before-end'):
        before, after = rows[label][0], rows['after'][0]
        keys = sorted(before['stage_mean_us'].keys() | after['stage_mean_us'].keys())
        pieces = {key: before['stage_mean_us'].get(key, 0.) - after['stage_mean_us'].get(key, 0.)
                  for key in keys}
        pieces['gap'] = before['inter_kernel_gap_mean_us'] - after['inter_kernel_gap_mean_us']
        pieces['overlap_adjustment'] = after['overlap_mean_us'] - before['overlap_mean_us']
        pieces['outside_spans'] = before['outside_invocations_mean_us'] - after['outside_invocations_mean_us']
        print(f'DELTA {label}_minus_after mean_us ' + ' '.join(f'{k}={v:+.3f}' for k, v in pieces.items())
              + f' total={sum(pieces.values()):+.3f}')
    brackets = all(c < b for label in ('before', 'before-end')
                   for c, b in zip(metrics['after'], metrics[label]))
    print(f'OBSERVED both_metrics_beat_both_control_arms={brackets} matched_screen={bridge_ok and thp_ok}')
    print(f'RESULT_ROOT={root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('arm', nargs='?', choices=ARMS)
    args = parser.parse_args()
    if args.arm:
        row = load_arm(args.root, args.arm)
        arm_line(args.arm, row)
        if args.arm == 'serial' and not (bridge_matches(row[0]) and placement_matches(row[1])):
            raise SystemExit('BASELINE_RECHECK_FAILED: inspect the first ARM serial timing, THP, NUMA and device')
    else:
        report(args.root)
