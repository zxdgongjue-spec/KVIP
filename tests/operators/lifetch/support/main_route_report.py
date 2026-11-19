# SPDX-License-Identifier: Apache-2.0
"""Report same-input G2 phase costs without repeating any benchmark arm."""

import argparse
from pathlib import Path
from statistics import mean, median

from peer_delta import load_arm, pair_placement, read

LABELS = ('serial', 'main', 'zero', 'early')


def read_result(root, label):
    timing, placement, _ = load_arm(root, label)
    mode = 'serial' if label == 'serial' else 'fused'
    file, = (root / label / 'results').glob(f'**/{mode}/path_result.json')
    result = read(file)
    detail = {} if label == 'serial' else read(file.parent / 'diagnose/round-1/lean_stages.json')
    if label in ('zero', 'early'):
        expected_revision = (root / 'candidate.txt').read_text().strip()
        route = result['auto_route']
        if result.get('candidate_revision') != expected_revision or route['policy'] != label:
            raise ValueError(f'{label}: wrong production revision or routing policy')
        detail['measured_route'] = route['measured']
    profile = read(file.parent / 'diagnose/round-1/graph_profile.json')
    li_intervals = [sorted(sample['intervals_us']['li']) for sample in profile['samples']]
    if len({len(calls) for calls in li_intervals}) != 1:
        raise ValueError(f'{label}: LI dispatch count changed within the measured graph')
    timing['li_dispatch_mean_us'] = [mean(end - start for start, end in one_dispatch)
                                      for one_dispatch in zip(*li_intervals)]
    timing['stage_p50_us'] = {k: median(s['stages_us'][k] for s in profile['samples'])
                              for k in timing['stage_mean_us']}
    if label == 'serial':
        timing['tail_p50_us'] = median(s['stages_us']['lidu_prepare'] + s['stages_us']['fetch']
                                       for s in profile['samples'])
    else:
        timing['tail_p50_us'] = timing['stage_p50_us']['resolve_update']
    return timing, placement, detail


def number(value):
    return 'NA' if value is None else f'{value:.2f}'


def saved(base, current):
    return f'{base-current:+.3f} ({100*(base-current)/base:+.2f}%)'


def delta(base, current):
    return 'NA' if base is None or current is None else f'{base-current:+.2f}'


def print_arm(label, timing, detail):
    spans = timing['samples_us']
    first, last = spans[:5], spans[-5:]
    print(f"ARM {label} event_mean={timing['event_mean_us']:.3f} span_p50={timing['p50_us']:.3f} "
          f"span_mean={timing['profiler_mean_us']:.3f} "
          f"within_graph_last5_minus_first5_mean/p50={mean(last)-mean(first):+.3f}/{median(last)-median(first):+.3f}")
    print('KERNEL '+label+' mean/p50 '+' '.join(f"{k}={v:.3f}/{timing['stage_p50_us'][k]:.3f}"
                                                for k, v in timing['stage_mean_us'].items())
          + f" gap={timing['inter_kernel_gap_mean_us']:.3f} overlap={timing['overlap_mean_us']:.3f}"
            f" outside_calls={timing['outside_invocations_mean_us']:.3f}")
    if len(timing['li_dispatch_mean_us']) > 1:
        print(f'LI_DISPATCH {label} mean_us_by_launch=' +
              '/'.join(f'{v:.2f}' for v in timing['li_dispatch_mean_us']))
    if detail:
        counts = detail['traffic']
        if 'measured_route' in detail:
            observed = detail['measured_route']
            print(f"ROUTE {label} measured_low/high={observed['low']}/{observed['high']} "
                  f"feedback_reads_p50={median(observed['feedback_demand_per_call']):g} "
                  f"diagnostic_plan_rows={detail['direct_plan_rows']}")
        print('TRAFFIC '+label+' '+' '.join(
            f"{k}={number(v['p50'])}[{number(v['min'])},{number(v['max'])}]"
            for k, v in counts.items()))
        window_counts = '/'.join(number(v['p50']) for v in detail['window_posted'])
        active_requests = '/'.join(number(v['p50']) for v in detail['window_nonempty_requests'])
        groups = detail['diagnostic_final_get_submitted_groups']
        print(f'WINDOW {label} tokens_p50={window_counts} nonempty_requests_p50={active_requests} '
              f"final_Get_submitted_groups_p50={number(groups['p50'])}")
        for window in detail['window_stages']:
            if not window['service_us']['n']:
                continue
            print(f"WINDOW_TIME {label} epoch={window['epoch']} p50/max " +
                  ' '.join(f"{k}={number(window[k]['p50_us'])}/{number(window[k]['max_us'])}"
                           for k in ('service_us','scan_us','candidate_us','post_us','predecessor_wait_us')))
        stages = detail['stages']
        fields = []
        for name, metric in stages.items():
            if name.startswith('kernel_'):
                continue
            fields.append(f"{name}={number(metric['p50_us'])}/{number(metric['max_us'])}"
                          f"[{metric['n']}/{metric.get('eligible', metric['n'])}]")
        for start in range(0, len(fields), 5):
            print(f'DETAIL {label} '+' '.join(fields[start:start+5]))


def report(root, *, gains_only=False):
    rows = {label: read_result(root, label) for label in LABELS}
    if not gains_only:
        for label, (timing, _, detail) in rows.items():
            print_arm(label, timing, detail)
    for name in ('zero', 'early'):
        current, placement, study = rows[name]
        for baseline in ('serial', 'main'):
            ref, ref_place, _ = rows[baseline]
            stages, other = current['stage_mean_us'], ref['stage_mean_us']
            other_tail = other['resolve_update'] if baseline == 'main' else other['lidu_prepare'] + other['fetch']
            print(f"GAIN {name}_vs_{baseline} event_mean={saved(ref['event_mean_us'], current['event_mean_us'])} "
                  f"span_p50={saved(ref['p50_us'], current['p50_us'])} "
                  f"placement={pair_placement(placement, ref_place)}")
            print(f"KERNEL_DELTA {name}_vs_{baseline} saved_li_mean/p50={other['li']-stages['li']:+.3f}/"
                  f"{ref['stage_p50_us']['li']-current['stage_p50_us']['li']:+.3f} "
                  f"saved_tail_mean/p50={other_tail-stages['resolve_update']:+.3f}/"
                  f"{ref['tail_p50_us']-current['tail_p50_us']:+.3f} "
                  f"saved_gap={ref['inter_kernel_gap_mean_us']-current['inter_kernel_gap_mean_us']:+.3f} "
                  f"overlap_adjustment={current['overlap_mean_us']-ref['overlap_mean_us']:+.3f} "
                  f"saved_outside_calls={ref['outside_invocations_mean_us']-current['outside_invocations_mean_us']:+.3f}")
        main_detail = rows['main'][2]['stages']
        fields = []
        for k, v in study['stages'].items():
            if k.startswith('kernel_') or k not in main_detail:
                continue
            other = main_detail[k]
            if v.get('scope') != other.get('scope'):
                continue
            fields.append(f"{k}={delta(other['p50_us'], v['p50_us'])}/{delta(other['max_us'], v['max_us'])}")
        for start in range(0, len(fields), 5):
            print(f'DETAIL_DELTA {name}_vs_main '+' '.join(fields[start:start+5]))
    print('CHECK output_KV_Index=PASS timing_reconciled=PASS')
    print(f'RESULT_ROOT={root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--arm', choices=LABELS)
    action.add_argument('--gains-only', action='store_true')
    args = parser.parse_args()
    if args.arm:
        timing, _, detail = read_result(args.root, args.arm)
        print_arm(args.arm, timing, detail)
    else:
        report(args.root, gains_only=args.gains_only)
