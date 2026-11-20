# SPDX-License-Identifier: Apache-2.0
"""Compare the learned route with the requested native/fixed performance target."""

import argparse
import json
from pathlib import Path

import numpy as np

from peer_candidate import RANKED_BASE
from ranked_auto_report import FIXED, GROUPS, comparison, headline, load, pair
from ranked_help_report import matched_samples
from ranked_index_report import fmt, graph_savings, li_launch_means, marker, routed_arm
from ranked_joint_report import drift, joint_endpoints


PREVIOUS = '91c0271ce98c19a50bab59530d76a9f82de07b00'


def screen(target, auto, match, drifts, group):
    """A two-percent observational screen over groups with matched route and workload."""
    n = auto['timing']['graph_calls']
    desired = 'low' if group == 2 else 'high'
    if auto['route']['measured'][desired] != n or auto['route']['diagnostic'][desired] != n:
        return 'unexpected_route'
    if match and match['n'] * 2 < n:
        return 'insufficient_same_TopK'
    if max(abs(x) for values in drifts for x in values) > .02 * headline(target)[1]:
        return 'drift_exceeds_2pct'
    left, right = headline(target), headline(auto)
    changes = [(b-a)/a for a, b in zip(left, right)]
    if match:
        changes.append(-match['span_p50_saved_us']/left[1])
    if max(changes) < -.02:
        return 'faster_gt2pct'
    if max(changes) <= .02:
        return 'within_2pct_or_better'
    return 'regression_or_mixed'


def report(root, reference, previous, group):
    case = root / f'g{group}'
    revision = marker(root, 'candidate.txt')
    for baseline in (previous, reference):
        for name in ('host.txt', 'physical_npu.txt', f'g{group}/input.sha256'):
            if marker(root, name) != marker(baseline, name):
                raise ValueError(f'G{group}: archived host, physical card or input differs')
    if (marker(root, 'fixed.txt') != FIXED or marker(reference, 'baseline.txt') != RANKED_BASE
            or marker(previous, 'candidate.txt') != PREVIOUS):
        raise ValueError('Unexpected fixed, original or previous control')
    auto = routed_arm(root, group, 'auto', revision)
    label = 'serial32' if group == 2 else 'fixed'
    target = load(root, group, label, FIXED if group != 2 else None)
    if group != 2 and (target['binary'] != marker(root, f'ranked-{FIXED[:8]}.binary.sha256')
                       or target['config'].get('ranked_auto_route') or target['config'].get('ranked_q_workers') != 2):
        raise ValueError(f'G{group}: incorrect fixed fusion binary or configuration')
    original = load(reference, group, 'baseline', RANKED_BASE)
    prior = routed_arm(previous, group, 'auto', PREVIOUS)
    controls = {label: target, 'original_archive': original, 'previous_archive': prior}
    deltas = {key: comparison(arm, auto) for key, arm in controls.items()}
    drifts = [drift(a['timing']['samples_us']) for a in (target, auto)]
    matches = {key: matched_samples(arm, auto) for key, arm in controls.items() if key != 'serial32'}
    match = matches.get(label)
    n = auto['timing']['graph_calls']
    route = auto['route']['measured']
    verdict = screen(target, auto, match, drifts, group)
    print(f'G{group} GRAPH target={label} event_mean/span_p50_us={pair(headline(target))}->{pair(headline(auto))} '
          f'saved_us={pair(deltas[label]["saved_us"], True)} gain_pct={pair(deltas[label]["gain_pct"], True)} '
          f'route_low/high={route["low"]}/{route["high"]} screen={verdict}')
    print(f'G{group} ARCHIVE original_ranked_saved_us={pair(deltas["original_archive"]["saved_us"], True)} '
          f'previous_auto_saved_us={pair(deltas["previous_archive"]["saved_us"], True)}')
    exact = 'native_independently_validated' if match is None else f'{match["n"]}/{n} matched_span_p50_saved_us={fmt(match["span_p50_saved_us"])}'
    original_match = matches['original_archive']
    print(f'G{group} SAME_TOPK target={exact} original={original_match["n"]}/{n} '
          f'original_matched_span_p50_saved_us={fmt(original_match["span_p50_saved_us"])} '
          f'last5_minus_first5_span_mean/p50_us target={pair(drifts[0], True)} auto={pair(drifts[1], True)}')
    if group == 2:
        print('G2 KERNEL_MEAN_SAVED_us vs_previous_auto ' + ' '.join(
            f'{k}={v:+.2f}' for k, v in graph_savings(prior, auto).items()))
        print(f'G2 LI_MEAN_us first/threshold/cold/final previous={pair(li_launch_means(prior))} '
              f'auto={pair(li_launch_means(auto))}')
        for name, arm in ((label, target), ('auto', auto)):
            print(f'G{group} KERNEL_MEAN_us {name} '+' '.join(f'{k}={v:.2f}' for k, v in arm['timing']['stage_mean_us'].items()))
    else:
        print(f'G{group} KERNEL_MEAN_SAVED_us vs_fixed '+' '.join(f'{k}={v:+.2f}' for k, v in graph_savings(target, auto).items()))
        print(f'G{group} LI_FIRST_us fixed/auto={pair(li_launch_means(a)[0] for a in (target, auto))}')
    stage_keys = ('routing_probe_aic', 'routing_probe_aiv', 'planning', 'final_get_post', 'final_get_cq_wait')
    if group == 2:
        stage_keys += ('direct_plan_and_describe', 'index_active_work')
    aux = {}
    for name, arm in (('target', target), ('auto', auto)):
        aux[name] = {k: arm['detail'].get('stages', {}).get(k, {}).get('p50_us') for k in stage_keys}
        if group != 2:
            with np.load(arm['directory'] / 'lean_ticks.npz', allow_pickle=False) as ticks:
                aux[name]['last_Index_minus_CQ'] = joint_endpoints(ticks)['last_index_minus_last_cq']['p50']
    print(f'G{group} AUX_P50_us target/auto '+' '.join(
        f'{k}={fmt(v)}/{fmt(aux["auto"][k])}' for k, v in aux['target'].items()))
    probes = auto['detail']['stages']
    print(f'G{group} ROUTE_TIMERS candidate_nonzero/eligible ' + ' '.join(
        f'{k}={probes[k]["n"]}/{probes[k]["eligible"]}' for k in ('routing_probe_aiv', 'routing_probe_aic')))
    traffic = auto['detail']['traffic']
    lifecycle = 'PASS' if group in (2, 7) else 'covered_by_G2_G7'
    print(f'G{group} CHECK TopK/KV/ordered_Index/hot_valid=PASS lifecycle={lifecycle} '
          f'prefetch/final/unused_p50={pair(traffic[k]["p50"] for k in ("prefetch", "final", "unused_prefetch"))}')
    result = dict(revision=revision, target=label, headline={name: headline(a) for name, a in {**controls, 'auto': auto}.items()},
                  comparisons=deltas, matched=matches, drifts=drifts, auxiliary_us=aux, route=route, screen=verdict)
    (case/'parity.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--previous', required=True, type=Path)
    parser.add_argument('--group', required=True, type=int, choices=GROUPS)
    args = parser.parse_args()
    report(args.root, args.reference, args.previous, args.group)
