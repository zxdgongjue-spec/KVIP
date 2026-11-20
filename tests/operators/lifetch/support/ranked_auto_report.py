# SPDX-License-Identifier: Apache-2.0
"""Matched fixed/learned replay. Times in us, positive savings favor auto."""

import argparse
import json
import re
from pathlib import Path
from statistics import median

import numpy as np

from peer_candidate import RANKED_BASE
from peer_delta import pair_placement, read
from ranked_help_report import matched_samples
from ranked_joint_report import drift, early_index_stats, joint_endpoints

FIXED = '6bca602ad9e548e9f12c0142fea25f2bc5c6877d'
GROUPS = (2, 7, 10, 17, 4)


def load(root, group, label, revision=None):
    folder = root / f'g{group}' / label
    native = label.startswith('serial')
    path, = (folder / 'results').glob(f'**/{"serial" if native else "fused"}/path_result.json')
    result = read(path)
    cfg = read(path.parents[1] / 'config.json')
    inp = read(path.parents[1] / 'replay_input.json')
    timing = result['all_rounds']
    profile = read(path.parent / 'diagnose/round-1/graph_profile.json')
    place = read(folder / 'thp.json')
    counts = timing['samples']
    expected_workers = int(label[6:]) if native else 40
    if (cfg['group'] != group or cfg['warmup'] != 4 or result['correctness'] != 'PASS'
            or result['failures'] or result['transport'] != 'urma' or result['runtime_workers'] != expected_workers
            or timing['timing_protocol'] != 'one_profiled_graph' or timing['measured_graphs'] != 1
            or timing['graph_calls'] != counts or counts != cfg['iterations'] or counts < 2
            or profile['accounting'].get('unclassified')
            or result['input_paths'] != inp['source_paths'] or inp['rounds'] != 1
            or place['status'] != 'measured' or place['page_mode'] != 'thp'
            or place['bytes'] != (inp['page_count']+2)*656*128):
        raise ValueError(f'{label}: incomplete replay, baseline, or placement')
    geometry = inp['geometry']
    if (any(geometry.get(k) != v for k, v in dict(batch=16, heads=32, token_bytes=656,
                                                   resident=16384, history=131328).items())
            or sorted(r[0] for r in inp['identities']) != sorted(list(range(8))*2)):
        raise ValueError(f'{label}: wrong workload')
    samples = profile['samples']
    if (len(samples) != counts or len(timing['samples_us']) != counts
            or any(s['invocation'] != i or abs(s['us']-t) > .01
                   for i, (s, t) in enumerate(zip(samples, timing['samples_us'])))):
        raise ValueError(f'{label}: profiler samples differ from headline')
    stages = timing['stage_mean_us']
    accounted = (sum(stages.values()) + timing['inter_kernel_gap_mean_us'] - timing['overlap_mean_us']
                 + timing['outside_invocations_mean_us'])
    if abs(accounted-timing['event_mean_us']) > .02:
        raise ValueError(f'{label}: device timing does not reconcile')
    digests = result.get('topk_set_digests', [])
    if not native and (result['production_revision'] != revision or len(result['binary_sha256']) != 64
                       or len(digests) != counts or not all(re.fullmatch('[0-9a-f]{64}', x) for x in digests)):
        raise ValueError(f'{label}: wrong fused source or incomplete TopK verification')
    diagnostic_dir = path.parent / 'diagnose/round-1'
    detail = {} if native else read(diagnostic_dir / 'lean_stages.json')
    return dict(inputs=inp, config=cfg, timing=timing, samples=samples, digests=digests, placement=place,
                detail=detail, directory=diagnostic_dir, route=result.get('auto_route'), binary=result.get('binary_sha256'))


def verify_pair(left, right):
    keys = ('group', 'device', 'transport', 'warmup', 'iterations', 'round_range', 'kvaf_root',
            'prefetch_rank_start', 'intermediate_fetches', 'prefetch_tokens')
    if left['inputs'] != right['inputs'] or any(left['config'][k] != right['config'][k] for k in keys):
        raise ValueError('Different captured requests, geometry, KVAF, or timing protocol')
    if pair_placement(left['placement'], right['placement'], local_baseline=True) != 'matched':
        raise ValueError('CPU NUMA, THP or transport device differs across arms')


def headline(arm):
    timing = arm['timing']
    return timing['event_mean_us'], timing['p50_us']


def pair(values, signed=False):
    return '/'.join(format(v, '+.2f' if signed else '.2f') for v in values)


def comparison(left, right):
    verify_pair(left, right)
    a, b = headline(left), headline(right)
    saved = [x-y for x, y in zip(a, b)]
    return dict(saved_us=saved, gain_pct=[100*s/x for s, x in zip(saved, a)])


def main(root, reference, group):
    sha = (root/'candidate.txt').read_text().strip()
    if (not re.fullmatch('[0-9a-f]{40}', sha) or sha == FIXED or (root/'fixed.txt').read_text().strip() != FIXED
            or (reference/'baseline.txt').read_text().strip() != RANKED_BASE):
        raise ValueError('Unexpected control or candidate revision')
    digest = (root/f'g{group}/input.sha256').read_text().strip()
    if not re.fullmatch('[0-9a-f]{64}', digest) or digest != (reference/f'g{group}/input.sha256').read_text().strip():
        raise ValueError('Capture bytes differ from original')
    fixed, auto = load(root, group, 'fixed', FIXED), load(root, group, 'auto', sha)
    for arm, revision in ((fixed, FIXED), (auto, sha)):
        compiled = (root/f'ranked-{revision[:8]}.binary.sha256').read_text().strip()
        if arm['binary'] != compiled:
            raise ValueError('Loaded library differs from preflight build')
    if fixed['binary'] == auto['binary'] or any((a['config'].get('ranked_q_workers') or 2) != 2 for a in (fixed, auto)):
        raise ValueError('Unexpected binary or final communication width')
    controls = {'fixed':fixed, 'original_archive':load(reference, group, 'baseline', RANKED_BASE)}
    if group == 2:
        for label in ('serial2', 'serial32'):
            controls[label] = load(root, group, label)
    comparisons = {k:comparison(a, auto) for k, a in controls.items()}
    route = auto['route']
    measured, cold = route['measured'], route['warmup'][0]
    n = auto['timing']['graph_calls']
    if (route['policy'] != 'zero' or cold['high'] != n or measured['low']+measured['high'] != n
            or len(measured['feedback_demand_per_call']) != n):
        raise ValueError('Missing device route coverage')
    match = matched_samples(fixed, auto)
    deltas = comparisons['fixed']
    within = {k: drift(a['timing']['samples_us']) for k, a in {'fixed':fixed,'auto':auto}.items()}
    enough = match['n'] >= (n+1)//2
    expected = measured['low' if group == 2 else 'high'] == n
    pct = deltas['gain_pct']
    exact = match['span_p50_saved_us']
    span = headline(fixed)[1]
    if not expected:
        verdict = 'mixed_or_unexpected_route'
    elif not enough:
        verdict = 'insufficient_same_TopK'
    elif max(abs(x) for values in within.values() for x in values) > .02*span:
        verdict = 'drift_exceeds_2pct'
    elif group == 2:
        verdict = 'low_gain_gt2pct' if min(pct)>2 and exact > .02*span else 'low_gain_not_established'
    elif min(pct)>2 and exact>.02*span:
        verdict = 'high_faster_gt2pct'
    elif max(abs(x) for x in pct)<=2 and abs(exact)<=.02*span:
        verdict = 'high_within_2pct'
    else:
        verdict = 'high_outside_band_or_mixed'
    lifecycle = route.get('lifecycle') or {}
    if group in (2, 7) and not (lifecycle.get('misprediction_recovered') and lifecycle.get('invalid_identity_preserved')):
        raise ValueError('Missing route lifecycle checks')
    print(f'G{group} ROUTE cold_high={cold["high"]}/{n} cold_event_mean_us={route["cold_event_mean_us"]:.2f} '
          f'steady_low/high={measured["low"]}/{measured["high"]} '
          f'Get_p50={median(measured["feedback_demand_per_call"]):g} screen={verdict}')
    for label, arm in controls.items():
        delta = comparisons[label]
        print(f'G{group} {label}->auto event_mean/span_p50_us={pair(headline(arm))}->{pair(headline(auto))} '
              f'saved_us={pair(delta["saved_us"], True)} gain_pct={pair(delta["gain_pct"],True)}')
    if group == 2:
        threshold = [min(x) for x in zip(headline(controls['serial2']), headline(controls['serial32']))]
        parity = all(value <= bound for value, bound in zip(headline(auto), threshold))
        print(f'G2 SERIAL_PARITY mean_and_p50_vs_both={parity}')
    print(f'G{group} SAME_TOPK={match["n"]}/{n} paired_span_p50_saved_us={exact if exact is None else round(exact,2)} '
          f'DRIFT last5-first5_span_mean/p50_us fixed={pair(within["fixed"],True)} auto={pair(within["auto"],True)}')
    stages = fixed['timing']['stage_mean_us']
    print(f'G{group} KERNEL_MEAN_SAVED_us ' + ' '.join(
        f'{s}={stages[s]-auto["timing"]["stage_mean_us"][s]:+.2f}' for s in stages))
    def aux(arm, name):
        value = arm['detail']['stages'].get(name, {}).get('p50_us')
        return 'NA' if value is None else f'{value:.2f}'
    print(f'G{group} STAGE_P50_us fixed/auto ' + ' '.join(
        f'{s}={aux(fixed,s)}/{aux(auto,s)}' for s in ('routing_probe_aiv', 'planning', 'direct_plan_and_describe', 'final_get_post',
                                                      'index_active_work', 'final_get_cq_wait')))
    high = {}
    for label, arm in (('fixed', fixed), ('auto', auto)):
        if label == 'auto' and route['diagnostic']['high'] != n:
            continue
        with np.load(arm['directory']/'lean_ticks.npz', allow_pickle=False) as ticks:
            team = joint_endpoints(ticks)
            early = early_index_stats(ticks)
        high[label] = dict(early_reads=early['hits'], early_moves=early['moves'],
                           last_index_minus_CQ_p50_us=team['last_index_minus_last_cq']['p50'])
    if 'auto' in high:
        print(f'G{group} HIGH_INDEX fixed/auto early_reads={high["fixed"]["early_reads"]}/{high["auto"]["early_reads"]} '
              f'last_Index_minus_CQ_p50_us={high["fixed"]["last_index_minus_CQ_p50_us"]}/{high["auto"]["last_index_minus_CQ_p50_us"]}')
    print(f'G{group} CHECK TopK/KV/Index=PASS lifecycle={"PASS" if lifecycle else "covered_by_G2_G7"}')
    result = dict(comparisons=comparisons, route=route, matched=match, drift=within, high_index=high, screen=verdict,
                  headline={k:headline(a) for k,a in {**controls,'auto':auto}.items()})
    (root/f'g{group}/auto_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--group', type=int, choices=GROUPS, required=True)
    args = parser.parse_args()
    main(args.root, args.reference, args.group)
