# SPDX-License-Identifier: Apache-2.0
"""Compare the previous and new cooperative Index kernels on the same learned route."""

import argparse
import json
from pathlib import Path
import re
from statistics import median

import numpy as np

from peer_candidate import RANKED_BASE
from ranked_auto_report import GROUPS, comparison, headline, load, pair, verify_pair
from ranked_help_report import matched_samples
from ranked_joint_report import drift


CONTROL = '96211844163f00d2fedda95712c78aa6649980ba'
CONTROL_HARNESS = '99d89b8579cc93c818652505bd217d4c9079090a'


def marker(root, name):
    return (root / name).read_text().strip()


def routed_arm(root, group, label, revision):
    arm = load(root, group, label, revision)
    route = arm['route']
    n = arm['timing']['graph_calls']
    if (arm['binary'] != marker(root, f'ranked-{revision[:8]}.binary.sha256')
            or arm['config'].get('ranked_q_workers') != 2 or not arm['config'].get('ranked_auto_route')
            or not route or route['source'] != revision or route['policy'] != 'zero'
            or route['warmup'][0]['high'] != n):
        raise ValueError(f'G{group} {label}: source, cold start, final width or graph policy differs')
    for phase in ('measured', 'diagnostic'):
        point = route[phase]
        if (len(point['modes']) != n or point['modes'].count(0) != point['high']
                or point['modes'].count(1) != point['low'] or point['low'] + point['high'] != n
                or len(point['feedback_demand_per_call']) != n):
            raise ValueError(f'G{group} {label}: incomplete device route observations in {phase}')
    lifecycle = route.get('lifecycle') or {}
    if group in (2, 7) and not (lifecycle.get('misprediction_recovered') and lifecycle.get('invalid_identity_preserved')):
        raise ValueError(f'G{group} {label}: missing generation and prediction recovery checks')
    return arm


def choose_control(root, reference, group, fresh):
    case = root / f'g{group}'
    digest = marker(case, 'input.sha256')
    if not re.fullmatch('[a-f0-9]{64}', digest) or digest != marker(reference/f'g{group}', 'input.sha256'):
        raise ValueError('Current capture bytes differ from original')
    original = load(reference, group, 'baseline', RANKED_BASE)
    reusable = None
    directories = sorted(root.parent.glob(f'help-{CONTROL[:8]}-*'),
                         key=lambda d: d.stat().st_mtime, reverse=True)
    for previous in directories:
        try:
            if (marker(previous, 'harness.txt') != CONTROL_HARNESS or marker(previous, 'candidate.txt') != CONTROL
                    or marker(previous, 'ranked-'+CONTROL[:8]+'.binary.sha256') != marker(root, 'ranked-'+CONTROL[:8]+'.binary.sha256')
                    or marker(previous, 'host.txt') != marker(root, 'host.txt')
                    or marker(previous, 'physical_npu.txt') != marker(root, 'physical_npu.txt')
                    or marker(previous/f'g{group}', 'input.sha256') != digest or not (previous/f'g{group}/auto/pass').is_file()):
                continue
            arm = routed_arm(previous, group, 'auto', CONTROL)
            verify_pair(original, arm)
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            continue
        reusable = previous.resolve()
        break
    # Choose a measured local control on fallback. A saved native baseline can
    # still be used as context when the caller asks for a fresh routed control.
    for name, eligible in (('reuse-auto.txt', not fresh), ('serial-root.txt', group == 2)):
        path = case/name
        if reusable is not None and eligible:
            path.write_text(str(reusable)+'\n')
        else:
            path.unlink(missing_ok=True)


def fmt(value):
    return 'NA' if value is None else f'{value:.2f}'


def report(root, reference, group):
    case = root / f'g{group}'
    revision = marker(root, 'candidate.txt')
    if not re.fullmatch('[a-f0-9]{40}', revision) or revision == CONTROL or marker(root, 'fixed.txt') != CONTROL:
        raise ValueError('Wrong routed control or candidate')
    reused = case/'reuse-auto.txt'
    current = routed_arm(Path(reused.read_text().strip()), group, 'auto', CONTROL) if reused.exists() else routed_arm(root, group, 'current', CONTROL)
    new = routed_arm(root, group, 'auto', revision)
    if new['binary'] == current['binary']:
        raise ValueError('Device binaries are identical')
    controls = {'current_auto': current, 'original_archive': load(reference, group, 'baseline', RANKED_BASE)}
    serial = case/'serial-root.txt'
    if serial.exists():
        base = Path(serial.read_text().strip())
        for label in ('serial2', 'serial32'):
            if (base / f'g{group}' / label / 'pass').exists():
                arm = load(base, group, label)
                try:
                    verify_pair(new, arm)
                except ValueError:
                    continue
                controls[label+'_archive'] = arm
    comparisons = {label: comparison(arm, new) for label, arm in controls.items()}
    match = matched_samples(current, new)
    original_match = matched_samples(controls['original_archive'], new)
    n = new['timing']['graph_calls']
    desired = 'low' if group == 2 else 'high'
    arms = [current, new]
    counts = [a['route']['measured'] for a in arms]
    placement = 'archived_control' if reused.exists() else 'local_control'
    agreement = all(c[desired] == n for c in counts)
    means = [drift(a['timing']['samples_us']) for a in arms]
    gain = comparisons['current_auto']['gain_pct']
    span = headline(current)[1]
    matched = match['span_p50_saved_us']
    if not agreement:
        screen = 'route_mixed_or_changed'
    elif match['n']*2 < n:
        screen = 'insufficient_same_TopK'
    elif max(abs(v) for values in means for v in values) > .02*span:
        screen = 'drift_exceeds_2pct'
    elif min(gain) > 2 and matched > .02*span:
        screen = 'faster_gt2pct'
    elif max(abs(g) for g in gain) <= 2 and abs(matched) <= .02*span:
        screen = 'within_2pct'
    elif max(gain) < -2 and matched < -.02*span:
        screen = 'slower_gt2pct'
    else:
        screen = 'mixed_metrics'
    print(f'G{group} ROUTE current_low/high={counts[0]["low"]}/{counts[0]["high"]} '
          f'new_low/high={counts[1]["low"]}/{counts[1]["high"]} Get_p50_current/new='
          + '/'.join(f'{median(c["feedback_demand_per_call"]):g}' for c in counts)
          + f' current_source={placement} screen={screen}')
    for label, arm in controls.items():
        delta = comparisons[label]
        print(f'G{group} vs_{label} event_mean/span_p50_us={pair(headline(arm))}->{pair(headline(new))} '
              f'saved_us={pair(delta["saved_us"],True)} gain_pct={pair(delta["gain_pct"],True)}')
    print(f'G{group} SAME_TOPK={match["n"]}/{n} span_p50_saved_us={fmt(matched)} '
          f'original_pairs={original_match["n"]}/{n} original_span_p50_saved_us={fmt(original_match["span_p50_saved_us"])} '
          f'DRIFT last5-first5_span_mean/p50_us current={pair(means[0],True)} new={pair(means[1],True)}')
    timings = [a['timing']['stage_mean_us'] for a in arms]
    print(f'G{group} KERNEL_MEAN_SAVED_us '+' '.join(f'{s}={timings[0][s]-timings[1][s]:+.2f}' for s in timings[0]))
    phases = [a['detail']['stages'] for a in arms]
    keys = ('planning', 'final_get_post', 'final_get_cq_wait', 'routing_probe_aiv')
    print(f'G{group} AUX_P50_us current/new '+' '.join(
        f'{s}='+'/'.join(fmt(p[s]['p50_us']) for p in phases) for s in keys))
    high = []
    if all(a['route']['diagnostic']['high'] == n for a in arms):
        from ranked_index_pair_stats import describe_high
        for candidate, arm in zip((False, True), arms):
            with np.load(arm['directory']/'lean_ticks.npz', allow_pickle=False) as ticks:
                high.append(describe_high(ticks, candidate=candidate))
        for kind in ('p50_us', 'counts'):
            summaries = [h[kind] for h in high]
            print(f'G{group} HIGH_{kind.upper()} current/new '+' '.join(
                f'{s}='+'/'.join(fmt(p.get(s)) if kind == 'p50_us' else str(p.get(s, 'NA')) for p in summaries)
                for s in dict.fromkeys(k for st in summaries for k in st)))
    print(f'G{group} CHECK TopK/KV/ordered_Index/hot_valid=PASS')
    outcome = dict(revision=revision, control=CONTROL, current_source=placement,
                   route=counts, comparisons=comparisons, headline={k: headline(a) for k,a in {**controls, 'new':new}.items()},
                   drift=means, matched=match, original_matched=original_match, high_index=high, screen=screen)
    (case/'index_comparison.json').write_text(json.dumps(outcome, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--group', type=int, choices=GROUPS, required=True)
    parser.add_argument('--choose-control', action='store_true')
    parser.add_argument('--fresh-control', action='store_true')
    args = parser.parse_args()
    if args.choose_control:
        choose_control(args.root, args.reference, args.group, args.fresh_control)
    else:
        report(args.root, args.reference, args.group)
