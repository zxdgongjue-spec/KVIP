# SPDX-License-Identifier: Apache-2.0
"""Account for the last-finishing terminal worker on a separate diagnostic replay."""
import json
from collections import Counter
from math import isclose, isfinite
from statistics import mean, median

# These are the contiguous timestamp intervals exported by both production
# versions' diagnostics.decode_terminal_timeline. All four GET/CQ batches can
# occur in final_CQ_join; this is elapsed time through the final CQ completion.
PHASES = ('initialize', 'TopK', 'publish.TopK', 'deduplicate_and_slots', 'output',
          'WQE_and_publish', 'release_scratch', 'diagnostic_snapshot', 'index_update_and_join',
          'final_CQ_join', 'prefetch_CQ_join', 'retire')
STAGES = ('start_delay', *PHASES, 'control_gap')


def close(left, right):
    return isclose(left, right, rel_tol=1e-9, abs_tol=1e-6)


def inspect(timeline):
    cores = sorted(timeline['cores'], key=lambda core: core['block'])
    if [c['block'] for c in cores] != list(range(64)):
        raise ValueError('Expected every request/Q exactly once in BS16/Q4 detail')
    mhz = timeline['clock_mhz']
    if not isfinite(mhz) or mhz <= 0:
        raise ValueError('Missing terminal device clock calibration')
    origin = min(c['stages'][0]['start_cycles'] for c in cores)
    if origin <= 0:
        raise ValueError('Missing terminal start timestamp')
    for c in cores:
        intervals = c['stages']
        if tuple(p['name'] for p in intervals if p['name'] != 'control_gap') != PHASES:
            raise ValueError('Terminal timestamp phases are missing, duplicated or out of order')
        cursor = intervals[0]['start_cycles']
        if not close(c['start_us'], (cursor - origin)/mhz):
            raise ValueError('Incorrect terminal worker start')
        for p in intervals:
            stop = p['end_cycles']
            if (p['start_cycles'] != cursor or type(stop) is not int or stop < cursor or
                    not close(p['start_us'], (cursor-origin)/mhz) or not close(p['end_us'], (stop-origin)/mhz) or
                    not close(p['duration_us'], (stop-cursor)/mhz)):
                raise ValueError('Terminal phase time does not reconcile to the device ticks')
            cursor = stop
        if not close(c['end_us'], (cursor-origin)/mhz) or not close(c['duration_us'], c['end_us']-c['start_us']):
            raise ValueError('Incorrect terminal worker duration')

    # Tie-break by lowest block ID. Earlier-finishing workers overlap with
    # this worker and must not be added to a terminal-phase breakdown.
    last = max(cores, key=lambda c: c['stages'][-1]['end_cycles'])
    if last['block'] != timeline['last_block'] or not close(last['end_us'], timeline['observed_span_us']):
        raise ValueError('Terminal extent and last-finishing worker disagree')
    stages = dict.fromkeys(STAGES, 0.)
    stages['start_delay'] = last['start_us']
    for p in last['stages']:
        stages[p['name']] += p['duration_us']
    if not close(sum(stages.values()), last['end_us']):
        raise ValueError('Terminal phases do not sum to the observed extent')
    ties = sum(c['stages'][-1]['end_cycles'] == last['stages'][-1]['end_cycles'] for c in cores)
    return dict(block=last['block'], ties=ties, clock_mhz=mhz, span=last['end_us'], stages=stages,
                finish_spread=last['end_us']-min(c['end_us'] for c in cores))


def terminal_detail(fused_folder):
    """Read existing production detail. Missing diagnostics are explicitly unavailable."""
    path = fused_folder / 'diagnose/round-1/detail_samples.json'
    if not path.exists():
        return dict(status='UNAVAILABLE', reason='missing_detail_samples', path=str(path))
    samples = json.loads(path.read_text())
    if ([s['round'] for s in samples] != list(range(20)) or any(s['source_round'] != 1 for s in samples)):
        raise ValueError(f'{path}: expected 20 independent diagnostic calls for capture round 1')
    rows = []
    for sample in samples:
        timelines = [r['terminal_timeline'] for r in sample['streaming_cycles'] if 'terminal_timeline' in r]
        if len(timelines) != 1:
            raise ValueError(f'{path}: expected one complete terminal timeline per diagnostic call')
        rows.append(inspect(timelines[0]))
    if len({r['clock_mhz'] for r in rows}) != 1:
        raise ValueError(f'{path}: device calibration changed inside one diagnostic graph')
    blocks = Counter(r['block'] for r in rows)
    stages = {name: [r['stages'][name] for r in rows] for name in STAGES}
    return dict(status='OK', source=str(path), calls=20, clock_mhz=rows[0]['clock_mhz'],
                phases_us={name: [mean(vals), median(vals)] for name, vals in stages.items()},
                span_mean_us=mean(r['span'] for r in rows), span_p50_us=median(r['span'] for r in rows),
                tie_calls=sum(r['ties'] > 1 for r in rows), last_blocks=dict(sorted(blocks.items())),
                finish_spread_mean_us=mean(r['finish_spread'] for r in rows))


def print_terminal_detail(group, arms):
    """Inputs are ordered reference then candidate. Positive deltas denote extra time."""
    for label, data in arms.items():
        if data['status'] != 'OK':
            print(f'G{group} {label}_terminal_detail UNAVAILABLE reason={data["reason"]}')
            continue
        print(f'G{group} {label}_terminal_detail calls=20 workers_per_call=64 '
              f'clock_MHz={data["clock_mhz"]:.2f} span_mean/P50_us={data["span_mean_us"]:.2f}/{data["span_p50_us"]:.2f} '
              f'worker_finish_spread_mean_us={data["finish_spread_mean_us"]:.2f} ties={data["tie_calls"]}/20')
        print(f'G{group} {label}_last_worker_phase_mean_us ' +
              ' '.join(f'{k}={v[0]:.2f}' for k, v in data['phases_us'].items()))
    if len(arms) == 2 and all(r['status'] == 'OK' for r in arms.values()):
        (first, old), (second, new) = arms.items()
        print(f'G{group} {second}_minus_{first}_last_worker_phase_mean_us ' +
              ' '.join(f'{k}={new["phases_us"][k][0]-old["phases_us"][k][0]:+.2f}' for k in STAGES))
