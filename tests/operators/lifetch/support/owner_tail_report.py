"""Summarize local AIV clock intervals in the Q4 owner-pack diagnostic replay."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


# owner_timer::Field, wire version 1. Intervals below always use one AIV.
FIELDS = dict(zip("""
tail_start block prefetch_wait prefetch_ready prefetch_end
resolve_start raw_wait raw_ready resolve_end raw_release conflict_wait
conflict_ready get_start get_end adapt_start adapt_end index_wait index_ready
index_start index_release index_end done finalize_start finalize_end
cleanup_end barrier_start barrier_end retire_issued count safe processed queue
prefetch_observed prefetch_polls prefetch_drain_cycles prefetch_return
prefetch_pending prefetch_completed sync_start sync_end
""".split(), range(16, 56)))


def span(a, b, optional=False):
    if optional and not a and not b:
        return 0
    if not 0 < a <= b:
        raise ValueError(f'incomplete/reversed timestamp pair {a}/{b}')
    return b-a


def owner(ticks, block, mhz):
    if len(ticks) != 848 or ticks[FIELDS['block']] != block+1:
        raise ValueError(f'bad trace ABI or owner mapping at block {block}')
    t = {k: ticks[i] for k, i in FIELDS.items()}
    q = block % 4
    n = t['count']
    if t['processed'] != n or not 0 <= t['safe'] <= n <= 2048:
        raise ValueError(f'bad descriptor or completion counts at block {block}')
    if t['queue'] != (2*(block//4)+1 if q == 0 else 32+3*(block//4)+q-1):
        raise ValueError(f'bad final queue at block {block}')

    def interval(first, last, optional=False):
        return span(t[first], t[last], optional)

    raw = interval('raw_wait', 'raw_ready', optional=(q == 0))
    resolve = interval('resolve_start', 'resolve_end')
    values = dict(
        owner_to_barrier=interval('tail_start', 'barrier_start'),
        barrier=interval('barrier_start', 'barrier_end'),
        resolver_handoff=raw, resolver_except_handoff=resolve-raw,
        resolver_launch=span(ticks[14], ticks[15], True) + span(ticks[56], ticks[57], True),
        resolver_save=span(ticks[60], ticks[61], True),
        conflict_wait=interval('conflict_wait', 'conflict_ready', optional=t['safe'] == n),
        desc_sync=interval('sync_start', 'sync_end', optional=not n),
        adapt=interval('adapt_start', 'adapt_end'),
        index_handoff=interval('index_wait', 'index_ready', optional=q == 0),
        index=interval('index_start', 'index_end'), index_apply=span(ticks[58], ticks[59], True),
        meta_commit=span(ticks[12], ticks[13], True),
        prefetch_flag=interval('prefetch_wait', 'prefetch_ready', optional=q != 3),
        prefetch_complete=interval('prefetch_ready', 'prefetch_end', optional=q != 3),
        row_finalize=interval('finalize_start', 'finalize_end', optional=q != 3),
    )
    values['prefetch_drain_call'] = t['prefetch_drain_cycles']
    values['row_postdone_cleanup'] = span(t['done'], t['barrier_start'])
    if values['resolver_except_handoff'] < 0:
        raise ValueError(f'raw wait outside resolver on block {block}')
    if q == 3:
        for f in range(6):
            values[f'join_flag{f+6}'] = span(ticks[2*f], ticks[2*f+1])
    # Doorbells and CQ polls have different units. The hardware may complete
    # during Index work. Only a poll actually executed after Index is exposed
    # to this owner. A CQ timestamp means first software observation.
    doors, cq, overflow = ticks[64:67]
    if not 0 <= doors <= 128 or not 0 <= cq <= 128:
        raise ValueError(f'invalid SQ/CQ capture count at block {block}')
    if n and not (doors and cq):
        raise ValueError(f'missing SQ or CQ while processing {n} descriptors')
    if not n and (doors or cq or t['get_start'] or t['get_end']):
        raise ValueError(f'Get activity on empty owner {block}')
    values.update(sq_active=ticks[67], cq_call=ticks[68])
    exposed = 0
    group_spans = []
    if n:
        interval('get_start', 'get_end')
        door_ticks = ticks[80:80+2*doors:2]
        if door_ticks != sorted(door_ticks) or not t['get_start'] <= door_ticks[0] <= door_ticks[-1] <= t['get_end']:
            raise ValueError(f'bad doorbell timeline at block {block}')
        door_by_group = dict(zip(ticks[81:81+2*doors:2], door_ticks))
        end_prev = t['get_start']
        for j in range(cq):
            begin, observed, group, polls = ticks[336+4*j:340+4*j]
            if not end_prev <= begin <= observed <= t['get_end']:
                raise ValueError(f'bad CQ poll at block {block}, CQ {j}')
            exposed += max(0, observed-max(begin, t['index_end']))
            if not overflow:
                if group not in door_by_group or door_by_group[group] > observed:
                    raise ValueError(f'unmatched doorbell/CQ group on block {block}: {group}')
                group_spans.append(observed-door_by_group[group])
            end_prev = observed
    if not overflow:
        values['cq_after_index'] = exposed
        if n:
            values['longest_group_to_observation'] = max(group_spans)
            values['to_first_doorbell'] = door_ticks[0]-t['tail_start']
            values['to_last_doorbell'] = door_ticks[-1]-t['tail_start']
    if any(v < 0 for v in values.values()):
        raise ValueError(f'negative duration on block {block}')
    return {k: v/mhz for k, v in values.items()}, bool(overflow)


def percent(values, fraction):
    a = sorted(values)
    i = (len(a)-1)*fraction
    lo = math.floor(i)
    return a[lo]+(a[math.ceil(i)]-a[lo])*(i-lo)


def summarize(values):
    if not values:
        return 'NA'
    return f'{median(values):.2f}/{percent(values, .95):.2f}'


def describe(path):
    data = json.loads(path.read_text())
    g = int(data['group'])
    mhz = data['ticks_per_us']
    if not math.isfinite(mhz) or mhz <= 0 or data['schema_version'] != 1:
        raise ValueError('uncalibrated counter or unrecognized trace schema')
    samples = data['samples']
    if not samples:
        raise ValueError('no checked trace samples')
    by_q = [defaultdict(list) for _ in range(4)]
    max_local = []
    truncated = 0
    longest = defaultdict(list)
    paired = []
    busy_owners = [0]*4
    capped_owners = [0]*4
    for s in samples:
        if len(s['ticks']) != 64:
            raise ValueError('expected all BS16 x Q4 trace rows')
        paired.append(s['event_on_us']-s['event_off_us'])
        if len(s['requests']) != 16 or s['ordered_queries'] != 64:
            raise ValueError('incomplete checked workload')
        measured = []
        for b, row in enumerate(s['ticks']):
            durations, overflow = owner(row, b, mhz)
            counts = s['requests'][b//4]['queries'][b%4]
            if counts['final_posted'] != row[FIELDS['count']] or counts['final_safe'] != row[FIELDS['safe']]:
                raise ValueError(f'workload does not match owner {b} in trace')
            busy_owners[b%4] += bool(counts['final_posted'])
            capped_owners[b%4] += overflow
            truncated += overflow
            for label, v in durations.items():
                by_q[b%4][label].append(v)
            measured.append(durations)
        slow = max(range(64), key=lambda b: measured[b]['owner_to_barrier'])
        for k, v in measured[slow].items():
            longest[k].append(v)
        max_local.append(f'r{slow//4}Q{slow%4+1}@{measured[slow]["owner_to_barrier"]:.2f}')
    print(f'G{g} trace_calls={len(samples)} owners_per_call=64 truncated_queues={truncated} '
          f'ticks_per_us={mhz:.2f} longest_local_prebarrier_us={",".join(max_local)}')
    print(f'G{g} diagnostic_pairs={len(paired)} graph_event_off/on_mean_us='
          f'{mean(s["event_off_us"] for s in samples):.2f}/{mean(s["event_on_us"] for s in samples):.2f} '
          f'all_calls_unadjusted_on_minus_off_mean/P50_us={mean(paired):+.2f}/{median(paired):+.2f}')
    match = data.get('plan_match')
    if match is not None:
        calls = len(samples)
        delta = match['matched_on_minus_off_p50_us']
        value = 'NA' if delta is None else f'{delta:+.2f}'
        print(f'G{g} plan_match archive_rows={match["archive_rows"]}/{16*calls} '
              f'archive_calls={match["archive_calls"]}/{calls} on_off_rows={match["on_off_rows"]}/{16*calls} '
              f'on_off_calls={match["on_off_calls"]}/{calls} matched_on_minus_off_P50_us={value}')
    columns = (
        'resolver_handoff', 'resolver_except_handoff', 'resolver_launch', 'resolver_save',
        'conflict_wait', 'desc_sync', 'sq_active', 'adapt', 'index_handoff', 'index',
        'index_apply', 'cq_call', 'cq_after_index', 'barrier',
    )
    print(f'G{g} owner_P50/P95_us ' + ' '.join(columns))
    for q, metrics in enumerate(by_q, 1):
        print(f'G{g} Q{q} ' + ' '.join(summarize(metrics[k]) for k in columns))
    metrics = by_q[3]
    extra = ['prefetch_flag','prefetch_complete','prefetch_drain_call','meta_commit','row_finalize',
             'join_flag8','join_flag9','join_flag10','join_flag11']
    print(f'G{g} Q4_P50/P95_us ' + ' '.join(f'{k}={summarize(metrics[k])}' for k in extra))
    for q, m in enumerate(by_q, 1):
        print(f'G{g} Q{q}_nonempty={busy_owners[q-1]} capped={capped_owners[q-1]} to_first/last_doorbell/max_group_doorbell_to_CQobs_P50_us=' +
              '/'.join(f'{median(m[k]):.2f}' if m[k] else 'NA'
                       for k in ('to_first_doorbell','to_last_doorbell','longest_group_to_observation')))
    key_parts = ['resolver_handoff','resolver_except_handoff','conflict_wait','sq_active','adapt',
                 'index_handoff','index','cq_after_index','row_finalize','barrier']
    print(f'G{g} longest_local_owner_P50_us ' +
          ' '.join(f'{k}={median(longest[k]):.2f}' if longest[k] else f'{k}=NA' for k in key_parts))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', nargs='+', type=Path)
    for trace in parser.parse_args().trace:
        describe(trace)
