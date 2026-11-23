"""Estimate how far an LI window could finish earlier on cores done with scoring."""
import argparse
import heapq
import json
import math
from collections import deque
from pathlib import Path
from statistics import mean, median


def replay(call, *, use_helpers, cost_us):
    """Keep the row's snapshot, original owner, and measured final service cost."""
    mhz = call['clock_mhz']
    rows = sorted(call['rows'], key=lambda r: r['row'])
    workers = call['workers']
    if not (math.isfinite(mhz) and mhz > 0 and rows and workers):
        raise ValueError('Invalid clock or empty worker/row records')
    owner_ids = {r['owner_aiv'] for r in rows}
    row_for_owner = {r['owner_aiv']: i for i, r in enumerate(rows)}
    hw = {w['aiv'] for w in workers}
    if (len(hw) != len(workers) or not owner_ids <= hw or len(owner_ids) != len(rows)
            or len({r['row'] for r in rows}) != len(rows)):
        raise ValueError('Duplicate rows or invalid owner mapping')
    if any(w['score_end_cycles'] <= 0 or w['safe_cycles'] < w['score_end_cycles'] for w in workers):
        raise ValueError('Worker reports availability before its last score')
    start, work, suffix, observed = [], [], [], []
    for r in rows:
        q = r['queries']
        if len(q) != 4 or [v['q'] for v in q] != list(range(4)):
            raise ValueError('Expected four queries in execution order')
        setup_end = q[0]['begin_cycles']
        if not 0 < r['freeze_begin_cycles'] <= r['freeze_end_cycles'] <= setup_end:
            raise ValueError('Invalid row snapshot or first query clock')
        previous = setup_end
        durations = []
        for v in q:
            if not previous <= v['begin_cycles'] < v['end_cycles']:
                raise ValueError('Query clock overlaps the preceding query')
            # Include the observed gap leading into this query. It covers any
            # per-query preparation that did not run inside AdvanceIncremental.
            durations.append(v['end_cycles'] - previous)
            previous = v['end_cycles']
        if not previous <= r['post_begin_cycles'] <= r['post_end_cycles']:
            raise ValueError('Invalid row service clocks')
        start.append(setup_end)
        work.append(durations)
        suffix.append(r['post_end_cycles'] - previous)
        observed.append(r['post_end_cycles'])

    cost = round(cost_us * mhz)
    if cost < 0:
        raise ValueError('Negative handoff cost')
    pending = [deque(range(4)) for _ in rows]
    helping = [False] * len(rows)
    last_finish = [0] * len(rows)
    ready = set()
    events = [(at, rows[i]['owner_aiv']) for i, at in enumerate(start)]
    helpers = {w['aiv'] for w in workers if w['aiv'] not in owner_ids}
    if use_helpers:
        events.extend((w['safe_cycles'], w['aiv']) for w in workers if w['aiv'] in helpers)
    heapq.heapify(events)
    assignments = 0

    def assign(core, row, query, time, external):
        finish = time + work[row][query] + (cost if external else 0)
        last_finish[row] = max(last_finish[row], finish)
        heapq.heappush(events, (finish, core))
        ready.remove(core)

    while events:
        now = events[0][0]
        while events and events[0][0] == now:
            ready.add(heapq.heappop(events)[1])
        # The original owner begins immediately, always taking the first
        # unclaimed query. Each spare takes a whole query from the other end.
        for core in sorted(ready & owner_ids):
            i = row_for_owner[core]
            if pending[i]:
                assign(core, i, pending[i].popleft(), now, False)
        for core in sorted(ready & helpers):
            eligible = [i for i in range(len(rows)) if start[i] <= now and pending[i]]
            if not eligible:
                break
            # Uses only queue depth and age visible at dispatch.
            i = min(eligible, key=lambda j: (-len(pending[j]), start[j], rows[j]['row']))
            assign(core, i, pending[i].pop(), now, True)
            helping[i] = True
            assignments += 1
    if any(pending):
        raise ValueError('Unassigned queries')
    row_ends = [last_finish[i] + suffix[i] + cost * helping[i] for i in range(len(rows))]
    end = max(row_ends)
    baseline = max(observed)
    free_limit = max(t + max(d) + tail for t, d, tail in zip(start, work, suffix))
    if not use_helpers and end != baseline:
        raise ValueError('Original schedule did not reproduce measured completion')
    if end < free_limit:
        raise ValueError('Schedule violated the indivisible-query lower bound')
    safe = {w['aiv']: w['safe_cycles'] for w in workers}
    cleanup = [safe[r['owner_aiv']] - observed[i] for i,r in enumerate(rows)]
    if min(cleanup) < 0:
        raise ValueError('Final-window owner reported safe before the final Post finished')
    # Preserve the owner cleanup interval and the availability time of each
    # non-owner. A helper finishes its last query no later than that row's Post.
    # The matrix cores and a real helper's polling/teardown are not measured.
    late_worker = max((safe[w] for w in helpers), default=0)
    aiv_before = max(safe.values())
    aiv_after = max(late_worker, max(t + c for t,c in zip(row_ends, cleanup)))
    ideal_aiv_end = max(late_worker, max(t + max(d) + s + c
            for t,d,s,c in zip(start, work, suffix, cleanup)))
    if not use_helpers and aiv_after != aiv_before:
        raise ValueError('Original schedule did not reproduce the AIV availability endpoint')
    last = max(range(len(rows)), key=observed.__getitem__)
    free_count = sum(w['aiv'] in helpers and w['safe_cycles'] <= start[last] for w in workers)
    return dict(gain_us=(baseline-end)/mhz, limit_us=(baseline-free_limit)/mhz,
                aiv_gain_us=(aiv_before-aiv_after)/mhz, aiv_limit_us=(aiv_before-ideal_aiv_end)/mhz,
                free_at_critical=free_count, helpers_used_for_queries=assignments)


def summarize(calls):
    if not calls:
        raise ValueError('No completed diagnostic calls')
    base = [replay(c, use_helpers=False, cost_us=0) for c in calls]
    summary = dict(calls=len(calls), ideal_limit_us=mean(r['limit_us'] for r in base),
                   aiv_limit_us=mean(r['aiv_limit_us'] for r in base),
                   available_at_last_row=mean(r['free_at_critical'] for r in base), cases={})
    for cost in (0, 1, 2, 4):
        out = [replay(c, use_helpers=True, cost_us=cost) for c in calls]
        gain = [r['gain_us'] for r in out]
        summary['cases'][cost] = dict(mean_us=mean(gain), p50_us=median(gain),
            aiv_mean_us=mean(r['aiv_gain_us'] for r in out), aiv_p50_us=median(r['aiv_gain_us'] for r in out),
            delegated=mean(r['helpers_used_for_queries'] for r in out))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    results = []
    for group in (10, 4, 17, 7, 2):
        source = args.root / f'g{group}/owner_availability.json'
        record = json.loads(source.read_text())
        if record['schema'] != 1 or record['group'] != group:
            raise ValueError(f'Unexpected schema or group in {source}')
        result = summarize(record['calls'])
        results.append(result)
        print(f"G{group} calls={result['calls']} last_row_spares_mean={result['available_at_last_row']:.2f} "
              f"ideal_mean_us={result['ideal_limit_us']:.2f} projected_saved_mean_us_handoff_0_1_2_4="
              + '/'.join(f"{v['mean_us']:.2f}" for v in result['cases'].values()))
        print(f"G{group} AIV_availability_ideal_mean_us={result['aiv_limit_us']:.2f} "
              "projected_saved_mean/P50_us_handoff_0_1_2_4="
              + ' '.join(f"{v['aiv_mean_us']:.2f}/{v['aiv_p50_us']:.2f}" for v in result['cases'].values()))
    print('FIVE_GROUP AIV_projected_saved_mean_us_handoff_0_1_2_4='
          + '/'.join(f"{mean(r['cases'][cost]['aiv_mean_us'] for r in results):.2f}" for cost in (0,1,2,4)))
    print('FIVE_GROUP last_window_projected_saved_mean_us_handoff_0_1_2_4='
          + '/'.join(f"{mean(r['cases'][cost]['mean_us'] for r in results):.2f}" for cost in (0,1,2,4)))


if __name__ == '__main__':
    main()
