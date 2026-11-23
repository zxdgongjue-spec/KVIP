"""Read the archived B24 five-case LI3 window events without executing a replay."""
import argparse
import json
import tarfile
from pathlib import Path
from statistics import mean, median, pstdev

MAIN = '609fa26fd715f9409bbe74e1b2d3685c58dfb0cf'


def correlation(x, y):
    sx, sy = pstdev(x), pstdev(y)
    if not sx or not sy:
        return None
    mx, my = mean(x), mean(y)
    return mean((a-mx)*(b-my) for a,b in zip(x, y))/(sx*sy)


def describe(events, records, clock, first_terminal):
    groups = [[e for e in events if e['epoch'] == epoch] for epoch in range(4)]
    if any(sorted(e['lane'] for e in group) != list(range(16)) for group in groups):
        raise ValueError('Incomplete service events')
    progress = {(b,q,epoch): covered for b,q,epoch,_,covered,_ in records}
    rank_counts = [sum(epoch == w for _,_,epoch in progress) for w in range(4)]
    if rank_counts != [0,64,64,64]:
        raise ValueError(f'Unexpected active window pattern {rank_counts}')
    by_epoch = []
    for group in groups:
        last = max(group, key=lambda e: e['end_cycles'])
        increment = [e['incremental_cycles']/clock for e in group]
        by_epoch.append({
            'post_count': sum(e['posted_count'] for e in group),
            'aggregate_increment_median_lane_us': median(increment),
            'last_lane_increment_us': last['incremental_cycles']/clock,
            'last_lane_predecessor_wait_us': last['predecessor_wait_cycles']/clock,
            'last_lane_service_us': (last['end_cycles']-last['begin_cycles'])/clock,
            'last_lane_scan_us': last['scan_cycles']/clock,
            'last_lane_candidate_us': last['candidate_cycles']/clock,
            'last_lane_post_us': last['post_cycles']/clock,
            'earliest_to_latest_completion_us': (last['end_cycles']-min(e['end_cycles'] for e in group))/clock,
            'last_completion_to_terminal_entry_us': (first_terminal-last['end_cycles'])/clock,
        })
    if by_epoch[0]['post_count']:
        raise ValueError('Unexpected first-window Post')
    final = sorted(groups[-1],key=lambda e:e['lane'])
    values = [[mean(progress[b,q,3]-progress[b,q,2] for q in range(1,5)),
               e['incremental_cycles']/clock,e['posted_count']] for b,e in enumerate(final)]
    halves = [[mean(v[index] for v in half) for index in range(3)] for half in (values[:8],values[8:])]
    delta, duration, _ = zip(*values)
    final_end = max(e['end_cycles'] for e in final)
    # Independent releases, instantaneous teams, and zero coordination overhead.
    ideal_64 = [(final_end-max(e['end_cycles']-fraction*e['incremental_cycles'] for e in final))/clock
                for fraction in (0.5,0.75)]
    # Advance the previous window on the same lane. Only its overlap with the
    # measured wait can advance final-window work; keep the other lanes in the max.
    previous = {e['lane']: e for e in groups[-2]}
    two_window = []
    for last_reduction in (0.5, 0.75):
        finishes = []
        for event in final:
            previous_saved = 0.5 * previous[event['lane']]['incremental_cycles']
            wait_saved = min(previous_saved, event['predecessor_wait_cycles'])
            finishes.append(event['end_cycles'] - last_reduction * event['incremental_cycles'] - wait_saved)
        two_window.append((final_end - max(finishes)) / clock)
    ready = [pct for _,_,epoch,pct,_,_ in records if epoch == 3]
    return by_epoch, halves, correlation(delta,duration), rank_counts, ideal_64, two_window, ready


def window_report(profile, details):
    invocations = profile['samples']
    if len(invocations) != 20 or len(details) != 20:
        raise ValueError('Expected 20 performance and 20 additional diagnostic invocations')
    li3 = []
    for sample in invocations:
        stages = sample['accounting']['intervals_us']['li']
        if len(stages) != 3:
            raise ValueError(f'Expected three LI launches, saw {len(stages)}')
        lo,hi = sorted(stages)[2]
        li3.append(hi-lo)
    windows, halves, correlations, counts, ideal, both, readiness = [], [], [], [], [], [], []
    for detail in details:
        head = detail['streaming_cycles'][0]
        timeline = head['terminal_timeline']
        first = min(s['start_cycles'] for c in timeline['cores'] for s in c['stages'] if 'start_cycles' in s)
        w,h,c,r,upper,two_window,ready = describe(head['prefetch_services'],detail['rank_progress'],timeline['clock_mhz'],first)
        windows.append(w); halves.append(h); correlations.append(c); counts.append(r)
        ideal.append(upper); both.append(two_window); readiness.extend(ready)
    return dict(
        timed_graph_mean_us=profile['timing']['event_mean_us'],
        timed_LI3_mean_us=mean(li3),
        last_window_min_median_max_ready_pct=[min(readiness), median(readiness), max(readiness)],
        last_window_queries_with_all_scores_reported_ready=len([x for x in readiness if x >= 100.0]),
        last_window_zero_helper_delay_end_advance_us={
            'half_aggregate_time_removed':mean(v[0] for v in ideal),
            'three_quarters_removed':mean(v[1] for v in ideal),
            'three_quarters_spare_above_graph_10pct':mean(v[1] for v in ideal)-0.1*profile['timing']['event_mean_us'],
        },
        two_window_ideal_final_event_advance_us={
            'penultimate_half_last_half': mean(v[0] for v in both),
            'penultimate_half_last_three_quarters': mean(v[1] for v in both),
            'assumptions': 'Fixed snapshot work, immediate helper availability, no new coordination cost. '
                           'Only the original final-event finish is modeled.',
        },
        separate_diagnostic_calls=len(details),
        diagnostic_mean_Q_with_rank_per_window=list(map(mean,zip(*counts))),
        diagnostic_windows=[{k:mean(v[epoch][k] for v in windows) for k in windows[0][epoch]} for epoch in range(4)],
        last_window_lane_halves=[
            dict(zip(('newly_covered_percentage_points','increment_us','posted_count'),
                     [mean(h[side][j] for h in halves) for j in range(3)])) for side in range(2)],
        last_window_mean_within_call_pearson_new_coverage_vs_increment=mean(x for x in correlations if x is not None),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    with tarfile.open(args.archive, 'r:*') as archive:
        def read(name):
            entry = archive.getmember(name)
            if not entry.isfile():
                raise ValueError(f'Not a file: {name}')
            return archive.extractfile(entry).read()
        if read('main.txt').decode().strip() != MAIN or read('physical_npu.txt').decode().strip() != '4':
            raise ValueError('Archive identity does not match the measured mainline on B24/NPU4')
        for group in (10,4,17,7,2):
            prefix = f'g{group}/mainline/results/'
            def load(name):
                matches = [p for p in archive.getnames() if p.startswith(prefix) and
                           p.endswith('/cases/diagnose/fused/diagnose/round-1/'+name)]
                path, = matches
                return json.loads(read(path))
            print(json.dumps(dict(group=group,**window_report(load('full_attribution.json'),
                                                             load('detail_samples.json')))))
