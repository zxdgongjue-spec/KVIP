"""Read the B24 logical-ties archive. All times are microseconds."""
import json
import re
import sys
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import mean, median


REVISION = '24c619b9581aad43fd53c815928def26a1e19560'


def distribution(values):
    """Mean, median, linearly interpolated P95, max."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.95
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    p95 = ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return '/'.join(f'{v:.2f}' for v in (mean(ordered), median(ordered), p95, ordered[-1]))


def mean_median(values):
    return f'{mean(values):.2f}/{median(values):.2f}'


def read_json(path):
    return json.loads(path.read_bytes())


def query_counts(group, call, algorithm):
    requests = algorithm['requests']
    if len(requests) != 16:
        raise ValueError(f'G{group} call {call}: expected 16 requests')
    columns = [[0] * 6 for _ in range(4)]
    owners = []
    for row, request in enumerate(requests):
        qs = request['queries']
        if len(qs) != 4 or request['request'] != row:
            raise ValueError(f'G{group} call {call}: invalid request/Q layout')
        if (sum(q['final_posted'] for q in qs) != request['final']
                or sum(q['new_unique'] for q in qs) != request['selected_unique']):
            raise ValueError(f'G{group} call {call}: incomplete per-Q post or dedup counts')
        for q_id, q in enumerate(qs, 1):
            final, safe = q['final_posted'], q['final_safe']
            repeated, added = q['cross_q_duplicate'], q['new_unique']
            if (q['query'] != q_id or not 0 <= safe <= final or q['unique'] != 2048
                    or min(added, repeated) < 0 or added + repeated != q['unique']):
                raise ValueError(f'G{group} call {call} row {row} Q{q_id}: invalid query counts')
            column = columns[q_id - 1]
            for j, count in enumerate((final, safe, final-safe, int(final == 0), repeated, added)):
                column[j] += count
            owners.append((row, q_id, final))
    totals = algorithm['totals']
    if (sum(r['prefetch'] for r in requests) != totals['prefetch']
            or sum(c[0] for c in columns) != totals['final']
            or sum(c[5] for c in columns) != totals['selected_unique']):
        raise ValueError(f'G{group} call {call}: inconsistent aggregate workload')
    largest_row = max(requests, key=lambda r: r['final'])
    return columns, largest_row, max(owners, key=lambda item: item[2])


def graph_report(group, profile):
    """Return rows, graph timing and report lines. Call and Q are 1-based, row is 0-based."""
    timing = profile['timing']
    samples = profile['samples']
    if (len(samples) != 20 or profile['graph_calls'] != 20 or len(timing['samples_us']) != 20
            or timing['timing_protocol'] != 'one_profiled_graph' or timing['measured_graphs'] != 1):
        raise ValueError(f'G{group}: expected one measured graph with 20 calls')
    rows = []
    for i, sample in enumerate(samples):
        a = sample['accounting']
        phases = a['intervals_us']
        if set(phases) != {'li', 'resolve_update'} or len(phases['li']) != 4 or len(phases['resolve_update']) != 1:
            raise ValueError(f'G{group} call {i+1}: expected four LI kernels followed by one resolve_update')
        li = sorted(phases['li'])
        tail_start, end = phases['resolve_update'][0]
        start = li[0][0]
        intervals = li + [(tail_start, end)]
        if (a['invocation'] != i or sample['repeat'] != i
                or any(not (isfinite(x) and isfinite(y) and y > x) for x, y in intervals)
                or any(y > next_x + 1e-5 for (_, y), (next_x, _) in zip(intervals, intervals[1:]))):
            raise ValueError(f'G{group} call {i+1}: unordered or overlapping kernels')
        tail = end - tail_start
        li_us = sum(y-x for x, y in li)
        gap = tail_start - li[-1][1]
        span = end - start
        if max(abs(span-a['us']), abs(span-timing['samples_us'][i]),
               abs(tail-a['stages_us']['resolve_update']), abs(li_us-a['stages_us']['li'])) > 0.01:
            raise ValueError(f'G{group} call {i+1}: profile disagrees with timing')
        alg = sample['algorithm']
        columns, r, (owner_row, query, owner_count) = query_counts(group, i+1, alg)
        rows.append(dict(call=i+1, start=start, end=end, span=span, li=li_us, li4=li[-1][1]-li[-1][0],
                         launch_gap=gap, tail=tail, internal=span-li_us-tail,
                         prefetch=alg['totals']['prefetch'], final=alg['totals']['final'],
                         row_max=r['request'], row_n=r['final'], q_row=owner_row, q=query, q_n=owner_count,
                         columns=columns))
    event = timing['graph_event_us']
    total_span = sum(r['span'] for r in rows)
    inter_call = sum(b['start']-a['end'] for a, b in zip(rows, rows[1:]))
    if any(b['start'] < a['end'] - 1e-5 for a, b in zip(rows, rows[1:])):
        raise ValueError(f'G{group}: overlapping invocations')
    edges = event - (rows[-1]['end']-rows[0]['start'])
    if abs((event-total_span)/20 - timing['outside_invocations_mean_us']) > 0.01:
        raise ValueError(f'G{group}: inconsistent outside-invocation time')
    prefix = f'G{group}'
    tails = [r['tail'] for r in rows]
    lines = [
        f'{prefix} GRAPH mean_event_us={event/20:.2f} mean_invocation_span_us={total_span/20:.2f} '
        f'mean_LI_us={mean(r["li"] for r in rows):.2f} mean_tail_us={mean(tails):.2f} '
        f'mean_within_call_gaps_us={mean(r["internal"] for r in rows):.2f} '
        f'intercall_gap_total_div20_us={inter_call/20:.2f} event_minus_first_to_last_div20_us={edges/20:.2f}',
        f'{prefix} POST_LI mean/P50/P95/max_us launch_gap={distribution([r["launch_gap"] for r in rows])} '
        f'tail_kernel={distribution(tails)} LI4_end_to_tail_end={distribution([r["launch_gap"]+r["tail"] for r in rows])}',
        f'{prefix} TAIL_BY_CALL us=' + ','.join(f'{t:.2f}' for t in tails)]
    typical = min(rows, key=lambda r: abs(r['tail']-median(tails)))
    slowest = sorted(rows, key=lambda r: -r['tail'])[:3]
    for role, row in [('near_median', typical)] + [('slow', r) for r in slowest]:
        lines.append(f'{prefix} TAIL_CALL {role} call={row["call"]} span_us={row["span"]:.2f} '
                     f'li_us={row["li"]:.2f} launch_gap_us={row["launch_gap"]:.2f} tail_us={row["tail"]:.2f} '
                     f'prefetch/final={row["prefetch"]}/{row["final"]} '
                     f'largest_final_row=R{row["row_max"]}:{row["row_n"]} '
                     f'largest_final_owner=R{row["q_row"]}Q{row["q"]}:{row["q_n"]}')
    for q in range(4):
        per_call = [r['columns'][q] for r in rows]
        labels = ('final','safe','conflict','empty_rows','crossQ_dups','new_unique')
        entries = (f'{name}={distribution(values)}' for name, values in zip(labels,zip(*per_call)))
        lines.append(f'{prefix} Q{q+1} ' + ' '.join(entries))
    # Use the same published stage fields as li_restore_report after reconciling the intervals.
    stages = [sample['accounting']['stages_us'] for sample in samples]
    pairs = {'LI': mean_median([s['li'] for s in stages]), 'LI4': mean_median([r['li4'] for r in rows]),
             'graph': f'{event/20:.2f}/{median(timing["samples_us"]):.2f}',
             'tail': mean_median([s['resolve_update'] for s in stages])}
    return rows, pairs, lines


def plan_spread(group, receipts, rows):
    if len(receipts) != len(rows):
        raise ValueError(f'G{group}: missing per-call Index receipts')
    plans = defaultdict(list)
    row_states = {}
    for call, entries in zip(rows, receipts):
        if len(entries) != 16:
            raise ValueError(f'G{group}: missing row receipts')
        signature = []
        for row, (plan, state) in enumerate(entries):
            if not all(re.fullmatch(r'[0-9a-f]{64}', digest) for digest in (plan, state)):
                raise ValueError(f'G{group}: invalid Index digest')
            if row_states.setdefault((row, plan), state) != state:
                raise ValueError(f'G{group}: different final Index for identical row and ordered transfers')
            signature.append(plan)
        plans[tuple(signature)].append(call)
    repeated = [calls for calls in plans.values() if len(calls) > 1]
    line = (f'G{group} INDEX_PLANS distinct_full_plans={len(plans)}/{len(rows)} '
            f'calls_with_full_plan_peer={sum(map(len, repeated))}/{len(rows)}')
    if not repeated:
        return line + ' widest_same_plan_tail_spread_us=NA'
    bounds = [(min(calls, key=lambda c: c['tail']), max(calls, key=lambda c: c['tail'])) for calls in repeated]
    low, high = max(bounds, key=lambda pair: pair[1]['tail']-pair[0]['tail'])
    return (line + f' widest_same_plan_tail_spread_us={high["tail"]-low["tail"]:.2f} '
            f'fast_call={low["call"]}@{low["tail"]:.2f} slow_call={high["call"]}@{high["tail"]:.2f}')


def report(root):
    if ((root / 'physical_npu.txt').read_text().strip() != '4'
            or (root / 'logical-ties.txt').read_text().strip() != REVISION):
        raise ValueError('Expected physical NPU4 and logical-ties ' + REVISION)
    output = [f'ARCHIVE root={root} physical_NPU=4 native={REVISION}',
              'Q_COUNTS mean/P50/P95/max across 20 calls; each Q sums 16 rows. '
              'empty_rows=rows with zero final posts; crossQ_dups=unique IDs also selected by earlier Qs. '
              'Call/Q are 1-based; R is 0-based.']
    for group in (17, 4, 10, 7):
        case = root / f'g{group}'
        matches = list((case / 'logical-ties/results').glob('*/cases/diagnose/fused/diagnose/round-1/full_attribution.json'))
        if len(matches) != 1:
            raise ValueError(f'G{group}: found {len(matches)} full_attribution archives')
        path = matches[0]
        fused = path.parents[2]
        archive = read_json(path)
        out = read_json(fused / 'path_result.json')
        checks = read_json(fused / 'native_li_compare.json')
        strong = checks['comparisons']['strong_serial']
        if (out['production_revision'] != REVISION or out['correctness'] != 'PASS' or out['failures']
                or (checks['group'], checks['phase'], checks['measured_calls'], checks['queries_per_call'])
                    != (group,'measured',20,64)
                or (strong['matched_calls'], strong['matched_queries'], strong['total_queries']) != (20,1280,1280)
                or checks['serial_reference_stability']['stable_queries'] != 64):
            raise ValueError(f'G{group}: correctness or native revision receipt mismatch')
        rows, pairs, lines = graph_report(group, archive)
        receipts = out['query_owners']
        if len(receipts) != 20 or any(
                receipt != dict(owners=64, queues=64, processed=row['final'])
                for row, receipt in zip(rows,receipts)):
            raise ValueError(f'G{group}: final Get/CQ receipt count mismatch')
        archived = (case / 'li-summary.txt').read_text().splitlines()
        check = (f'G{group} CHECK TopK_set_vs_strong_serial=PASS calls=20/20 queries=1280/1280 '
                 'serial_repeat_queries=64/64 KV=PASS ordered_Index=PASS rows=320/320')
        timing_line = f'G{group} logical-ties ' + ' '.join(f'{k}_mean/P50_us={v}' for k, v in pairs.items())
        if archived.count(check) != 1 or archived.count(timing_line) != 1:
            raise ValueError(f'G{group}: li-summary.txt and attribution/correctness receipts disagree')
        output.extend([check, timing_line, *lines, plan_spread(group, out['layout_receipts'], rows)])
    return output


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python3 ' + Path(__file__).name + ' <archive-root>')
    try:
        print('\n'.join(report(Path(sys.argv[1]))))
    except (ValueError, KeyError, OSError, TypeError) as e:
        raise SystemExit(f'ARCHIVE_ERROR {e}')
