"""Read B24's instrumented per-query final TopK graph alongside its production archive."""
import argparse
from collections import Counter
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median

from joined_tail_capture import PASSES, kernel_duration, trace_enabled
from owner_tail_report import percent
from peer_delta import pair_placement, read
from query_owner_report import checked_layout_plan

PRODUCTION = '888db6b1603d22f4142bd2ccffe0518611856a31'


def graph_rows(path, group):
    folder, = (path / f'g{group}/logical-ties/results').glob('*/cases/diagnose/fused')
    return folder


def distribution(values):
    return '/'.join(f'{x:.2f}' for x in (min(values), median(values), percent(values, .95), max(values)))


def check_calls(data, archive_data, group):
    if (data['group'] != group or data['warmup'] != 4 or data['independent_states'] != 20
            or data['checked_calls'] != 80 or data['source_round'] != 1
            or not isfinite(data['ticks_per_us']) or data['ticks_per_us'] <= 0
            or [p['name'] for p in data['passes']] != list(PASSES)):
        raise ValueError('Expected four checked passes through the same calibrated twenty-state graph')
    plan_states = {}
    archived = [checked_layout_plan(row, plan_states, 'production') for row in archive_data['layout_receipts']]
    if len(archived) != 20:
        raise ValueError('Production receipts missing')
    calls = {}
    for p in data['passes']:
        name = p['name']
        if len(p['calls']) != 20 or not isfinite(p['graph_event_us']) or p['graph_event_us'] <= 0:
            raise ValueError(f'Invalid measured graph for {name}')
        calls[name] = []
        for i, call in enumerate(p['calls']):
            on = trace_enabled(name, i + 1)
            score, requests, receipts = call['score_check'], call['requests'], call['query_owners']
            if (call['call'] != i + 1 or call['traced'] != on or score['calls'] != 1 or score['queries'] != 64
                    or not 0 <= score['exact_queries'] <= 64 or len(requests) != 16
                    or receipts != dict(owners=64, queues=64, processed=sum(r['final'] for r in requests))):
                raise ValueError(f'Invalid slot, correctness or final CQ receipts in {name} at slot {i+1}')
            for b,r in enumerate(requests):
                if (r['request'] != b or len(r['queries']) != 4 or len(r['window_posted']) != 4
                        or r['prefetch'] != sum(r['window_posted'])
                        or r['final'] != sum(q['final_posted'] for q in r['queries'])
                        or any(q['query'] != j or not 0 <= q['final_safe'] <= q['final_posted'] <= 2048
                               for j, q in enumerate(r['queries'], 1))):
                    raise ValueError(f'Invalid per-query final descriptors in row {b}')
            t = kernel_duration(call['accounting'], i + 1)
            if not isclose(t, call['tail_us'], abs_tol=.015):
                raise ValueError('AIV capture assigned to wrong tail duration')
            plan = checked_layout_plan(call['layout_receipts'], plan_states, name)
            calls[name].append((t, plan, call))
    return calls, archived


def local_intervals(fields, ticks, block, query, mhz):
    """Partition one owner's time up to SyncAll. Never subtract clocks on different AIVs."""
    if len(ticks) != len(fields) or len(set(fields)) != len(fields):
        raise ValueError('Wrong joined-owner clock record width')
    t = dict(zip(fields, ticks))
    if (len(t) != 96 or t['miss_count'] != query['final_posted'] or t['safe_count'] != query['final_safe']
            or query['query'] != block % 4 + 1):
        raise ValueError(f'Clock record misaligned with query {block}')

    def interval(a, b, *, optional=False):
        first, last = t[a], t[b]
        if optional and not first and not last:
            return 0
        if not 0 < first <= last:
            raise ValueError(f'Block {block} invalid {a}/{b} clocks')
        return (last - first) / mhz

    def stage(a, b):
        return interval(a, b, optional=True)

    n = t['miss_count']
    if n and not (t['final_posts'] > 0 and t['final_cq_events'] > 0 and
                  0 < t['get_begin'] <= t['first_doorbell'] <= t['last_post'] <= t['callback_begin'] <=
                  t['callback_end'] <= t['completion_done'] <= t['get_end']):
        raise ValueError(f'Final SQ/CQ observations missing on active owner {block}')
    if not n and any(t[k] for k in ('get_begin', 'post_active_cycles', 'cq_drain_cycles', 'final_posts',
                                    'final_cq_events', 'first_doorbell')):
        raise ValueError(f'Unexpected final SQ/CQ activity on empty owner {block}')
    q4 = block % 4 == 3
    # The final selection also classifies hits. Publication includes the fence and flag.
    parts = dict(topk_classify=interval('entry', 'topk_end'),
                 class_publish=interval('topk_end', 'class_published'))
    parts['pref_gate'] = interval('class_published', 'prefetch_ready') if q4 else 0
    parts['pref_complete'] = interval('prefetch_ready', 'prefetch_end') if q4 else 0
    resolve = interval('resolve_begin', 'resolve_end')
    parts['alloc_wait'] = stage('raw_wait', 'raw_ready')
    parts['bitmap_wait'] = stage('bitmap_wait', 'bitmap_ready')
    parts['resolve_load'] = stage('resolver_restore_begin', 'resolver_restore_end')
    parts['resolve_save'] = stage('save_begin', 'save_end')
    parts['resolve_work'] = resolve - sum(parts[k] for k in ('alloc_wait', 'bitmap_wait', 'resolve_load', 'resolve_save'))
    parts['unsafe_wait'] = stage('conflict_wait', 'conflict_ready')
    parts['desc_sync'] = stage('descriptor_sync_begin', 'descriptor_sync_end')
    parts['post'] = t['post_active_cycles'] / mhz
    parts['final_cq'] = t['cq_drain_cycles'] / mhz
    parts['output'] = interval('adapt_begin', 'adapt_end')
    parts['class_wait'] = stage('classification_wait', 'classification_ready')
    parts['index_wait'] = stage('index_wait', 'index_ready')
    parts['index_load'] = stage('index_restore_begin', 'index_restore_end')
    parts['index_save'] = stage('index_save_begin', 'index_save_end')
    parts['index_apply'] = stage('apply_begin', 'apply_end')
    parts['old_read_wait'] = stage('q1_read_wait', 'q1_read_ready')
    parts['commit'] = stage('commit_begin', 'commit_end')
    index = interval('index_begin', 'index_release')
    parts['index_other'] = index - sum(parts[k] for k in ('index_load', 'index_save', 'index_apply', 'old_read_wait', 'commit'))
    close = interval('finalize_begin', 'finalize_end') if q4 else 0
    parts['row_join'] = sum(stage(f'final_flag{i}_wait', f'final_flag{i}_ready') for i in range(6, 12))
    parts['row_output_wait'] = stage('final_output_wait', 'final_output_ready')
    parts['row_close'] = close - parts['row_join'] - parts['row_output_wait']
    work = interval('entry', 'barrier_enter')
    parts['other'] = work - sum(parts.values())
    if min(parts.values()) < -.01:
        raise ValueError(f'Intervals overlap on owner {block}: {parts}')
    barrier = interval('barrier_enter', 'barrier_exit')
    after = interval('barrier_exit', 'retire_issued')
    return dict(parts=parts, work=work, barrier=barrier, retire=after,
                first_post=stage('entry', 'first_doorbell') if t['first_doorbell'] else None,
                buckets=t['bucket_words'], final=t['miss_count'],
                # Software work between return from Index/outputs and completion observation
                # includes polling and CQ processing. It is not a hardware completion latency.
                post_callback=(interval('callback_end', 'completion_done') if n else 0),
                resolve_bytes=t['resolver_restore_bytes'] + t['resolver_save_bytes'],
                index_bytes=t['index_restore_bytes'] + t['index_save_bytes'], q4_poll=t['q4_prefetch_polls'],
                q4_drain=t['q4_prefetch_wait_cycles']/mhz)


def compare_slots(prefix, data, calls, archived, prod_tail):
    archive_plans = set(archived)
    archive_rows = {item for plan in archive_plans for item in enumerate(plan)}
    mhz, fields = data['ticks_per_us'], data['trace_fields']
    exact_deltas, all_deltas, crossover, exact_archives, matched_rows = [], [], [], 0, 0
    active_owners, bins = [], {'lower': [], 'upper': [], 'outside': []}
    lo, split, hi = min(prod_tail), median(prod_tail), max(prod_tail)
    for i in range(20):
        active, inactive = ('odd_on', 'even_on') if i % 2 == 0 else ('even_on', 'odd_on')
        tail, plan, c = calls[active][i]
        off_tail, off_plan, off_c = calls[inactive][i]
        if len(c['ticks']) != 64:
            raise ValueError('Expected 64 clock records, one per (row, query)')
        matched_rows += sum(pair in archive_rows for pair in enumerate(plan))
        exact_archives += plan in archive_plans
        # A matching layout plan alone does not fix every query's traffic or selected IDs.
        pair_ok = plan == off_plan and c['requests'] == off_c['requests'] and c['topk_set_digest'] == off_c['topk_set_digest']
        if pair_ok:
            exact_deltas.append(tail - off_tail)
        all_deltas.append(tail - off_tail)
        crossover.append(calls['off_after'][i][0] - calls['off_before'][i][0])
        owners = [local_intervals(fields, ticks, b, c['requests'][b//4]['queries'][b%4], mhz)
                  for b, ticks in enumerate(c['ticks'])]
        # The shortest same-core SyncAll wait bounds arrival lag. It may include release overhead.
        near = min(range(64), key=lambda b: owners[b]['barrier'])
        last = owners[near]
        longest = max(range(64), key=lambda b: owners[b]['work'])
        bucket = 'outside' if not lo <= tail <= hi else ('lower' if tail <= split else 'upper')
        bins[bucket].append(last)
        active_owners.extend(owners)
        r = near//4
        row = owners[4*r:4*r+4]
        first = 'NA' if last['first_post'] is None else f'{last["first_post"]:.2f}'

        def qtimes(key):
            return '/'.join(f'{x["parts"][key]:.2f}' for x in row)
        print(f'{prefix} SLOT={i+1} tail_on/off_us={tail:.2f}/{off_tail:.2f} paired_work={int(pair_ok)} '
              f'archive_plan={int(plan in archive_plans)} archive_band={bucket} near_last=R{r}Q{near%4+1} '
              f'lag_bound_us={last["barrier"]:.2f} final={last["final"]} local_us={last["work"]:.2f} '
              f'longest_local=R{longest//4}Q{longest%4+1} barrier64_P50_us={median(x["barrier"] for x in owners):.2f} '
              f'all64_local_P50/max_us={median(x["work"] for x in owners):.2f}/{max(x["work"] for x in owners):.2f} '
              f'row_Q1_to_Q4_alloc_wait_us={qtimes("alloc_wait")} row_Q1_to_Q4_index_wait_us={qtimes("index_wait")} '
              f'first_post_since_entry_us={first} callback_end_to_final_observation_us={last["post_callback"]:.2f} '
              f'row_Q4_pref_drain_us={row[3]["q4_drain"]:.2f} row_Q4_pref_polls={row[3]["q4_poll"]} '
              f'near_last_resolve/index_handoff_KiB={last["resolve_bytes"]/1024:.2f}/{last["index_bytes"]/1024:.2f}')
    print(f'{prefix} CHECKED_DIAGNOSTIC calls=80/80 queries=5120/5120 rows=1280/1280 '
          f'traced_calls=20 clocks_per_call=64 all_local_clock_differences=1')
    print(f'{prefix} TRACE_EFFECT production_tail_min/max/P50_us={lo:.2f}/{hi:.2f}/{split:.2f} '
          f'archive_event_plans={exact_archives}/20 archive_event_rows={matched_rows}/320 '
          f'exact_on_off_work={len(exact_deltas)}/20 matched_on_minus_off_min/P50/P95/max_us='
          f'{distribution(exact_deltas) if exact_deltas else "NA"} all_on_minus_off_min/P50/P95/max_us='
          f'{distribution(all_deltas)} off_after_minus_before_min/P50/P95/max_us={distribution(crossover)}')
    for name, selected in bins.items():
        if not selected:
            print(f'{prefix} NEAR_LAST band={name} n=0')
            continue
        # Each column is the mean on the same selected owners. Columns add to their mean local duration.
        parts = ' '.join(f'{k}={mean(v["parts"][k] for v in selected):.2f}' for k in selected[0]['parts'])
        print(f'{prefix} NEAR_LAST band={name} n={len(selected)} local_mean_us={mean(v["work"] for v in selected):.2f} '
              f'lag_bound_P50_us={median(v["barrier"] for v in selected):.2f} exclusive_mean_us {parts}')
    print(f'{prefix} ALL_OWNER counted_hash_buckets={dict(sorted(Counter(v["buckets"] for v in active_owners).items()))} '
          f'resolve/index_handoff_KiB_per_call_mean={sum(v["resolve_bytes"] for v in active_owners)/(20*1024):.2f}/'
          f'{sum(v["index_bytes"] for v in active_owners)/(20*1024):.2f} '
          f'post_barrier_to_retire_issue_P50/max_us={median(v["retire"] for v in active_owners):.2f}/'
          f'{max(v["retire"] for v in active_owners):.2f}')


def report(root, group, production):
    prefix = f'G{group}'
    if ((production / 'logical-ties.txt').read_text().strip() != PRODUCTION or
            any((r / 'physical_npu.txt').read_text().strip() != '4' for r in (root, production))):
        raise ValueError('Expected the B24 NPU4 production candidate and its diagnostic')
    prod_folder, diag_folder = (graph_rows(p, group) for p in (production, root))
    data = read(diag_folder / 'joined_tail.json')
    sample = read(prod_folder / 'path_result.json')
    diag_result = read(diag_folder / 'path_result.json')
    if diag_result['correctness'] != 'PASS' or diag_result['failures']:
        raise ValueError('Diagnostic graph did not pass its correctness checks')
    prod_inputs, diag_inputs = (read(f.parent / 'replay_input.json') for f in (prod_folder, diag_folder))
    if (sample['correctness'] != 'PASS' or sample['failures'] or sample['production_revision'] != PRODUCTION
            or data['native_revision'] != (root/'logical-ties.txt').read_text().strip()
            or data['native_revision'] == PRODUCTION or data['input_paths'] != prod_inputs['source_paths']
            or diag_inputs != prod_inputs):
        raise ValueError('Different production revision, input capture, or failed run')
    prod_samples = read(prod_folder/'diagnose/round-1/full_attribution.json')['samples']
    if len(prod_samples) != 20:
        raise ValueError('Expected twenty production intervals')
    prod_tail = [kernel_duration(x['accounting'], i + 1) for i,x in enumerate(prod_samples)]
    placement = pair_placement(*(read(p/f'g{group}/logical-ties/thp.json') for p in (production, root)), local_baseline=True)
    calls, archived = check_calls(data, sample, group)
    oracle_digest = read(prod_folder / 'score_topk_check.json')['serial_set_digest']
    if any(call['score_check']['serial_set_digest'] != oracle_digest for p in data['passes'] for call in p['calls']):
        raise ValueError('Diagnostic and production used different strong serial TopK references')
    binary = (root / f"ranked-{data['native_revision'][:8]}.binary.sha256").read_text().strip()
    if diag_result['production_revision'] != data['native_revision'] or diag_result['binary_sha256'] != binary:
        raise ValueError('Incorrect diagnostic binary')
    print(f'{prefix} PRODUCTION tail_min/P50/P95/max_us={distribution(prod_tail)} placement={placement}')
    for name in PASSES:
        print(f'{prefix} DIAGNOSTIC {name} tail_min/P50/P95/max_us={distribution([x[0] for x in calls[name]])}')
    compare_slots(prefix, data, calls, archived, prod_tail)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root', type=Path)
    cli.add_argument('group', type=int, choices=(2, 7))
    cli.add_argument('production', type=Path)
    args = cli.parse_args()
    report(args.root, args.group, args.production)
