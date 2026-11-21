# SPDX-License-Identifier: Apache-2.0
"""Associate each live per-owner trace with its own kernel in a twenty-state graph."""
import argparse
import hashlib
from collections import Counter
from math import isfinite
from pathlib import Path
from statistics import mean, median

import numpy as np

from owner_pack_tail import check_serial, reference, workload
from owner_tail_report import FIELDS, owner, percent
from peer_delta import pair_placement, read
from query_owner_report import checked_layout_plan
from reporting import atomic_json


PASSES = ('off_before', 'odd_on', 'even_on', 'off_after')
CALLS = 20
SHAPE = (CALLS, 64, 848)


def traced(name, call):
    return (name == 'odd_on' and call % 2 == 1) or (name == 'even_on' and call % 2 == 0)


def counter_digest(ticks):
    return hashlib.sha256(np.ascontiguousarray(ticks, dtype='<i8').view('u1')).hexdigest()


def check_kernel(accounting, call):
    phases = accounting['intervals_us']
    if (accounting['invocation'] != call-1 or set(phases) != {'li', 'resolve_update'} or
            len(phases['li']) != 4 or len(phases['resolve_update']) != 1):
        raise ValueError(f'Call {call}: profiler kernels do not match the captured LI/tail sequence')
    ordered = sorted(phases['li']) + phases['resolve_update']
    if (any(not (isfinite(a) and isfinite(b) and b > a) for a,b in ordered) or
            any(b > c+1e-5 for (_,b),(c,_) in zip(ordered,ordered[1:]))):
        raise ValueError(f'Call {call}: invalid or overlapping profiler kernels')
    begin, end = ordered[-1]
    if abs(end-begin-accounting['stages_us']['resolve_update']) > 0.01:
        raise ValueError(f'Call {call}: inconsistent tail-kernel timing')
    return end-begin


def capture(states, inputs, config, directory, ticks_per_us, profile_pass, expected):
    if len(states) != CALLS or not isfinite(ticks_per_us) or ticks_per_us <= 0:
        raise ValueError('Expected 20 independent graph states and a calibrated device clock')
    raw = np.empty(SHAPE, dtype='<i8')
    passes = []
    for name in PASSES:
        for call, state in enumerate(states, 1):
            state.reset()
            state.forward.set_query_owner_capture(traced(name, call))
            # Clear disabled rows as well so the readback detects graph/slot aliasing.
            state.forward.query_owner_trace.zero_()
        profile = profile_pass(directory / 'owner-graph-profiler' / name)
        timing = profile['timing']
        if (len(profile['samples']) != CALLS or timing['graph_calls'] != CALLS or
                timing['measured_graphs'] != 1 or timing['timing_protocol'] != 'one_profiled_graph' or
                not (isfinite(timing['graph_event_us']) and timing['graph_event_us'] > 0)):
            raise ValueError(f'{name}: incomplete twenty-state graph profile')
        calls = []
        for call, (state, accounting) in enumerate(zip(states, profile['samples']), 1):
            check_kernel(accounting, call)
            enabled = traced(name, call)
            ticks = state.forward.query_owner_trace.cpu().numpy()
            if ticks.shape != SHAPE[1:] or (enabled and np.any(ticks[:, FIELDS['tail_start']] <= 0)):
                raise ValueError(f'{name}/{call}: missing AIV trace')
            if enabled:
                raw[call-1] = ticks
            elif np.any(ticks):
                raise ValueError(f'{name}/{call}: disabled graph slot produced AIV trace')
            check_serial(state, expected)
            state.validate(inputs)
            calls.append(dict(call=call, traced=enabled, accounting=accounting,
                              topk_set_digest=state.topk_set_digest, query_owners=state.query_owners,
                              layout_receipts=state.layout_receipts, requests=workload(state)))
        passes.append(dict(name=name, graph_event_us=timing['graph_event_us'], calls=calls))
    npz = directory/'owner_graph_ticks.npz'
    tmp = directory/'owner_graph_ticks.tmp'
    with tmp.open('wb') as target:
        np.savez_compressed(target, ticks=raw)
    tmp.replace(npz)
    atomic_json(directory/'owner_graph.json', dict(schema_version=1, group=config.group,
                native_revision=config.production_revision, source_round=inputs.source_round,
                input_paths=list(map(str, inputs.source_paths)), ticks_per_us=ticks_per_us,
                ticks_sha256=counter_digest(raw), checked_calls=CALLS*len(PASSES), passes=passes))
    return CALLS*len(PASSES)


def archive_bands(times):
    """Split G7 at its largest observed gap; retain the observed ranges, including their endpoints."""
    values = sorted(times)
    if len(values) != CALLS or min(values) <= 0 or not all(map(isfinite, values)):
        raise ValueError('Invalid archived tail samples')
    index = max(range(1, len(values)), key=lambda i: values[i]-values[i-1])
    return values[0], values[index-1], values[index], values[-1]


def band(tail, bounds):
    if bounds[0] <= tail <= bounds[1]:
        return 'lower'
    if bounds[2] <= tail <= bounds[3]:
        return 'upper'
    return 'outside'


def distribution(values):
    return '/'.join(f'{v:.2f}' for v in (min(values), median(values), percent(values, .95), max(values)))


def near_last(call, ticks, mhz):
    """All durations are computed on one AIV. The barrier interval bounds arrival lag."""
    durations = []
    truncated = 0
    requests = call['requests']
    if len(requests) != 16:
        raise ValueError('Incomplete request inventory')
    for block, raw in enumerate(ticks):
        b, q = divmod(block, 4)
        request = requests[b]
        if request['request'] != b or len(request['queries']) != 4:
            raise ValueError('Request/Q mapping differs from the graph')
        query = request['queries'][q]
        if (query['query'] != q+1 or query['final_posted'] != raw[FIELDS['count']] or
                query['final_safe'] != raw[FIELDS['safe']]):
            raise ValueError(f'Profiler call and AIV trace describe different owner {block}')
        if q == 3 and raw[FIELDS['prefetch_pending']] != raw[FIELDS['prefetch_completed']]:
            raise ValueError(f'Incomplete prefetch CQ on request {b}')
        metrics, capped = owner(raw.tolist(), block, mhz)
        truncated += int(capped)
        durations.append(metrics)
    block = min(range(64), key=lambda b: durations[b]['barrier'])
    b, q = divmod(block, 4)
    m = durations[block]
    work = requests[b]
    pref = durations[4*b+3]
    m4raw = ticks[4*b+3]
    prebarrier = [d['owner_to_barrier'] for d in durations]
    barriers = [d['barrier'] for d in durations]
    cq = m.get('cq_after_index')
    rank = lambda key: '/'.join(f'{durations[4*b+i][key]:.2f}' for i in range(4))
    return dict(block=block, owner=f'R{b}Q{q+1}', prebarrier=m['owner_to_barrier'], lag=m['barrier'],
                local_P50=median(prebarrier), local_max=max(prebarrier),
                all_barrier_P50=median(barriers), all_barrier_P95=percent(barriers,.95),
                q_barrier='/'.join(f'{median(barriers[i::4]):.2f}' for i in range(4)),
                row_windows='/'.join(map(str, work['window_posted'])),
                final=work['queries'][q]['final_posted'], pre_drain=m['prefetch_drain_call'],
                row_q4_flag=pref['prefetch_flag'], row_q4_complete=pref['prefetch_complete'],
                row_q4_drain=pref['prefetch_drain_call'], row_q4_polls=int(m4raw[FIELDS['prefetch_polls']]),
                row_q4_pending=int(m4raw[FIELDS['prefetch_pending']]),
                allocator_wait=m['resolver_handoff'], resolve=m['resolver_except_handoff'],
                post=m['sq_active'], adapt=m['adapt'], index_wait=m['index_handoff'], index=m['index'],
                exposed_final_cq=cq, row_index=rank('index'), row_index_wait=rank('index_handoff'),
                row_finalize=m['row_finalize'], capped=truncated)


def summarize(data, ticks, archived_calls):
    if (data['schema_version'] != 1 or data['checked_calls'] != 80 or
            [p['name'] for p in data['passes']] != list(PASSES) or
            ticks.shape != SHAPE or ticks.dtype != np.int64 or counter_digest(ticks) != data['ticks_sha256'] or
            not (isfinite(data['ticks_per_us']) and data['ticks_per_us'] > 0)):
        raise ValueError('Incomplete graph trace or mismatched raw tick payload')
    g = data['group']
    prefix = f'G{g}'
    bounds = archive_bands([c['tail_us'] for c in archived_calls])
    reference_states = {}
    archived_plans = {checked_layout_plan(c['layout_receipts'], reference_states, 'archive') for c in archived_calls}
    stages = {}
    lines = [f'{prefix} ARCHIVE_TAIL_BANDS_US lower={bounds[0]:.2f}..{bounds[1]:.2f} '
             f'upper={bounds[2]:.2f}..{bounds[3]:.2f} grouping=largest_observed_archive_gap']
    for p in data['passes']:
        name, cells = p['name'], p['calls']
        if len(cells) != CALLS:
            raise ValueError('Incomplete pass')
        stages[name] = []
        for i, c in enumerate(cells, 1):
            if c['call'] != i or c['traced'] != traced(name, i):
                raise ValueError(f'{name}/{i}: trace attached to wrong call')
            t = check_kernel(c['accounting'], i)
            plan = checked_layout_plan(c['layout_receipts'], reference_states, name)
            if (len(c['requests']) != 16 or c['query_owners'] !=
                    dict(owners=64, queues=64, processed=sum(r['final'] for r in c['requests']))):
                raise ValueError(f'{name}/{i}: final CQ receipts disagree')
            totals = c['accounting'].get('work_totals')
            if totals is not None and (sum(r['final'] for r in c['requests']) != totals['final'] or
                                       sum(r['prefetch'] for r in c['requests']) != totals['prefetch']):
                raise ValueError(f'{name}/{i}: workload snapshots and physical posts disagree')
            stages[name].append((t, plan, c))
        values = [v[0] for v in stages[name]]
        counts = Counter(band(v,bounds) for v in values)
        lines.append(f'{prefix} GRAPH_PASS {name} graph_event_div20_us={p["graph_event_us"]/20:.2f} '
                     f'tail_min/P50/P95/max_us={distribution(values)} '
                     f'lower/upper/outside={counts["lower"]}/{counts["upper"]}/{counts["outside"]}')
    by_band = {key: [] for key in ('lower','upper','outside')}
    delta = []
    unpaired_differences = []
    off_drift = [end[0]-start[0] for start, end in zip(stages['off_before'], stages['off_after'])]
    alloff = [stages['off_before'], stages['off_after']]
    exact_brackets = []
    archive_matches = peer_matches = 0
    for i in range(CALLS):
        on, off = ('odd_on','even_on') if i % 2 == 0 else ('even_on','odd_on')
        tail, plan, c = stages[on][i]
        off_tail, off_plan, _ = stages[off][i]
        unpaired_differences.append(tail-off_tail)
        if plan == off_plan:
            delta.append(tail-off_tail)
        brackets = [p[i][0] for p in alloff if p[i][1] == plan]
        if len(brackets) == 2:
            exact_brackets.append((tail-max(brackets), tail-min(brackets)))
        with_archive = plan in archived_plans
        archive_matches += int(with_archive)
        peer_matches += int(plan == off_plan)
        phase = band(tail,bounds)
        m = near_last(c, ticks[i], data['ticks_per_us'])
        by_band[phase].append(m)
        cq = m['exposed_final_cq']
        cq_text = 'NA' if cq is None else f'{cq:.2f}'
        lines.append(f'{prefix} GRAPH_SLOT {i+1} trace={on} band={phase} tail_us={tail:.2f} '
                     f'same_position_mixed_off_us={off_tail:.2f} on_off_plan_match={int(plan==off_plan)} '
                     f'archive_plan_match={int(with_archive)} near_last={m["owner"]} lag_bound_us={m["lag"]:.2f} '
                     f'local_prebarrier_us={m["prebarrier"]:.2f} prebarrier64_P50/max={m["local_P50"]:.2f}/{m["local_max"]:.2f} '
                     f'barrier64_P50/P95={m["all_barrier_P50"]:.2f}/{m["all_barrier_P95"]:.2f} '
                     f'Q1to4_barrier_P50={m["q_barrier"]} row_window_posts={m["row_windows"]} owner_final={m["final"]} '
                     f'row_Q4_pref_flag/complete/drain_us={m["row_q4_flag"]:.2f}/{m["row_q4_complete"]:.2f}/{m["row_q4_drain"]:.2f} '
                     f'row_Q4_pref_pending/polls={m["row_q4_pending"]}/{m["row_q4_polls"]} '
                     f'resolve/alloc_wait/post/adapt_us={m["resolve"]:.2f}/{m["allocator_wait"]:.2f}/{m["post"]:.2f}/{m["adapt"]:.2f} '
                     f'index_wait/work_us={m["index_wait"]:.2f}/{m["index"]:.2f} '
                     f'after_index_CQ_us={cq_text} row_finalize_us={m["row_finalize"]:.2f} '
                     f'row_Q1to4_index_work_us={m["row_index"]} row_Q1to4_index_wait_us={m["row_index_wait"]} '
                     f'truncated_queues={m["capped"]}')
    bracket_range = 'NA' if not exact_brackets else (
        f'{min(lo for lo,_ in exact_brackets):+.2f}..{max(hi for _,hi in exact_brackets):+.2f}')
    lines.append(f'{prefix} GRAPH_TRACE_EFFECTS_US alloff_last_minus_first_same_slot_min/P50/P95/max={distribution(off_drift)} '
                 f'mixed_on_minus_off_same_slot_all_plans_min/P50/P95/max={distribution(unpaired_differences)} '
                 f'bracketing_full_plan_matches={len(exact_brackets)}/20 observed_delta_range={bracket_range}')
    lines.append(f'{prefix} GRAPH_TRACE_COMPAT placement={data.get("placement", "unverified")} '
                 f'archive_matched_plans={archive_matches}/20 crossover_same_slot_exact_plans={peer_matches}/20 '
                 f'observed_exact_plan_on_minus_off_min/P50/P95/max_us={distribution(delta) if delta else "NA"}')
    for phase, items in by_band.items():
        if items:
            leaders = Counter(m['owner'] for m in items)
            lines.append(f'{prefix} GRAPH_TRACE_BAND {phase} n={len(items)} near_last_counts={dict(leaders)} '
                         f'near_last_pref_draining_P50_us={median(m["pre_drain"] for m in items):.2f} '
                         f'near_last_index_wait_P50_us={median(m["index_wait"] for m in items):.2f}')
    coverage = 'both_archive_bands_sampled' if by_band['lower'] and by_band['upper'] else 'missing_archive_band'
    lines.append(f'{prefix} GRAPH_TRACE_COVERAGE {coverage} lower/upper/outside='
                 + '/'.join(str(len(by_band[k])) for k in ('lower','upper','outside')))
    return lines


def verify(root, group):
    serial_base = Path((root/'owner-tail-reference.txt').read_text().strip())
    inputs, digest, plans, old_thp = reference(serial_base, group)
    measured_file = root/'cq-measured-reference.txt'
    base = serial_base
    if measured_file.is_file():
        from cutoff_tail_archive import Q1_CQ_REVISION
        base = Path(measured_file.read_text().strip())
        if Path((base/'early-handoff-reference.txt').read_text().strip()).resolve() != serial_base.resolve():
            raise ValueError('当前版本的强串行参考目录与诊断不一致')
        measured_inputs, measured_digest, plans, old_thp = reference(base, group, Q1_CQ_REVISION)
        if measured_inputs != inputs or measured_digest != digest:
            raise ValueError('当前版本与原始融合版的输入或 TopK 集合不一致')
    archive = read(root/'archive-calls.json')
    calls = archive['groups'][str(group)]
    if (Path(archive['root']).resolve() != base.resolve() or
            [c['layout_receipts'] for c in calls] != plans or
            archive['physical_npu'] != 4 or (root/'physical_npu.txt').read_text().strip() != '4'):
        raise ValueError('Wrong uninstrumented archive')
    group_dir = root/f'g{group}'
    path, = (group_dir/'logical-ties/results').glob('*/cases/diagnose/fused/owner_graph.json')
    data = read(path)
    result = read(path.parent/'path_result.json')
    config = read(path.parent.parent/'config.json')
    expected_config = dict(group=group, query_owner_arm='owner-pack', mode='diagnose', level=1, device=0,
                           warmup=4, iterations=20, ranked_q_workers=2, intermediate_fetches=4,
                           prefetch_tokens=1024, prefetch_rank_start=1024, window_limit=256, fused_only=True)
    if (any(config.get(k) != v for k,v in expected_config.items()) or
            config.get('route_policy') or config.get('ranked_auto_route') or
            result['transport'] != 'urma' or result['runtime_workers'] != 40 or
            result['correctness'] != 'PASS' or result['failures'] or result['production_revision'] != data['native_revision'] or
            result['binary_sha256'] != (root/f'ranked-{data["native_revision"][:8]}.binary.sha256').read_text().strip() or
            data['group'] != group or (root/'logical-ties.txt').read_text().strip() != data['native_revision'] or
            read(path.parent.parent/'replay_input.json') != inputs or data['input_paths'] != inputs['source_paths'] or
            any(c['topk_set_digest'] != digest for p in data['passes'] for c in p['calls'])):
        raise ValueError('Diagnostic graph source, correctness or input/configuration differs from pinned run')
    data['placement'] = pair_placement(old_thp, read(group_dir/'logical-ties/thp.json'), local_baseline=True)
    tick_path = path.parent/'owner_graph_ticks.npz'
    with np.load(tick_path, allow_pickle=False) as raw:
        if measured_file.is_file():
            from prefetch_cq_graph_report import summarize as cq_summary
            comparison, = (base/f'g{group}/logical-ties/results').glob('*/cases/diagnose/fused/native_li_compare.json')
            previous = read(comparison)['native_reference']['replay_identity']
            current = read(path.parent/'cq_replay_identity.json')
            changed = sorted(k for k in previous.keys() | current.keys() if previous.get(k) != current.get(k))
            data['runtime_identity_diff'] = changed
            lines = [f'G{group} 诊断与本版本实测归档的内存放置={data["placement"]}，运行环境差异字段={",".join(changed) or "无"}',
                     *cq_summary(data, raw['ticks'], calls)]
        else:
            lines = summarize(data, raw['ticks'], calls)
    atomic_json(path, data)
    for src in (path, tick_path):
        dest = group_dir/src.name
        if dest.is_symlink() and dest.resolve() == src.resolve():
            continue
        dest.symlink_to(src.relative_to(group_dir))
    print(f'G{group} GRAPH_CHECK TopK_set_vs_archived_strong_serial=PASS calls=80/80 '
          f'queries=5120/5120 KV=PASS ordered_Index=PASS rows=1280/1280')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify'])
    parser.add_argument('root', type=Path)
    parser.add_argument('group', type=int)
    args = parser.parse_args()
    verify(args.root, args.group)
