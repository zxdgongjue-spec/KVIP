# SPDX-License-Identifier: Apache-2.0
"""Compare ranked Index scheduling on a saved BS16 capture and decode the tail endpoints."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from statistics import median

import numpy as np

from peer_candidate import RANKED_BASE
from peer_delta import pair_placement
from ranked_five_report import GROUPS, fmt, load_arm, team_fields


FIXED_REVISION = '5072f2c907f0fbd3bf1c1fbe499dd979c8c1dfca'
BATCH = 16
WORKERS = 2


def distribution(values):
    x = np.asarray(values, dtype=float).ravel()
    if not x.size:
        return dict(n=0, min=None, p50=None, max=None)
    return dict(n=x.size, min=float(x.min()), p50=float(np.median(x)), max=float(x.max()))


def owner_name(block):
    if block < 48:
        return ('producer_finalizer', 'index_planner', 'output')[block % 3]
    return 'get' if block < 56 else 'spare'


def tail_endpoints(raw, abi):
    clock = float(raw['cycles_per_us'])
    # Cast before subtraction: a signed difference identifies which endpoint was later.
    ticks = np.asarray(raw['times'], dtype=np.int64)
    n = len(ticks)
    u = ticks[:, :BATCH*64].reshape(n, BATCH, 64)[:, :, 32:]
    fetch = ticks[:, BATCH*128:BATCH*256].reshape(n, BATCH, 128)
    # A Get worker's aggregate return/group count is replicated over all eight rows it serves.
    q = fetch[:, :WORKERS, :96].reshape(n, WORKERS, 4, 24)
    idx = u[..., 24]
    if (np.any(idx <= 0) or np.any(q[..., 9] <= 0) or np.any(fetch[..., 99] <= 0)
            or np.any(fetch[..., 104] <= 0) or np.any(fetch[..., 105] < fetch[..., 104])):
        raise ValueError('Incomplete Index, Get publication, or final-status endpoints')
    get = np.zeros((n, WORKERS, 4), dtype=np.int64)
    for count, start, end in ((16, 3, 4), (17, 7, 8)):
        active = q[..., count] > 0
        if np.any(active & ((q[..., start] <= 0) | (q[..., end] < q[..., start]))):
            raise ValueError('Missing return from a submitted final Get')
        get = np.maximum(get, np.where(active, q[..., end], 0))
    get_end, index_end = get.max(axis=(1, 2)), idx.max(axis=1)
    has_get = get_end > 0
    delta = (index_end - get_end) / clock
    publish = q[..., 9].max(axis=(1, 2))
    calls = [dict(invocation=i, index_minus_nonempty_get_us=float(delta[i]) if has_get[i] else None,
                  last_index_rows=np.flatnonzero(idx[i] == index_end[i]).tolist(),
                  last_get_q_lane=[(int(qid), int(lane)) for lane, qid in np.argwhere(get[i] == get_end[i])] if has_get[i] else [],
                  get_publication_minus_index_us=float((publish[i]-index_end[i])/clock)) for i in range(n)]
    result = dict(calls=calls, index_minus_nonempty_get_us=distribution(delta[has_get]),
                  index_later=int(sum(delta[has_get] > 0)), get_later=int(sum(delta[has_get] < 0)),
                  same_tick=int(sum(delta[has_get] == 0)), calls_without_get=int(sum(~has_get)),
                  get_publication_minus_index_us=distribution((publish-index_end)/clock),
                  index_flag_wait_us=distribution((fetch[..., 105]-fetch[..., 104])/clock),
                  final_status_minus_last_index_or_get_us=distribution((fetch[..., 99].max(axis=1)-np.maximum(index_end, get_end))/clock))
    if abi == 'original':
        return result

    stamp = ticks[:, BATCH*256+64*32:].reshape(n, 64, 8)
    owner, count, arrival, work, end = team_fields(stamp, flexible=abi == 'flexible', joint=abi == 'joint')
    active = count[..., 0] > 0
    last_field = end.max(axis=2)
    owner_counts = Counter()
    tail_blocks = Counter()
    tail_rows = Counter()
    for i, call in enumerate(calls):
        for block in owner[i, active[i], 3]:
            owner_counts[owner_name(int(block))] += 1
        # Include all exact ties and retain idle rows' join endpoints in the Index/Get comparison.
        if active[i].any():
            finish = end[i, active[i]].max()
            coords = np.argwhere((end[i] == finish) & active[i, :, None])
            call['last_busy_fields'] = []
            for row, field in coords:
                block = int(owner[i, row, field])
                tail_blocks[block] += 1
                tail_rows[int(row)] += 1
                call['last_busy_fields'].append(dict(row=int(row), field=int(field), physical_block=block,
                                                     role=owner_name(block), moves=int(count[i, row, field])))
    has_fields = active.any(axis=1)
    primary_end = np.where(active, end[..., :3].max(axis=2), 0).max(axis=1)
    fourth_end = np.where(active, end[..., 3], 0).max(axis=1)
    last_busy = np.maximum(primary_end, fourth_end)
    observed = has_fields & has_get
    # A row only loses a handoff opportunity if the fourth field has not
    # started when a sibling finishes. Include zero waits in the percentiles.
    free_to_copy = np.maximum(work[..., 3] - end[..., :3].min(axis=2), 0)[active] / clock
    result['row_handoff_gap_us'] = dict(n=int(free_to_copy.size),
        p50=float(np.median(free_to_copy)) if free_to_copy.size else None,
        p95=float(np.percentile(free_to_copy, 95)) if free_to_copy.size else None)
    result.update(last_fourth_minus_last_primary_us=distribution((fourth_end-primary_end)[has_fields]/clock),
                  last_busy_field_minus_nonempty_get_us=distribution((last_busy-get_end)[observed]/clock),
                  last_busy_field_blocks=dict(tail_blocks), last_busy_field_rows=dict(tail_rows),
                  fourth_owner_busy_rows=dict(owner_counts), moves_per_call=distribution(count[..., 0].sum(axis=1)),
                  fourth_finish_minus_last_primary_us=distribution((end[..., 3]-end[..., :3].max(axis=2))[active]/clock),
                  row_status_minus_last_field_us=distribution((idx-last_field)[active]/clock),
                  fourth_arrival_to_copy_us=distribution((work[..., 3]-arrival[..., 3])[active]/clock),
                  fourth_copy_us=distribution((end[..., 3]-work[..., 3])[active]/clock),
                  per_row=[dict(row=r, moves=distribution(count[:, r, 0]),
                                fourth_owners=dict(Counter(int(v) for v in owner[:, r, 3])),
                                fourth_finish_minus_last_primary_us=distribution((end[:, r, 3]-end[:, r, :3].max(axis=1))[active[:, r]]/clock))
                           for r in range(BATCH)])
    if abi in ('flexible', 'joint'):
        plan, joined = u[..., 22], u[..., 23]
        if np.any(plan <= 0) or np.any(joined < plan) or np.any(idx < joined):
            raise ValueError('Incomplete planner publication/join stamps')
        for primary in range(3):
            helper = owner[..., 3] == (np.arange(BATCH)*3 + primary)
            if np.any(helper & (arrival[..., 3] < end[..., primary])):
                raise ValueError('A fourth owner claimed before finishing its own primary field')
        result.update(plan_to_join_us=distribution((joined-plan)[active]/clock),
                      fourth_copy_start_minus_plan_us=distribution((work[..., 3]-plan)[active]/clock),
                      join_minus_last_field_us=distribution((joined-last_field)[active]/clock),
                      status_minus_join_us=distribution((idx-joined)[active]/clock))
    return result


def load(root, group, label, revision, abi):
    arm = load_arm(root/f'g{group}', label, revision, field_abi=abi)
    with np.load(arm['folder']/'lean_ticks.npz', allow_pickle=False) as raw:
        arm['tail'] = tail_endpoints(raw, abi)
    samples = arm['samples']
    if ([v['invocation'] for v in samples] != list(range(len(samples))) or
            not np.allclose([v['us'] for v in samples], arm['timing']['samples_us'], rtol=0, atol=.01)):
        raise ValueError(f'{label}: profile sample order or timing differs from the headline graph')
    digests = arm['digests']
    aux = arm['detail'].get('diagnostic_topk_set_digests', [])
    if (len(digests) != len(arm['samples']) or any(not re.fullmatch('[a-f0-9]{64}', d) for d in digests+aux)
            or (aux and len(aux) != len(digests)) or (abi in ('flexible', 'joint') and len(aux) != len(digests))):
        raise ValueError(f'{label}: incomplete TopK fingerprint coverage')
    if arm['config']['group'] != group or arm['detail']['diagnostic_graph_calls'] != len(digests):
        raise ValueError(f'{label}: wrong group or incomplete auxiliary replay')
    arm['input_hash'] = None
    hashfile = root/f'g{group}/input.sha256'
    if hashfile.exists():
        arm['input_hash'] = hashfile.read_text().strip()
        if not re.fullmatch('[a-f0-9]{64}', arm['input_hash']):
            raise ValueError('Invalid capture SHA256')
    return arm


def by_fingerprint(arm):
    samples = defaultdict(list)
    for i, digest in enumerate(arm['digests']):
        samples[digest].append(i)
    return samples


def matched_samples(before, after):
    a, b = by_fingerprint(before), by_fingerprint(after)
    pairs = [pair for d in sorted(a.keys() & b.keys()) for pair in zip(a[d], b[d])]
    result = dict(pairs=pairs, n=len(pairs), before_calls=len(before['samples']), after_calls=len(after['samples']),
                  shared_fingerprints=len(a.keys() & b.keys()), before_fingerprints=len(a), after_fingerprints=len(b))
    for name in ('span', 'li', 'resolve_update'):
        def stat(arm, i):
            s = arm['samples'][i]
            return s['us'] if name == 'span' else s['stages_us'][name]
        result[name+'_p50_saved_us'] = (median(stat(before, i) for i, _ in pairs) - median(stat(after, j) for _, j in pairs)) if pairs else None
    return result


def compare(before, after):
    if before['inputs'] != after['inputs']:
        raise ValueError('Captured manifest paths, DP order, page mapping, or geometry differ')
    config = ('input', 'level', 'transport', 'device', 'kvaf_root', 'trace_root', 'round_range', 'warmup', 'iterations',
              'prefetch_tokens', 'prefetch_rank_start', 'intermediate_fetches')
    if any(before['config'][k] != after['config'][k] for k in config):
        raise ValueError('Incompatible graph settings or capture root')
    sha_a, sha_b = before['input_hash'], after['input_hash']
    if sha_a and sha_b and sha_a != sha_b:
        raise ValueError('Captured input bytes differ')
    place = pair_placement(before['placement'], after['placement'], local_baseline=True)
    match = matched_samples(before, after)
    timing = before['timing'], after['timing']
    saved = {k: timing[0][k]-timing[1][k] for k in ('event_mean_us', 'p50_us', 'profiler_mean_us',
             'inter_kernel_gap_mean_us', 'outside_invocations_mean_us', 'overlap_mean_us')}
    kernels = {k: dict(mean=x[0]-after['kernel'][k][0], p50=x[1]-after['kernel'][k][1]) for k, x in before['kernel'].items()}
    identical = len(set(before['digests']+after['digests'])) == 1
    if place != 'matched':
        confidence = 'placement differs or is unknown'
    elif not identical:
        confidence = 'graph means include BF16 tie variants; exact-TopK profiler subset is descriptive'
    else:
        confidence = 'matched placement and final TopK; one graph per arm'
    shared_stages = sorted(before['phases'].keys() & after['phases'].keys())
    stage_saved = {}
    for name in shared_stages:
        left, right = before['phases'][name][0], after['phases'][name][0]
        stage_saved[name] = None if left is None or right is None else left-right
    return dict(placement=place, same_capture_sha256=True if sha_a and sha_b else None,
                topk_sets='identical' if identical else 'BF16_boundary_ties', confidence=confidence,
                same_binary=before['binary'] == after['binary'], observed_saved_us=saved,
                observed_saved_pct={k: 100*saved[k]/timing[0][k] for k in ('event_mean_us', 'p50_us')},
                kernels_saved_us=kernels, auxiliary_stage_p50_saved_us=stage_saved, exact_topk_matched=match)


def summary(arm):
    t = arm['timing']
    aux = arm['detail'].get('diagnostic_topk_set_digests')
    spans = t['samples_us']
    k = min(5, len(spans)//2)
    return dict(aux_headline_common_topk_calls=sum((Counter(aux)&Counter(arm['digests'])).values()) if aux else None,
                within_graph_last_minus_first_p50_us=median(spans[-k:])-median(spans[:k]), drift_calls_each_end=k,
                revision=arm['revision'], binary_sha256=arm['binary'], capture_sha256=arm['input_hash'],
                graph_calls=t['graph_calls'], event_mean_us=t['event_mean_us'], profiler_p50_us=t['p50_us'],
                profiler_mean_us=t['profiler_mean_us'], kernel_mean_p50_us=arm['kernel'], traffic=arm['traffic'],
                stages_p50_max_us=arm['phases'], window_stages=arm['windows'], index=arm['index'],
                placement=arm['placement'], tail=arm['tail'], topk_set_digests=arm['digests'],
                diagnostic_topk_set_digests=arm['detail'].get('diagnostic_topk_set_digests'),
                span_us=t['samples_us'])


def report(root, reference, group):
    if ((reference/'baseline.txt').read_text().strip() != RANKED_BASE
            or (reference/'candidate.txt').read_text().strip() != FIXED_REVISION):
        raise ValueError('Reference must contain the original and fixed-field revisions')
    revision = (root/'candidate.txt').read_text().strip()
    if not re.fullmatch('[a-f0-9]{40}', revision) or revision in (RANKED_BASE, FIXED_REVISION):
        raise ValueError('Invalid flexible candidate revision')
    arms = dict(original=load(reference, group, 'baseline', RANKED_BASE, 'original'),
                fixed=load(reference, group, 'candidate', FIXED_REVISION, 'fixed'),
                flexible=load(root, group, 'candidate', revision, 'flexible'))
    if len({a['binary'] for a in arms.values()}) != len(arms):
        raise ValueError('Distinct source revisions produced identical tested binaries')
    pairs = [('original', 'fixed'), ('original', 'flexible'), ('fixed', 'flexible')]
    if (root/f'g{group}/anchor').is_dir():
        arms['anchor'] = load(root, group, 'anchor', FIXED_REVISION, 'fixed')
        pairs += [('fixed', 'anchor'), ('anchor', 'flexible')]
    results = {f'{a}_minus_{b}': compare(arms[a], arms[b]) for a, b in pairs}
    detail = dict(group=group, arms={k: summary(v) for k, v in arms.items()}, comparisons=results,
                  interpretation=(
                      'Positive saved_us means the second arm was faster. One profiled graph per arm. '
                      'Historical deltas include cross-run drift. Exact-TopK subsets pair within digest in '
                      'invocation order. Speculation and polling may still vary. Event time is graph-wide. '
                      'Auxiliary Index endpoint is the last row status preparation after applying/joining Index fields. '
                      'Last Get is the last return from nonempty KVAF final Get calls. Get publication can depend '
                      'on Index. Durations overlap. Endpoint order is observational. Archived final TopK '
                      'fingerprints describe headline calls only.'))
    path = root/f'g{group}/comparison.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(detail, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)
    print(f'G{group} PASS TopK/KV/ordered_Index/hot_valid; event_mean/profiler_p50 us; one graph per arm')
    for key, arm in arms.items():
        t, e = arm['timing'], arm['tail']
        print(f'  {key}: {t["event_mean_us"]:.2f}/{t["p50_us"]:.2f}  index_later/get_later/equal={e["index_later"]}/{e["get_later"]}/{e["same_tick"]} '
              f'index_minus_get_p50_us={fmt(e["index_minus_nonempty_get_us"]["p50"])}')
    for label, comparison in results.items():
        if label == 'original_minus_fixed':
            continue
        saved, kernel, m = comparison['observed_saved_us'], comparison['kernels_saved_us'], comparison['exact_topk_matched']
        print(f'  {label}: observed_event/p50={saved["event_mean_us"]:+.2f}/{saved["p50_us"]:+.2f} '
              f'LI/resolve_update_mean={kernel["li"]["mean"]:+.2f}/{kernel["resolve_update"]["mean"]:+.2f} '
              f'exact_TopK_p50={fmt(m["span_p50_saved_us"])}(pairs={m["n"]} of {m["before_calls"]}/{m["after_calls"]}) '
              f'TopK={comparison["topk_sets"]} placement={comparison["placement"]}')
    # These intervals overlap. Print them individually so a shorter wait can be
    # distinguished from work moving to another part of the same graph.
    stages = results['original_minus_flexible']['auxiliary_stage_p50_saved_us']
    focus = ('planning', 'index_row_incl_wait', 'final_get_cq_wait', 'finalizer_wait_index_flag')
    print('  STAGE original_minus_flexible p50_us ' +
          ' '.join(f'{key}={fmt(stages.get(key))}' for key in focus))
    print('  DRIFT within_graph_last5_minus_first5_p50_us ' +
          ' '.join(f'{label}={row["within_graph_last_minus_first_p50_us"]:+.2f}'
                   for label, row in detail['arms'].items()))
    fixed, new = (arms[label]['tail'] for label in ('fixed', 'flexible'))
    gaps = [v['row_handoff_gap_us'] for v in (fixed, new)]
    print('  HANDOFF row_gap_p50/p95_us ' + ' '.join(
        f'{label}={fmt(g["p50"])}/{fmt(g["p95"])}' for label, g in zip(('fixed', 'flexible'), gaps)))
    print(f'  fixed/flexible last_fourth_minus_last_primary_p50_us={fmt(fixed["last_fourth_minus_last_primary_us"]["p50"])}/'
          f'{fmt(new["last_fourth_minus_last_primary_us"]["p50"])} last_busy_field_minus_get_p50_us='
          f'{fmt(fixed["last_busy_field_minus_nonempty_get_us"]["p50"])}/{fmt(new["last_busy_field_minus_nonempty_get_us"]["p50"])}')
    print(f'  fourth_field busy_owners={new["fourth_owner_busy_rows"]} last_field_blocks={new["last_busy_field_blocks"]} '
          f'join_after_last_field_p50_us={fmt(new["join_minus_last_field_us"]["p50"])} detail={path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--group', type=int, choices=GROUPS, required=True)
    args = parser.parse_args()
    report(args.root, args.reference, args.group)
