# SPDX-License-Identifier: Apache-2.0
"""Compare the joint Post/Index/CQ schedule against a local or archived ranked control."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

import numpy as np

from peer_candidate import RANKED_BASE
from peer_delta import pair_placement
from ranked_five_report import GROUPS, final_q_workers, fmt, team_fields
from ranked_help_report import compare, distribution, load, summary


PREVIOUS = '56dcd74a1450db75ab3133ebf71a3b356e8b2928'
JOINT_CONTROL = '772293d4f0ccb002a1d5210016619af534008e88'
METRICS = ('event_mean_us', 'profiler_p50_us')
STAGES = ('resolve_per_query', 'planning', 'descriptor', 'final_get_post', 'index_row_incl_wait',
          'final_get_cq_wait', 'finalizer_wait_index_flag')


def joint_endpoints(raw):
    """All endpoint differences use timestamps from the same diagnostic invocation."""
    ticks = np.asarray(raw['times'], dtype=np.int64)
    n, clock = len(ticks), float(raw['cycles_per_us'])
    cores = ticks[:, 4096:6144].reshape(n, 64, 32)
    workers = final_q_workers(raw)
    queues = 4*workers
    comm = cores[:, 48:48+queues]
    entered = cores[..., 0].min(axis=1)
    first, post, cq, win_begin, win_end, probe, probe_end, row, prefix, total, resume, resumed = (
        comm[..., i] for i in range(16, 28))
    fetch = ticks[:, 2048:4096].reshape(n, 16, 128)
    lanes = fetch[:, :workers, :96].reshape(n, workers, 4, 24).transpose(0, 2, 1, 3).reshape(n, queues, 24)
    sent = (lanes[..., 16] + lanes[..., 17]) > 0
    tried, won, rest = probe > 0, row > 0, resume > 0
    if (np.any(entered <= 0) or np.any(sent != (first > 0)) or np.any(sent != (cq > 0))
            or np.any(sent != (post > 0)) or np.any(sent & ((first > post) | (post > cq)))
            or np.any(tried != (probe_end > 0)) or np.any(tried & (~sent | (post > probe) | (probe > probe_end) | (probe_end > cq)))
            or np.any(won & (~tried | (row > 16) | (win_begin != probe) | (win_end != probe_end)))
            or np.any((~won) & ((win_begin != 0) | (win_end != 0) | (prefix != 0) | (total != 0)))
            or np.any(won & ((prefix <= 0) | (prefix > 1024) | (prefix > total)))
            or np.any(rest != (resumed > 0))
            or np.any(rest & (~won | (total <= prefix) | (cq > resume) | (resume > resumed)))):
        raise ValueError('Invalid final doorbell/Post/CQ or help interval')
    owners, counts, *_ = team_fields(ticks[:, 6144:].reshape(n, 64, 8), flexible=False, joint=True, workers=workers)
    for call, lane in zip(*np.nonzero(won)):
        target = row[call, lane] - 1
        if (target//4 != lane//workers or target%workers != lane%workers or (row[call] == target+1).sum() != 1
                or counts[call, target, 3] != total[call, lane] or
                ((prefix[call, lane] == total[call, lane] or rest[call, lane])
                 and owners[call, target, 3] != 48 + lane)):
            raise ValueError('Pre-CQ ticket differs from the committed fourth Index field')
    has_final = sent.any(axis=1)
    first_post = np.where(sent, first, np.iinfo('int64').max).min(axis=1)
    last_post, last_cq = post.max(axis=1), cq.max(axis=1)
    last_index = ticks[:, :1024].reshape(n, 16, 64)[..., 56].max(axis=1)
    last_status = fetch[..., 99].max(axis=1)
    if np.any(last_status < np.maximum(last_index, last_cq)):
        raise ValueError('Result was published before Index or final CQ')
    def elapsed(endpoint, mask):
        return distribution((endpoint[mask]-entered[mask])/clock)
    all_calls = np.ones(n, dtype=bool)
    owner = owners[..., 3]
    home = np.where(owner < 48, owner//3, owner-48)
    cross_completions = (home != np.arange(16)) & (counts[..., 3] > 0)
    return dict(calls=n, nonempty_lanes=int(sent.sum()), max_groups_per_phase=int(lanes[..., 16:18].max()),
                conflict_lane_calls=int((lanes[..., 17] > 0).sum()), attempts=int(tried.sum()), wins=int(won.sum()),
                remote_wins=int((won & (row != np.arange(1, queues+1))).sum()),
                cross_row_completions=int(cross_completions.sum()),
                precq_moves=int(prefix.sum()), unprocessed_by_prefix_moves=int((total-prefix).sum()),
                native_suffix_moves=int((total-prefix)[rest].sum()),
                first_doorbell=elapsed(first_post, has_final), last_post=elapsed(last_post, has_final),
                last_cq=elapsed(last_cq, has_final), last_index=elapsed(last_index, all_calls),
                last_status=elapsed(last_status, all_calls),
                last_index_minus_last_cq=distribution((last_index[has_final]-last_cq[has_final])/clock),
                failed_probe_us=distribution((probe_end-probe)[tried & ~won]/clock))


def drift(spans):
    k = min(5, len(spans)//2)
    if k == 0:
        raise ValueError('Need at least two measured invocations')
    return mean(spans[-k:])-mean(spans[:k]), median(spans[-k:])-median(spans[:k])


def history(previous, group, original, candidate):
    archive = json.loads((previous/f'g{group}/comparison.json').read_text())
    ref, prev = (archive['arms'][tag] for tag in ('original', 'flexible'))
    if (archive['group'] != group or ref['revision'] != RANKED_BASE or prev['revision'] != PREVIOUS
            or ref['binary_sha256'] != original['binary_sha256'] or ref['span_us'] != original['span_us']
            or not candidate['capture_sha256'] or prev['capture_sha256'] != candidate['capture_sha256']
            or prev['graph_calls'] != candidate['graph_calls'] or len(prev['span_us']) != prev['graph_calls']
            or prev['binary_sha256'] in (original['binary_sha256'], candidate['binary_sha256'])):
        raise ValueError(f'G{group}: mismatched original, previous or captured input')
    return prev


def early_index_stats(raw):
    """Decode only the dedicated Index cores; communication cores reuse these words."""
    workers = final_q_workers(raw)
    ticks = np.asarray(raw['times'], dtype=np.int64)
    cores = ticks[:, 4096:6144].reshape(-1, 64, 32)
    idle = cores[:, 48 + 4 * workers:]
    start, end, scatter, moves, target = (idle[..., i] for i in (16, 17, 18, 19, 20))
    active = moves > 0
    if np.any(active & ((start < idle[..., 0]) | (start <= 0) | (end < start) |
                        (scatter < end) | (target < 1) | (target > 16))):
        raise ValueError('Invalid early Index read/scatter stamps')
    mhz = float(raw['cycles_per_us'])
    return dict(q_workers=workers, eligible_core_calls=int(active.size), hits=int(active.sum()),
                moves=int(moves[active].sum()), reads_by_row=np.bincount(target[active]-1, minlength=16).tolist(),
                entry_to_read_us=distribution((start-idle[..., 0])[active]/mhz),
                read_us=distribution((end-start)[active]/mhz),
                wait_after_read_us=distribution((scatter-end)[active]/mhz))


def run(root, reference, previous, group, *, joint_control=False):
    if previous is None and not joint_control and root.resolve() != reference.resolve():
        raise ValueError('Local comparison requires the baseline and candidate in the same run directory')
    original = load(reference, group, 'baseline', RANKED_BASE, 'original')
    candidate = load(root, group, 'candidate', (root/'candidate.txt').read_text().strip(), 'joint')
    if joint_control:
        if ((reference/'baseline.txt').read_text().strip() != RANKED_BASE or
                (reference/'candidate.txt').read_text().strip() != JOINT_CONTROL or
                any((root/f).read_text().strip() != (reference/f).read_text().strip()
                    for f in ('host.txt', 'physical_npu.txt'))):
            raise ValueError('Wrong archived candidate, host or physical device')
        prior = load(reference, group, 'candidate', JOINT_CONTROL, 'joint')
        for arm, source in ((original, reference), (prior, reference), (candidate, root)):
            digest = source / f'ranked-{arm["revision"][:8]}.binary.sha256'
            if digest.read_text().strip() != arm['binary']:
                raise ValueError('Recorded binary digest differs from the tested arm')
        prior_comparison = compare(prior, candidate)
    match = compare(original, candidate)
    if joint_control and any(c['placement'] != 'matched' or not c['same_capture_sha256']
                             for c in (match, prior_comparison)):
        raise ValueError('Current and archived runs require identical input bytes and matched THP/NUMA/UB device')
    if match['same_binary']:
        raise ValueError('Candidate binary matches original')
    with np.load(candidate['folder']/'lean_ticks.npz', allow_pickle=False) as raw:
        ends = joint_endpoints(raw)
        early = early_index_stats(raw) if joint_control else None
    original, candidate = summary(original), summary(candidate)
    controls = {'original': original}
    if joint_control:
        controls['previous8'] = summary(prior)
    elif previous is not None:
        controls['previous'] = history(previous, group, original, candidate)
    rows = {**controls, 'candidate': candidate}
    deltas = {}
    for label, baseline in controls.items():
        gains = [baseline[k]-candidate[k] for k in METRICS]
        stages = {}
        for stage in STAGES:
            x, y = (r['stages_p50_max_us'][stage][0] for r in (baseline, candidate))
            stages[stage] = None if x is None or y is None else x-y
        kernel = {k: baseline['kernel_mean_p50_us'][k][0]-candidate['kernel_mean_p50_us'][k][0]
                  for k in ('li', 'resolve_update')}
        place = pair_placement(baseline['placement'], candidate['placement'], local_baseline=True)
        deltas[label] = dict(saved_us=gains, saved_pct=[100*g/baseline[k] for g,k in zip(gains, METRICS)],
                             kernel_mean_saved_us=kernel, auxiliary_p50_saved_us=stages, placement=place)
    drifts = {k: drift(v['span_us']) for k,v in rows.items()}
    source = 'archived controls (cross-run drift unmeasured)' if previous is not None or joint_control else 'local baseline and candidate'
    result = dict(group=group, arms=rows, comparisons=deltas, within_graph_drift=drifts, joint_tail=ends,
                  original_topk_match=match['exact_topk_matched'],
                  protocol=f'One profiled graph per arm, {source}. Mean is graph event, p50 is per-invocation device span. '
                           'Within-graph drift is last five minus first five device spans. Stage diagnostics are a separate graph and overlap.')
    if joint_control:
        result['previous8_topk_match'] = prior_comparison['exact_topk_matched']
        result['early_index'] = early
    path = root/f'g{group}/comparison.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)
    print(f'G{group} PASS TopK/KV/ordered_Index/hot_valid')
    print('  GRAPH event_mean/device_span_p50_us ' + ' '.join(f'{k}={r[METRICS[0]]:.2f}/{r[METRICS[1]]:.2f}' for k,r in rows.items()))
    for label,d in deltas.items():
        def pair(values):
            return '/'.join(f'{v:+.2f}' for v in values)
        print(f'  GAIN vs_{label} saved_us={pair(d["saved_us"])} saved_pct={pair(d["saved_pct"])} placement={d["placement"]}')
        print(f'  STAGE vs_{label} kernel_mean_saved_us ' + ' '.join(f'{k}={v:+.2f}' for k,v in d['kernel_mean_saved_us'].items())
              + ' auxiliary_p50_saved_us ' + ' '.join(f'{k}={fmt(v)}' for k,v in d['auxiliary_p50_saved_us'].items()))
    if joint_control:
        for name, comp in (('original', match), ('previous8', prior_comparison)):
            m = comp['exact_topk_matched']
            print(f'  MATCH vs_{name} pairs={m["n"]}/{m["before_calls"]} '
                  f'span_p50_saved_us={fmt(m["span_p50_saved_us"])} archived=true')
        print('  FOURTH previous8/current_us ' + ' '.join(
            f'{label}=' + '/'.join(fmt(arm['tail'][key][q]) for arm in (controls['previous8'], candidate))
            for label, key, q in (('fourth_minus_primary_p50', 'last_fourth_minus_last_primary_us', 'p50'),
                                  ('row_handoff_p95', 'row_handoff_gap_us', 'p95'),
                                  ('plan_to_fourth_start_p50', 'fourth_copy_start_minus_plan_us', 'p50'))))
        print(f'  EARLY_INDEX comm_cores={4 * early["q_workers"]} '
              f'used_spare_core_calls={early["hits"]}/{early["eligible_core_calls"]} copied_moves={early["moves"]} '
              f'entry_to_read_p50_us={fmt(early["entry_to_read_us"]["p50"])} early_read_p50_us={fmt(early["read_us"]["p50"])} '
              f'wait_to_scatter_p50_us={fmt(early["wait_after_read_us"]["p50"])}')
    print('  DRIFT last5_minus_first5_span_mean/p50_us ' + ' '.join(f'{k}={v[0]:+.2f}/{v[1]:+.2f}' for k,v in drifts.items()))
    print('  TAIL auxiliary_p50_us_from_kernel_entry ' + ' '.join(f'{k}={fmt(ends[k]["p50"])}' for k in
          ('first_doorbell','last_post','last_cq','last_index','last_status','last_index_minus_last_cq')))
    print(f'  HELP attempts/wins/crossrow={ends["attempts"]}/{ends["wins"]}/{ends["remote_wins"]} '
          f'all_remote_field_finishes={ends["cross_row_completions"]} prefix/remainder/native_suffix_moves={ends["precq_moves"]}/{ends["unprocessed_by_prefix_moves"]}/{ends["native_suffix_moves"]} failed_probe_p50_us={fmt(ends["failed_probe_us"]["p50"])} '
          f'max_native_groups_per_phase={ends["max_groups_per_phase"]} conflict_lanes={ends["conflict_lane_calls"]} '
          f'exact_original_TopK_pairs={match["exact_topk_matched"]["n"]}/{candidate["graph_calls"]}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--reference', type=Path, help='Baseline result root (defaults to this run)')
    history_type = p.add_mutually_exclusive_group()
    history_type.add_argument('--joint-control', type=Path, help='Reuse the same-node 772293d4 and original arms')
    history_type.add_argument('--previous', type=Path, help='Previous archived candidate; omit for a local two-arm comparison')
    p.add_argument('--group', type=int, required=True, choices=GROUPS)
    args = p.parse_args()
    if args.reference and args.joint_control:
        p.error('select one reference directory')
    run(args.root, args.joint_control or args.reference or args.root, args.previous, args.group,
        joint_control=bool(args.joint_control))
