# SPDX-License-Identifier: Apache-2.0
"""Compare two ranked URMA source trees on the same captured group."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np

from peer_delta import pair_placement, read
from peer_candidate import RANKED_BASE


MAGIC = 0x49445834
FLEXIBLE_MAGIC = 0x49445835
JOINT_MAGIC = 0x49445836
GROUPS = (2, 7, 10, 17, 4)


def fmt(value):
    return 'NA' if value is None else f'{value:.2f}'


def percentiles(values):
    return (median(values), max(values)) if len(values) else (None, None)


def paired(a, b):
    return f'{fmt(a)}/{fmt(b)}'


def verify_swaps(start_layout, layout, initial_valid, valid, source, dest, counts, *, track_cross_q=False):
    """Replay the five ordered event phases, including overlapping 512-event chunks."""
    history = len(layout)
    resident = len(valid)
    expected_valid = initial_valid.copy()
    positions = np.arange(history)
    owners = positions.copy()
    successful = []
    # Each chunk is a closed permutation: all destinations are also the
    # physical rows read by that chunk, including displaced victims.
    last_query = np.full(history, -1, dtype=np.int8) if track_cross_q else None
    cross_q = False
    suffix = 2048
    for phase, count in enumerate(counts):
        begin = suffix if phase else 0
        if phase:
            suffix += count
        if count < 0 or count > 2048 or begin+count > len(source):
            raise ValueError('Out-of-range Index event count')
        for offset in range(begin, begin+count, 512):
            end = min(offset+512, begin+count)
            orig, dst = source[offset:end], dest[offset:end]
            if np.any((orig < 0) | (orig >= history) | (dst < 0) | (dst >= resident)):
                raise ValueError('Out-of-range Index event')
            src = positions[orig]
            if len(np.unique(src)) != len(src) or len(np.unique(dst)) != len(dst):
                raise ValueError('Index chunk contains duplicate sources/destinations')
            # Move the displaced old owners to freed slots, preserving publication
            # order on both sides. All reads within one chunk see its input state.
            displaced = dst[~np.isin(dst, src, assume_unique=True)]
            freed = src[~np.isin(src, dst, assume_unique=True)]
            all_dest = np.r_[dst, freed]
            if track_cross_q and not cross_q:
                query = max(0, phase - 1)  # Q0 also applies the prefetch events.
                previous = last_query[all_dest]
                cross_q = bool(np.any((previous >= 0) & (previous != query)))
                last_query[all_dest] = query
            next_owners = owners[np.r_[src, displaced]]
            owners[all_dest] = next_owners
            positions[next_owners] = all_dest
            expected_valid[freed[freed < resident]] = 0
            successful.append(dst)
    # Residency of all completed Gets is finally published after the update.
    for slots in successful:
        expected_valid[slots] = 1
    if not np.array_equal(layout, start_layout[owners]):
        raise ValueError('Committed Index does not implement ordered Get/eviction events')
    if not np.array_equal(valid, expected_valid):
        raise ValueError('Committed hot-valid bitmap differs from actual promotions and evictions')
    return cross_q


def final_q_workers(raw):
    # Archived traces lack this field and were recorded with eight final AIVs.
    workers = int(raw['q_workers']) if 'q_workers' in raw else 2
    if workers not in (1, 2, 4):
        raise ValueError(f'Invalid final workers per Q: {workers}')
    return workers


def team_fields(stamp, *, flexible, joint=False, workers=2):
    """Validate logical ownership and return [call,row,field] vectors for each trace word."""
    n = len(stamp)
    row = np.array([block//3 if block < 48 else block-48 for block in range(64)])
    field = np.array([block%3 if block < 48 else 3 for block in range(64)])
    magic = JOINT_MAGIC if joint else (FLEXIBLE_MAGIC if flexible else MAGIC)
    if stamp.shape != (n, 64, 8) or np.any(stamp[..., 0] != magic):
        raise ValueError('Unexpected Index field trace format')
    packed = stamp[..., 2]
    physical = np.arange(64)
    if flexible or joint:
        owner = packed >> 32
        legal_fourth = (owner == 3*row) | (owner == 3*row+1) | (owner == 3*row+2) | (owner == physical)
        if joint:
            # A completed field may belong to its original row, an idle
            # neighbour, or the final Get lane that ran its bounded prefix.
            lane = owner - 48
            queues = 4 * workers
            near = np.where(lane < queues, lane + 1, lane - queues)
            queue_row = (row//4 == lane//workers) & (row%workers == lane%workers)
            remote = np.where(owner < 48, row == (owner//3+1+5*(owner%3))%16,
                              (row == near%16) | (row == (near+1)%16)
                              | ((lane < queues) & queue_row)
                              | ((lane >= queues) & (row == lane % queues)))
            legal_fourth |= (owner >= 0) & (owner < 64) & remote
        if (np.any(stamp[..., 1] != row) or np.any((packed & 0xffffffff) != field)
                or np.any(np.where(field == 3, ~legal_fourth, owner != physical))):
            raise ValueError('Invalid Index row, field, or physical owner')
    else:
        row, field = stamp[..., 1], packed
        if np.any(row < 0) or np.any(row >= 16) or np.any(field < 0) or np.any(field >= 4):
            raise ValueError('Invalid fixed Index row or field')
        owner = np.where(field == 3, 48+row, 3*row+field)
    if np.any(stamp[..., 7]) or np.any(np.sort(row*4+field, axis=-1) != np.arange(64)):
        raise ValueError('Failed, missing, or duplicate Index field')
    order = np.broadcast_to(np.argsort(row*4+field, axis=-1), (n, 64))
    def rows(a):
        return np.take_along_axis(a, order, axis=1).reshape(n, 16, 4)
    count, begin, end, work = (rows(stamp[..., k]) for k in (3, 4, 5, 6))
    if np.any(count != count[..., :1]) or np.any(count < 0):
        raise ValueError('Index fields disagree on ordered moves')
    if np.any(begin <= 0) or np.any(work < begin) or np.any(end < work):
        raise ValueError('Invalid Index field counter interval')
    return rows(owner), count, begin, work, end


def diagnostic_intervals(raw, detail, *, candidate, flexible=False, joint=False):
    """Eligible intervals from an auxiliary replay. Roles overlap in device time."""
    clock = float(raw['cycles_per_us'])
    ticks = raw['times']
    if not np.isfinite(clock) or clock <= 0:
        raise ValueError('Invalid counter calibration')
    n, batch = len(ticks), 16
    base = batch * 256
    workers = final_q_workers(raw)
    old_cores = 3 * batch + 4 * workers
    core_count = 64 if candidate else old_cores
    trailer = base + 64*32
    expected_words = trailer + 64*8 if candidate else base + old_cores*32
    if ticks.shape != (detail['measured_graph_calls'], expected_words):
        raise ValueError(f'Timing buffer shape mismatch: {ticks.shape}, expected {expected_words} words/call')
    per_row = ticks[:, :batch*64].reshape(n, batch, 64)
    fetch = ticks[:, batch*128:base].reshape(n, batch, 128)
    core = ticks[:, base:base+core_count*32].reshape(n, core_count, 32)
    output = core[:, 2:3*batch:3]
    intervals = {}
    def measure(name, begin, end):
        valid = (begin > 0) & (end >= begin)
        intervals[name] = percentiles(((end[valid]-begin[valid])/clock).reshape(-1))

    produce = per_row[:, :, 10:30].reshape(n, batch, 4, 5)
    measure('resolve_per_query', produce[..., 0], produce[..., 1])
    out = output[:, :, 4:28].reshape(n, batch, 4, 6)
    for name, left, right in (('output_ID', 0, 1), ('output_wait', 1, 2),
                              ('descriptor', 2, 3), ('output_slots', 3, 4),
                              ('output_publish', 4, 5)):
        measure(name, out[..., left], out[..., right])
    measure('final_residency', fetch[..., 97], fetch[..., 98])
    measure('final_status', fetch[..., 98], fetch[..., 99])
    for name in ('planning', 'li_final_classification', 'prefetch_cq_drain',
                 'final_get_post', 'final_get_cq_wait', 'finalize_including_waits'):
        x = detail['stages'][name]
        intervals[name] = x['p50_us'], x['max_us']

    u = per_row[:, :, 32:]
    measure('index_row_incl_wait', u[..., 10], u[..., 24])
    # The finalizer's flag 6 is published by the Index owner on both arms.
    # Its wait covers only the part still exposed when this finalizer arrives.
    measure('finalizer_wait_index_flag', fetch[..., 104], fetch[..., 105])
    flags = (u[..., 10] > 0) & (u[..., 24] >= u[..., 10])
    row_walls = [(u[i, mask, 24].max() - u[i, mask, 10].min()) / clock
                 for i, mask in enumerate(flags) if mask.any()]
    ig = detail['stages']['index_apply_general']
    record = {'row_owner_envelope_us': percentiles(row_walls),
              'five_phase_probed_total_rows': (ig['n'], ig['eligible'])}
    if candidate:
        stamp = ticks[:, trailer:].reshape(n, 64, 8)
        owner, count, begin, work, end = team_fields(stamp, flexible=flexible, joint=joint, workers=workers)
        active = count > 0
        counts = active.sum(axis=(1, 2))
        if not np.all(counts):
            raise ValueError('Missing nonempty Index work on a real ranked capture')
        # A helper may handle two fields. Count distinct physical AIVs.
        busy = [len(np.unique(owner[i, active[i]])) for i in range(n)]
        record['active_AIV_min_max'] = min(busy), max(busy)
        record['moved_p50_max'] = percentiles(count[..., 0].sum(axis=1).tolist())
        measure('index_lane_wait', begin[active], work[active])
        measure('index_lane_body', work[active], end[active])
        intervals['index_team_plan_five_phases'] = ig['p50_us'], ig['max_us']
        record['copy_team_envelope_us'] = percentiles([(end[i, active[i]].max()-begin[i, active[i]].min())/clock
                                                      for i in range(n)])
    else:
        # On the baseline the same five phase counters include both planning
        # and applying the exchange, so they cannot be subtracted from a
        # candidate plan-only or field-lane measurement.
        intervals['index_general_plan_and_apply'] = ig['p50_us'], ig['max_us']
    return intervals, record


def load_arm(case_dir, label, revision, *, field_abi=None):
    path, = (case_dir / label / 'results').glob('**/fused/path_result.json')
    config = read(path.parents[1] / 'config.json')
    inputs = read(path.parents[1] / 'replay_input.json')
    result = read(path)
    thp = read(case_dir / label / 'thp.json')
    geometry = dict(history=131328, resident=16384, heads=32, token_bytes=656, output_stride=16896, batch=16)
    if (inputs['rounds'] != 1 or any(inputs['geometry'].get(k) != v for k, v in geometry.items())
            or Counter(x[0] for x in inputs['identities']) != {dp:2 for dp in range(8)}
            or len(set(tuple(i) for i in inputs['identities'])) != 16 or inputs['page_count'] <= 0):
        raise ValueError(f'{label}: invalid eight-DP real BS16 input')
    if (result['correctness'] != 'PASS' or result['failures'] or result['production_revision'] != revision
            or result['path'] != 'fused' or result['transport'] != 'urma' or result['runtime_workers'] != 40
            or len(result['binary_sha256']) != 64 or result['input_paths'] != inputs['source_paths']):
        raise ValueError(f'{label}: unexpected binary, transport, or failed output/Index validation')
    if (thp['status'] != 'measured' or thp['page_mode'] != 'thp'
            or thp['bytes'] != (inputs['page_count'] + 2)*656*128):
        raise ValueError(f'{label}: wrong arena or missing THP observation')
    timing = result['all_rounds']
    folder = path.parent / 'diagnose/round-1'
    profile = read(folder / 'graph_profile.json')
    d = read(folder / 'lean_stages.json')
    if (timing['timing_protocol'] != 'one_profiled_graph' or timing['measured_graphs'] != 1
            or timing['samples'] != config['iterations'] or timing['samples'] < 2
            or timing['graph_calls'] != timing['samples'] or profile['accounting'].get('unclassified')
            or len(profile['samples']) != timing['samples'] or d['diagnostic_graph_replays'] != 1):
        raise ValueError(f'{label}: incomplete timing/diagnostics')
    accounting = timing['stage_mean_us']
    if set(accounting) != {'li','resolve_update'}:
        raise ValueError(f'{label}: unexpected kernels {sorted(accounting)}')
    edges = timing['inter_kernel_gap_mean_us'] - timing['overlap_mean_us'] + timing['outside_invocations_mean_us']
    if abs(sum(accounting.values()) + edges - timing['event_mean_us']) > .01:
        raise ValueError(f'{label}: device graph accounting does not reconcile')
    kernel = {k: (v, median(s['stages_us'][k] for s in profile['samples'])) for k, v in accounting.items()}
    with np.load(folder / 'lean_ticks.npz', allow_pickle=False) as raw:
        if final_q_workers(raw) != int(config.get('ranked_q_workers') or 2):
            raise ValueError(f'{label}: recorded final-worker geometry differs from config')
        if field_abi is None:
            field_abi = 'fixed' if label == 'candidate' else 'original'
        phases, index = diagnostic_intervals(raw, d, candidate=field_abi != 'original',
                                             flexible=field_abi == 'flexible', joint=field_abi == 'joint')
    return dict(config=config, inputs=inputs, placement=thp, timing=timing, kernel=kernel,
                phases=phases, index=index, traffic=d['traffic'], windows=d['window_stages'],
                digests=result['topk_set_digests'], binary=result['binary_sha256'], folder=folder,
                samples=profile['samples'], detail=d, revision=revision)


def report(root, group):
    case_dir = root / f'g{group}'
    before, after = [load_arm(case_dir, label, (root / f'{label}.txt').read_text().strip())
                     for label in ('baseline','candidate')]
    if (root / 'baseline.txt').read_text().strip() != RANKED_BASE or before['binary'] == after['binary']:
        raise ValueError('Invalid ranked source or identical baseline/candidate binary')
    if before['inputs'] != after['inputs']:
        raise ValueError('Baseline/candidate input manifest or DP order differs')
    keys = ('trace_root','group','round_range','level','device','warmup','iterations','transport',
            'prefetch_tokens','prefetch_rank_start','intermediate_fetches','kvaf_root')
    if (before['config']['group'] != group or
            any(before['config'][k] != after['config'][k] for k in keys)):
        raise ValueError('Run settings differ')
    if (len(before['digests']) != before['timing']['graph_calls']
            or len(after['digests']) != after['timing']['graph_calls']
            or any(len(d) != 64 for arm in (before, after) for d in arm['digests'])):
        raise ValueError('Incomplete final TopK fingerprint coverage')
    topk = 'identical' if len(set(before['digests'] + after['digests'])) == 1 else 'BF16_boundary_ties'
    place = pair_placement(before['placement'], after['placement'], local_baseline=True)
    for label, data in (('baseline',before),('candidate',after)):
        t = data['timing']
        spans = t['samples_us']
        k = min(5,len(spans)//2)
        drift = (mean(spans[-k:])-mean(spans[:k]), median(spans[-k:])-median(spans[:k]))
        print(f'G{group} {label} event_mean/span_p50_us={paired(t["event_mean_us"],t["p50_us"])} '
              f'span_mean_us={t["profiler_mean_us"]:.2f} within_graph_last{k}_minus_first{k}_mean/p50_us={paired(*drift)}')
        print(f'  KERNEL mean/p50_us', ' '.join(f'{name}={paired(*value)}' for name,value in data['kernel'].items()),
              f'outside_mean={t["outside_invocations_mean_us"]:.2f} gap_mean={t["inter_kernel_gap_mean_us"]:.2f} '
              f'overlap_mean={t["overlap_mean_us"]:.2f}')
        fields = [f'{name}={paired(*value)}' for name,value in data['phases'].items()]
        for i in range(0,len(fields),5):
            print('  COUNTERS p50/max_us', ' '.join(fields[i:i+5]))
        windows = [f'{w["epoch"]}:{fmt(w["service_us"]["p50_us"])}/{fmt(w["incremental_us"]["p50_us"])}/{fmt(w["active_post_us"]["p50_us"])}'
                   for w in data['windows']]
        print('  WINDOW epoch:service/incremental/active_post_p50_us', ' '.join(windows))
        print('  INDEX', ' '.join(f'{key}={paired(*value)}' for key,value in data['index'].items()))
        traffic = data['traffic']
        print('  READS', ' '.join(f'{key}={traffic[key]["p50"]:g}' for key in
                                  ('initial_cold','selected_unique','prefetch','final','unused_prefetch')))
    if place == 'matched' and topk == 'identical':
        ref, new = before['timing'], after['timing']
        fields = []
        for k in ('event_mean_us','p50_us'):
            diff = ref[k]-new[k]
            fields.append(f'{k}={diff:+.2f}({100*diff/ref[k]:+.2f}%)')
        print(f'G{group} GAIN saved_vs_ranked', ' '.join(fields))
        print('  KERNEL_DELTA saved_mean/p50_us', ' '.join(
            f'{k}={paired(*(a-b for a,b in zip(before["kernel"][k],after["kernel"][k])))}'
            for k in before['kernel']))
        shared = before['phases'].keys() & after['phases'].keys()
        deltas = []
        for key in sorted(shared):
            b50 = before['phases'][key][0]
            a50 = after['phases'][key][0]
            deltas.append(f'{key}={fmt(b50-a50 if b50 is not None and a50 is not None else None)}')
        for i in range(0, len(deltas), 5):
            print('  COUNTER_DELTA saved_p50_us', ' '.join(deltas[i:i+5]))
    else:
        print(f'G{group} GAIN saved_vs_ranked=NA placement={place} topk_sets={topk}')
    print(f'G{group} CHECK inputs/TopK/KV/ordered_Index/hot_valid=PASS topk_sets={topk} placement={place} '
          'counters=auxiliary_nonadditive Index_body_scopes=baseline_general_vs_candidate_field_lanes '
          'unobserved=NA causal_hidden_time=NA')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--group', type=int, choices=GROUPS, required=True)
    args = parser.parse_args()
    report(args.root, args.group)
