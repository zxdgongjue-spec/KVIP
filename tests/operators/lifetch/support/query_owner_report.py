"""One comparison per real G group. Readbacks run after graph timing."""
import hashlib
import re
import sys
from collections import Counter, defaultdict, deque
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median

import numpy as np

from peer_delta import pair_placement, read

SHA256 = re.compile(r'[0-9a-f]{64}')

def check_cutoff_ids(inputs, layout, ids):
    """The ordinary score oracle checks the Kth value. Also verify the physical tie break."""
    for b, row in enumerate(layout):
        for q, actual in enumerate(ids[b]):
            valid = inputs.logical_score_valid[b, q, row]
            tokens = row[valid]
            keys = inputs.logical_score_keys[b, q, tokens]
            k = actual.size
            cutoff = np.partition(keys, len(keys) - k)[-k]
            above = tokens[keys > cutoff]
            expected = np.concatenate((above, tokens[keys == cutoff][:k - len(above)]))
            if not np.array_equal(np.sort(actual), np.sort(expected)):
                missing = np.setdiff1d(expected, actual)[:6].tolist()
                extra = np.setdiff1d(actual, expected)[:6].tolist()
                raise AssertionError(f'Physical cutoff tie rule: request={b} Q={q+1} '
                                     f'cutoff={int(cutoff)} missing={missing} extra={extra}')


def owner_receipts(posts):
    if posts.shape != (16, 4, 32):
        raise ValueError(f"Unexpected FinalPost shape {posts.shape}")
    row, q = np.indices((16, 4))
    view = np.where(q == 0, 2 * row + 1, 32 + 3 * row + q - 1)
    if (not np.array_equal(posts[:, :, 8], 4 * row + q + 1)
            or not np.array_equal(posts[:, :, 9], view)
            or not np.array_equal(posts[:, :, 10], posts[:, :, 0])
            or np.any(posts[:, :, 1:4]) or np.any(posts[:, :, 0] > 2048)):
        raise ValueError("FinalPost owner, queue, completion count or CQ status differs")
    return dict(owners=64, queues=int(np.unique(view).size), processed=int(posts[:, :, 10].sum()))


def check_offset_plan(events_src, events_dst, counts, posts, candidates, selected, slots, reserved):
    """Check publication order, prefetch exclusion and victim choice after timing."""
    if np.any(posts[:, :, 5] != posts[:, :, 0]) or np.any(posts[:, :, 11:13] > 1):
        raise ValueError('Unsafe final Get or invalid offset-path receipt')
    if np.any(posts[:, :, 12] != posts[:, :1, 12]):
        raise ValueError('Index owners disagreed on the row plan')
    extended_slots = 0
    for row, (used, *per_q) in enumerate(counts):
        used = int(used)
        if not 0 <= used <= len(candidates[row]) or list(posts[row, :, 0]) != per_q:
            raise ValueError('Prefetch or per-query counts differ')
        total = sum(per_q)
        final_src = events_src[row, 2048:2048+total]
        final_dst = events_dst[row, 2048:2048+total]
        lru = candidates[row]
        if (len(set(final_dst)) != total or len(set(final_src)) != total or
                not np.array_equal(lru[:used], events_dst[row, :used]) or
                np.intersect1d(lru[:used], final_dst).size):
            raise ValueError(f'Genuine prefetch/final destination overlap or duplicate on row {row}')
        missing = set(final_src.tolist())
        assigned = dict(zip(final_src.tolist(), final_dst.tolist()))
        seen = set()
        cursor = 0
        for q, amount in enumerate(per_q):
            ordered = []
            for token, slot in zip(selected[row, q], slots[row, q]):
                if token in assigned and slot != assigned[token]:
                    raise ValueError(f'Final token maps to the wrong physical slot on row {row} Q{q+1}')
                if token in missing and token not in seen:
                    ordered.append(token)
                    seen.add(token)
            if not np.array_equal(ordered, final_src[cursor:cursor+amount]):
                raise ValueError(f'Wrong unique miss owner or order on row {row} Q{q+1}')
            cursor += amount
        known = slots[row][~np.isin(selected[row], final_src)]
        protected = set(known.tolist())
        inverse = reserved[row]
        taken = set(lru[:used])
        order = [slot for i, slot in enumerate(lru) if i >= used and 0 <= slot < len(inverse) and inverse[slot] == i
                 and slot not in protected and slot not in taken]
        if len(order) < total:
            extended_slots += total - len(order)
            # All ordinary ordinals follow the LRU suffix in physical order.
            occupied = protected.union(taken, order)
            order.extend(slot for slot in range(len(inverse)) if slot not in occupied)
        if not np.array_equal(order[:total], final_dst):
            raise ValueError(f'Wrong exclusive-prefix allocation on request {row}')
    return dict(extended_slots=extended_slots, independent_rows=int(posts[:, 0, 12].sum()))


def layout_receipt(layout, valid, sources, destinations, counts):
    prefetch, final = int(counts[0]), int(sum(counts[1:]))
    plan = hashlib.sha256(counts.tobytes())
    for events in (sources, destinations):
        plan.update(events[:prefetch].tobytes())
        plan.update(events[2048:2048 + final].tobytes())
    state = hashlib.sha256(layout.tobytes())
    state.update(valid.tobytes())
    return plan.hexdigest(), state.hexdigest()


def load(root, group, label, *, row_dependencies=False):
    paths = list((root / f'g{group}' / label / 'results').glob('**/fused/path_result.json'))
    if len(paths) != 1:
        raise ValueError(f'G{group} {label}: expected one fused path result, found {len(paths)}')
    path = paths[0]
    out = read(path)
    cfg = read(path.parents[1] / 'config.json')
    inputs = read(path.parents[1] / 'replay_input.json')
    revision = (root / f'{label}.txt').read_text().strip()
    expected = dict(batch=16, token_bytes=656, history=131328, resident=16384)
    if (out['production_revision'] != revision or out['correctness'] != 'PASS' or out['failures']
            or out['transport'] != 'urma' or out['runtime_workers'] != 40 or cfg['device'] != 0
            or cfg['window_limit'] != 256 or cfg['query_owner_arm'] != label or cfg['group'] != group or cfg['intermediate_fetches'] != 4 or cfg['prefetch_tokens'] != 1024
            or cfg.get('ranked_auto_route') or cfg.get('route_policy') or out.get('auto_route')
            or cfg.get('prefetch_rank_start') != 1024 or cfg.get('ranked_q_workers') != 2
            or cfg['level'] != 1 or cfg['warmup'] != 4 or cfg['iterations'] != 20
            or any(inputs['geometry'][k] != v for k, v in expected.items())
            or inputs['rounds'] != 1 or out['input_paths'] != inputs['source_paths']
            or Counter(i[0] for i in inputs['identities']) != {dp: 2 for dp in range(8)}
            or len(set(map(tuple, inputs['identities']))) != 16):
        raise ValueError(f'G{group} {label}: unexpected input, runtime, configuration or failed validation')
    timing = out['all_rounds']
    times = timing['samples_us']
    rows = read(path.parent / 'diagnose/round-1/full_attribution.json')['samples']
    if (len(times) != len(rows) or len(times) != 20 or timing['measured_graphs'] != 1
            or timing['timing_protocol'] != 'one_profiled_graph' or min(times) <= 0
            or len(out['topk_set_digests']) != len(rows)):
        raise ValueError(f'{label}: incomplete graph measurement')
    binary = (root / f'ranked-{revision[:8]}.binary.sha256').read_text().strip()
    if out['binary_sha256'] != binary:
        raise ValueError(f'{label}: replay loaded a different native library')
    for sample in rows:
        requests = sample['algorithm']['requests']
        if len(requests) != 16 or any(len(r['windows']) != 4 or r['prefetch'] > 1024
                                      or any(w['posted'] > 256 for w in r['windows'])
                                      for r in requests):
            raise ValueError(f'{label}: observed window count or prefetch budget differs')
        if label == 'owner-pack':
            for request in requests:
                queries = request['queries']
                if (len(queries) != 4 or sum(q['final_posted'] for q in queries) != request['final']
                        or any(not 0 <= q['final_safe'] <= q['final_posted'] for q in queries)):
                    raise ValueError('Per-Q safe/count does not match the final descriptors')
    traffic = [v['algorithm']['totals'] for v in rows]
    if any(v['KV_read_bytes'] != 656 * (v['final'] + v['prefetch']) for v in traffic):
        raise ValueError(f'{label}: inconsistent physical Get bytes')
    if label not in ('baseline', 'fixed-order'):
        receipts = out['query_owners']
        if len(receipts) != len(rows) or any(r != dict(owners=64, queues=64, processed=v['final'])
                                              for r, v in zip(receipts, traffic)):
            raise ValueError('Query-owner CQ counts differ from actual final Get traffic')
    miss = 100 * sum(v['initial_cold'] for v in traffic) / sum(v['selected_unique'] for v in traffic)
    kib = [mean(v[k] for v in traffic) * 656 / 1024 for k in ('unused_prefetch', 'final')]
    mib = mean(v['KV_read_bytes'] for v in traffic) / 2**20
    stages = timing['stage_mean_us']
    drift = [100 * (metric(times[-5:]) / metric(times[:5]) - 1) for metric in (mean, median)]
    m, p = timing['mean_us'], median(times)
    if not row_dependencies:
        print(f'G{group} {label:<9s} full_mean/P50_us={m:.2f}/{p:.2f} '
              f'LI/tail_mean_us={stages["li"]:.2f}/{stages["resolve_update"]:.2f} '
              f'span_drift_mean/P50_pct={drift[0]:+.2f}/{drift[1]:+.2f} miss_pct={miss:.3f} '
              f'waste/repair_KiB={kib[0]:.1f}/{kib[1]:.1f} total_MiB={mib:.4f}', flush=True)
    thp = read(root / f'g{group}' / label / 'thp.json')
    branch_counts = None
    if label in ('offset', 'row-deps', 'li-packets'):
        receipts = out['offset_checks']
        if len(receipts) != 20:
            raise ValueError('Missing post-timing slot checks')
        independent = [r['independent_rows'] for r in receipts]
        if any(type(n) != int or not 0 <= n <= 16 for n in independent):
            raise ValueError('Invalid disjoint request count')
        # Archived runs retain native branch counts but only the new runner
        # reconstructs physical row reuse across Qs after each measured call.
        overlap = out.get('cross_q_overlap_requests')
        if overlap is not None and (
                len(overlap) != 20 or any(type(n) != int or not 0 <= n <= 16 - fast
                                          for n, fast in zip(overlap, independent))):
            raise ValueError('Invalid cross-query physical row reuse count')
        branch_counts = (sum(independent), None if overlap is None else sum(overlap))
        if not row_dependencies:
            print(f"G{group} OFFSET_CHECK calls=20/20 prefetch_slot_reuse=0 "
                  f"hot_slots_selected_beyond_list={sum(r['extended_slots'] for r in receipts)} "
                  f"independent_Index_rows={sum(r['independent_rows'] for r in receipts)}/320", flush=True)
    result = (inputs, out['topk_set_digests'], thp, m, p, out['binary_sha256'], drift, times,
              stages['resolve_update'], mean(v['final'] for v in traffic))
    if not row_dependencies:
        return result
    if any(sample['accounting']['invocation'] != i or sample['us'] != times[i]
           for i, sample in enumerate(rows)):
        raise ValueError(f'G{group} {label}: graph samples do not match per-invocation attribution')
    tail = [sample['accounting']['stages_us']['resolve_update'] for sample in rows]
    if (not all(isfinite(t) and 0 < t <= span for t, span in zip(tail, times))
            or not isclose(mean(tail), stages['resolve_update'])
            or not isfinite(m) or m <= 0):
        raise ValueError(f'G{group} {label}: invalid resolver timing')
    return (*result, median(tail), out['layout_receipts'], branch_counts, stages['li'], cfg['kvaf_root'])


def report_resolver(root, group):
    offset = (root / 'offset.txt').is_file()
    names = ('chain', 'offset', 'fixed-order') if offset else ('chain', 'spill', 'local')
    arms = {arm: load(root, group, arm) for arm in names}
    pairs = [('chain', 'offset'), ('fixed-order', 'offset')] if offset else [('spill', 'local'), ('chain', 'local')]
    if not offset and group in (7, 4):
        arms['fixed-order'] = load(root, group, 'fixed-order')
        pairs.append(('fixed-order', 'local'))
    original = arms['chain']
    if len({v[5] for v in arms.values()}) != len(arms) or any(v[0] != original[0] for v in arms.values()):
        raise ValueError('Expected distinct binaries and identical captured inputs')
    digests = [d for v in arms.values() for d in v[1]]
    if len(digests) != 20 * len(arms) or len(set(digests)) != 1:
        raise ValueError('Final TopK differs between invocations or arms')
    print(f'G{group} EXACT_ID oracle_per_arm=20/20 queries_per_call=64 owners=64/64')
    for before, after in pairs:
        a, b = arms[before], arms[after]
        gains = [100 * (1 - b[i] / a[i]) for i in (3, 4)]
        drift = [max(abs(x), abs(y)) for x, y in zip(a[6], b[6])]
        place = pair_placement(a[2], b[2], local_baseline=True)
        if place != 'matched':
            gate = 'PLACEMENT_MISMATCH'
        elif all(g > d for g, d in zip(gains, drift)) and b[8] < a[8]:
            gate = 'POSITIVE'
        elif min(gains) > 0:
            gate = 'INCONCLUSIVE'
        else:
            gate = 'KEEP_CONTROL'
        print(f'G{group} {before}->{after} graph_saved_mean/P50_us={a[3]-b[3]:+.2f}/{a[4]-b[4]:+.2f} '
              f'tail_saved_mean_us={a[8]-b[8]:+.2f} repair_delta_KiB={(b[9]-a[9])*656/1024:+.1f} '
              f'gain_pct={gains[0]:+.2f}/{gains[1]:+.2f} drift_guard_pct={drift[0]:.2f}/{drift[1]:.2f} {gate}')


def compare_layout_receipts(arms, candidate):
    """Compare exact physical Index for identical ordered events on the same request."""
    states = {}
    arms_by_plan = defaultdict(set)
    calls_by_arm = {}
    for name, calls in arms.items():
        if len(calls) != 20 or any(len(rows) != 16 for rows in calls):
            raise ValueError(f'Incomplete layout receipts for {name}')
        calls_by_arm[name] = Counter(tuple(plan for plan, _ in rows) for rows in calls)
        for rows in calls:
            for request, (plan, state) in enumerate(rows):
                if not SHA256.fullmatch(plan) or not SHA256.fullmatch(state):
                    raise ValueError(f'Invalid layout receipt for {name}, request {request}')
                key = request, plan
                if states.setdefault(key, state) != state:
                    raise ValueError(f'Different physical Index for identical events, request {request}')
                arms_by_plan[key].add(name)
    candidate_calls = calls_by_arm[candidate]
    return {
        name: (sum(name in arms_by_plan[request, plan]
                   for rows in arms[candidate] for request, (plan, _) in enumerate(rows)),
               sum((candidate_calls & calls_by_arm[name]).values()))
        for name in arms if name != candidate
    }


def report_row_dependencies(root, group):
    packets = (root / 'li-packets.txt').is_file()
    candidate = 'li-packets' if packets else 'row-deps'
    controls = ('fixed-order', 'row-deps' if packets else 'offset')
    marker = root / 'packet-reference.txt'
    reference = Path(marker.read_text().strip()) if marker.exists() else None
    controls_root = reference or root
    if reference:
        for key in ('row-deps.txt', 'fixed-order.txt', 'physical_npu.txt'):
            if (reference / key).read_text().strip() != (root / key).read_text().strip():
                raise ValueError(f'Archived control identity differs: {key}')
    arms = {name: load(controls_root, group, name, row_dependencies=True) for name in controls}
    arms[candidate] = load(root, group, candidate, row_dependencies=True)
    new = arms[candidate]
    if len({v[5] for v in arms.values()}) != 3 or any(v[0] != new[0] for v in arms.values()):
        raise ValueError('Controls must use identical inputs and distinct verified binaries')
    digests = [digest for value in arms.values() for digest in value[1]]
    if len(digests) != 60 or len(set(digests)) != 1 or not SHA256.fullmatch(digests[0]):
        raise ValueError('Different TopK IDs on identical real inputs')
    if any(v[14] != new[14] for v in arms.values()):
        raise ValueError('Different KVAF runtime checkout in compared arms')
    compared = compare_layout_receipts({name: value[11] for name, value in arms.items()}, candidate)
    if packets and (new[12][1] is None or (not reference and arms['row-deps'][12][1] is None)):
        raise ValueError('Missing per-query physical row reuse readback')
    print(f'G{group} CHECK TopK=PASS same_TopK=20/20 KV=PASS ordered_Index=PASS per_arm=320/320')
    for name, v in arms.items():
        print(f'G{group} {name} graph_mean/P50_us={v[3]:.2f}/{v[4]:.2f} '
              f'tail_mean/P50_us={v[8]:.2f}/{v[10]:.2f} LI_mean_us={v[13]:.2f}')
        if v[12] is not None:
            fast, overlap = v[12]
            reuse = ' cross_Q_physical_row_reuse_requests=' + ('unavailable' if overlap is None else f'{overlap}/320')
            print(f'G{group} {name} disjoint_fast_requests={fast}/320 general_requests={320-fast}/320{reuse}')
    for name in controls:
        old = arms[name]
        gain = [100 * (1 - new[i] / old[i]) for i in (3, 4)]
        drift = [max(abs(x), abs(y)) for x, y in zip(old[6], new[6])]
        tail = [old[i] - new[i] for i in (8, 10)]
        placement = pair_placement(old[2], new[2], local_baseline=True)
        same_rows, same_calls = compared[name]
        if placement != 'matched':
            gate = 'PLACEMENT_MISMATCH'
        elif not same_rows:
            gate = 'NO_IDENTICAL_REQUEST_PLANS'
        elif same_calls != 20:
            gate = 'PARTIAL_IDENTICAL_CALL_PLANS'
        elif all(g > d for g, d in zip(gain, drift)) and min(tail) > 0:
            gate = 'CROSS_RUN_GAIN' if reference else 'POSITIVE'
        elif min(gain) > 0:
            gate = 'INCONCLUSIVE'
        else:
            gate = 'KEEP_CONTROL'
        print(f'G{group} vs_{name} graph_saved_mean/P50_us={old[3]-new[3]:+.2f}/{old[4]-new[4]:+.2f} '
              f'tail_saved_mean/P50_us={tail[0]:+.2f}/{tail[1]:+.2f} '
              f'identical_ordered_request_plans={same_rows}/320 identical_full_call_plans={same_calls}/20 '
              f'placement={placement} timing_scope={"archived_control" if reference else "same_run"} {gate}')


def report_owner_candidate(root, group, label):
    data = load(root, group, label, row_dependencies=True)
    digests = data[1]
    if len(set(digests)) != 1 or not SHA256.fullmatch(digests[0]):
        raise ValueError(f'G{group}: final TopK differs between candidate invocations')
    compare_layout_receipts({label: data[11]}, label)
    print(f'G{group} CHECK TopK=PASS calls=20/20 KV=PASS ordered_Index=PASS rows=320/320')
    print(f'G{group} {label} graph_mean/P50_us={data[3]:.2f}/{data[4]:.2f} '
          f'tail_mean/P50_us={data[8]:.2f}/{data[10]:.2f}')


def main(root, group):
    for arm in ('short-lru', 'owner-pack'):
        if (root / f'{arm}.txt').is_file():
            return report_owner_candidate(root, group, arm)
    if (root / 'row-deps.txt').is_file():
        return report_row_dependencies(root, group)
    if (root / 'local.txt').is_file() or (root / 'offset.txt').is_file():
        return report_resolver(root, group)
    old, new = [load(root, group, name) for name in ('baseline', 'candidate')]
    if old[0] != new[0] or old[5] == new[5]:
        raise ValueError('Expected identical inputs and distinct binaries')
    placement = pair_placement(old[2], new[2], local_baseline=True)
    spans = defaultdict(deque)
    for digest, value in zip(old[1], old[7]):
        spans[digest].append(value)
    paired = [spans[digest].popleft() - value for digest, value in zip(new[1], new[7]) if spans[digest]]
    paired_saving = median(paired) if paired else float('nan')
    same = len(paired) == len(new[1])
    gains = [100 * (1 - new[i] / old[i]) for i in (3, 4)]
    guard = [max(abs(old[6][i]), abs(new[6][i])) for i in range(2)]
    verdict = 'KEEP_BASELINE'
    if min(gains) > 0:
        verdict = 'BOTH_FASTER' if all(g > d for g, d in zip(gains, guard)) else 'WITHIN_DRIFT'
    if placement != 'matched' or not same:
        verdict = 'NOT_COMPARABLE'
    print(f'G{group} {verdict} gain_mean/P50_pct={gains[0]:+.2f}/{gains[1]:+.2f} '
          f'same_TopK={len(paired)}/20 paired_P50_saved_us={paired_saving:+.2f} '
          f'placement={placement} owners=64/64', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]), int(sys.argv[2]))
