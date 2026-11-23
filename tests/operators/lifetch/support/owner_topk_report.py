"""Compare a three-LI owner pipeline with the archived standalone final LI + tail."""
import argparse
import hashlib
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median

import numpy as np

from early_index_report import measurement
from li_restore_archive import dp_capture_id
from li_restore_report import saved, serial_graph_line, stats
from owner_pack_tail import HANDOFF_REVISION
from native_li_reference import check_positions
from peer_delta import pair_placement, read
from query_owner_report import compare_layout_receipts, load
from ranked_integration_sets import INPUT_AND_RESULTS
from score_equivalence_report import checked_outputs, measured_sets

FREE_TIES = '1040d7133028ff4cd36904bef0393783be884877'
# Host20 NPU0, four warmups and 20 profiler calls, primary original-fusion run.
# Root help-772293d4-fecf79e510598355a1ee; only use on its verified capture.
ORIGINAL_GRAPH = {2: (322.88, 315.59), 4: (355.32, 345.78), 10: (365.35, 356.88),
                  17: (395.96, 387.09), 7: (559.54, 533.88)}


def times(folder, graph, *, joined):
    rows = read(folder / 'diagnose/round-1/full_attribution.json')['samples']
    if len(rows) != 20:
        raise ValueError('Expected 20 profiled invocations')
    prefix, finals = [], []
    for index, row in enumerate(rows):
        sample = row['accounting']
        groups = sample['intervals_us']
        li, tail = sorted(groups['li']), groups['resolve_update']
        expected_li = 3 if joined else 4
        if (sample['invocation'] != index or len(li) != expected_li or len(tail) != 1 or
                (joined and sample.get('owner_topk_kernel') != 'v9_owner_topk_resolve_update')):
            raise ValueError(f'Invocation {index}: expected {expected_li} LI phases and one '
                             f'{"owner TopK" if joined else "original"} tail')
        intervals = li + tail
        if (any(not isfinite(a) or not isfinite(b) or a >= b for a, b in intervals)
                or any(y[0] < x[1] - 1e-5 for x, y in zip(intervals, intervals[1:]))):
            raise ValueError(f'Unexpected kernel order or overlap in invocation {index}')
        if (not isclose(sum(b-a for a, b in li), sample['stages_us']['li'], abs_tol=0.025)
                or not isclose(tail[0][1]-tail[0][0], sample['stages_us']['resolve_update'], abs_tol=0.025)):
            raise ValueError(f'Stage totals disagree with trace intervals in invocation {index}')
        prefix.append(sum(b-a for a, b in li[:3]))
        # Old interval includes the gap from LI4 completion to resolver launch.
        # Compute the *per-invocation* span before taking the median.
        final_start = tail[0][0] if joined else li[3][0]
        finals.append(tail[0][1] - final_start)
    return dict(graph=graph, LI1_to_LI3=(mean(prefix), median(prefix)),
                final_topk_through_tail=(mean(finals), median(finals)))


def native_g2_check(folder, new):
    receipt = read(folder / 'score_topk_check.json')
    source = receipt['native_reference']
    identity = source['identity']
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    paths = new[0]['source_paths']
    capture = dp_capture_id(2, new[0])
    merged = len(paths) == 8 and len(set(paths)) == 1
    if (source['origin'] not in ('fresh', 'validated-cache') or source['cache_key'] != key
            or identity['operator'] != 'torch_npu.npu_quant_lightning_indexer' or identity['visible'] != '4'
            or source['layout_query'] != 'BSND' or source['layout_key'] != 'PA_BSND'
            or source['sparse_count'] != 2048 or source['input_paths'] != paths
            or (capture != INPUT_AND_RESULTS[2][0] and not (capture is None and merged))):
        raise ValueError('Expected the logical native LI on the archived G2 DP0-7 capture and NPU4')
    with np.load(folder / 'score_topk_ids.npz', allow_pickle=False) as archive:
        serial = archive['strong_serial']
    ends = np.asarray(source['effective_native_ends'])
    if ends.shape != (16, 4):
        raise ValueError('Incomplete native G2 query bounds')
    check_positions(serial, ends)
    compare_layout_receipts({'new': new[11]}, 'new')
    return measured_sets(folder, 2, new, serial, receipt), source['origin']


def archived_parent_check(root, group, reference, new, after):
    oracle = Path((reference / 'exact-handoff-reference.txt').read_text().strip())
    if ((reference / 'logical-ties.txt').read_text().strip() != FREE_TIES
            or (oracle / 'logical-ties.txt').read_text().strip() != HANDOFF_REVISION
            or any((r / 'physical_npu.txt').read_text().strip() != '4' for r in (root, reference, oracle))):
        raise ValueError('Expected B24 NPU4 score-equivalent parent and its exact-serial archive')
    exact, _, metadata, checked, oracle_path = measurement(oracle, group)
    if not checked[2]:
        raise ValueError('Archived serial oracle failed exact-token validation')
    old = load(reference, group, 'logical-ties', row_dependencies=True)
    if old[0] != new[0] or old[0] != exact[0] or old[14] != new[14] or old[5] == new[5]:
        raise ValueError('Expected identical inputs and KVAF, and different native binaries')
    check_args = checked[4], metadata['identity'], oracle_path
    old_folder, old_source, _ = checked_outputs(reference, group, old, *check_args)
    _, new_source, matches = checked_outputs(root, group, new, *check_args)
    before = times(old_folder, old[3:5], joined=False)
    rows, calls = compare_layout_receipts({'old': old[11], 'new': new[11]}, 'new')['old']
    fields = ' '.join(f'{k}_archive_minus_candidate_mean/P50_us={saved(before[k], v)}' for k, v in after.items())
    comparison = (f'G{group} vs_q1_handoff_free_ties historical_unpaired {fields} '
                  f'placement={pair_placement(old[2], new[2], local_baseline=True)} '
                  f'same_runtime={int(old_source["replay_identity"] == new_source["replay_identity"])} '
                  f'same_event_layout_rows={rows}/320 same_event_calls={calls}/20')
    return matches, comparison


def report(root, group, reference):
    if (root / 'physical_npu.txt').read_text().strip() != '4':
        raise ValueError('Expected candidate replay on B24 NPU4')
    new = load(root, group, 'logical-ties', row_dependencies=True)
    folder, = (root / f'g{group}/logical-ties/results').glob('*/cases/diagnose/fused')
    after = times(folder, new[3:5], joined=True)
    source = 'completed-archive'
    if group == 2:
        matches, source = native_g2_check(folder, new)
        comparison = 'G2 vs_q1_handoff_free_ties UNAVAILABLE missing_G2_measurement'
    else:
        matches, comparison = archived_parent_check(root, group, reference, new, after)
    print(f'G{group} CHECK TopK_score_vs_strong_serial=PASS calls=20/20 queries=1280/1280 '
          f'exact_ID_sets={matches.sum()}/1280 cutoff_tie_only_sets={matches.size-matches.sum()}/1280 '
          f'KV=PASS ordered_Index=PASS rows=320/320 standalone_LI_per_call=3 fused_owner_topk_tail_per_call=1 '
          f'native_LI_reference={source}')
    print(f'G{group} owner-topk', *(f'{k}_mean/P50_us={stats(v)}' for k, v in after.items()))
    print(comparison)
    if group == 10:
        from strong_serial_g10 import comparison_line
        print(comparison_line(root.parent, new[0], after['graph']))
    else:
        print(serial_graph_line(group, new[0], after['graph']))
    if dp_capture_id(group, new[0]) == INPUT_AND_RESULTS[group][0]:
        print(f'G{group} vs_original_fusion historical_unpaired_host20_npu0 '
              f'graph_saved_mean/P50_us={saved(ORIGINAL_GRAPH[group], after["graph"])}')
    else:
        print(f'G{group} vs_original_fusion UNVERIFIED capture_identity_differs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('group', type=int, choices=sorted(ORIGINAL_GRAPH))
    parser.add_argument('reference', type=Path)
    args = parser.parse_args()
    report(args.root, args.group, args.reference)
