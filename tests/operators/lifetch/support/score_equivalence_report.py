"""Compare the Q1 handoff with and without the final cutoff tie selection."""
import argparse
from pathlib import Path

import numpy as np

from correctness import topk_fingerprint
from early_index_report import measurement
from li_restore_report import li_times, saved, stats
from owner_pack_tail import HANDOFF_REVISION
from peer_delta import pair_placement, read
from query_owner_report import compare_layout_receipts, load


def checked_outputs(root, group, current, serial, identity, oracle_path):
    folder, = (root / f'g{group}/logical-ties/results').glob('*/cases/diagnose/fused')
    receipt = read(folder / 'score_topk_check.json')
    source = receipt['native_reference']
    if (source['origin'] != 'completed-archive' or source['identity'] != identity
            or Path(source['archive_path']).resolve() != oracle_path.resolve()):
        raise ValueError('Different strong serial archive')
    return folder, source, measured_sets(folder, group, current, serial, receipt)


def measured_sets(folder, group, current, serial, receipt, *, expected_calls=20):
    if receipt['group'] != group or (receipt['calls'], receipt['queries']) != (expected_calls, expected_calls * 16 * 4):
        raise ValueError('Incomplete score-equivalent check')
    if receipt['serial_set_digest'] != topk_fingerprint(serial):
        raise ValueError('Strong serial oracle differs between runs')
    with np.load(folder / 'score_topk_ids.npz', allow_pickle=False) as raw:
        outputs = raw['fused']
    if (outputs.shape != (expected_calls, 16, 4, 2048) or outputs.dtype.kind not in 'iu'
            or [topk_fingerprint(call) for call in outputs] != current[1]):
        raise ValueError('Score-checked outputs differ from the measured and KV-checked outputs')
    exact = np.all(np.sort(outputs, axis=-1) == np.sort(serial, axis=-1), axis=-1)
    if (exact.sum(), exact.all(axis=(1, 2)).sum()) != (receipt['exact_queries'], receipt['exact_calls']):
        raise ValueError('Reported exact-token overlap differs from the saved outputs')

    return exact


def report(root, group, reference):
    if (reference / 'logical-ties.txt').read_text().strip() != HANDOFF_REVISION:
        raise ValueError('Expected the archived Q1 early Index handoff with exact tie selection')
    if any((r / 'physical_npu.txt').read_text().strip() != '4' for r in (root, reference)):
        raise ValueError('Expected measurements on physical NPU4')
    old, _, meta, checked, old_path = measurement(reference, group)
    if not checked[2]:
        raise ValueError('Archived exact-tie Q1 run failed strong serial comparison')
    current = load(root, group, 'logical-ties', row_dependencies=True)
    if old[0] != current[0] or old[14] != current[14] or old[5] == current[5]:
        raise ValueError('Expected identical replay inputs and KVAF, and distinct operator binaries')
    folder, source, exact = checked_outputs(root, group, current, checked[4], meta['identity'], old_path)

    print(f'G{group} CHECK TopK_score_vs_strong_serial=PASS calls=20/20 queries=1280/1280 '
          f'exact_ID_sets={exact.sum()}/1280 cutoff_tie_only_sets={1280-exact.sum()}/1280 '
          'KV=PASS ordered_Index=PASS rows=320/320')
    def times(data, directory):
        rows = read(directory / 'diagnose/round-1/full_attribution.json')['samples']
        li, final_li = li_times(rows)
        return dict(graph=data[3:5], LI=li, LI4=final_li, tail=(data[8], data[10]))
    before, after = times(old, old_path.parent), times(current, folder)
    print(f'G{group} q1-handoff-free-ties', *(f'{k}_mean/P50_us={stats(v)}' for k, v in after.items()))
    rows, calls = compare_layout_receipts({'old': old[11], 'new': current[11]}, 'new')['old']
    placement = pair_placement(old[2], current[2], local_baseline=True)
    runtime_match = meta['replay_identity'] == source['replay_identity']
    print(f'G{group} vs_q1_handoff_exact_ties historical_unpaired',
          *(f'{k}_archive_minus_candidate_mean/P50_us={saved(before[k], v)}' for k, v in after.items()),
          f'placement={placement} same_runtime={int(runtime_match)} '
          f'same_event_layout_rows={rows}/320 same_event_calls={calls}/20')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--group', type=int, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.group, args.reference)
