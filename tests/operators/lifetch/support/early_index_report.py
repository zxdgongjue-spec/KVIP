"""Compare a normal level-1 handoff run with its completed, uninstrumented parent."""
import sys
from math import isclose
from pathlib import Path

import numpy as np

from li_restore_report import eager_failure, li_times, read_sets, report as report_li, saved, stats
from peer_delta import pair_placement, read
from query_owner_report import compare_layout_receipts, load


def measurement(root, group):
    data = load(root, group, 'logical-ties', row_dependencies=True)
    results = root / f'g{group}/logical-ties/results'
    path, = results.glob('*/cases/diagnose/fused/native_li_compare.json')
    comparison = read(path)
    if (comparison['group'], comparison['phase'], comparison['measured_calls'],
            comparison['queries_per_call']) != (group, 'measured', 20, 64):
        raise ValueError(f'G{group}: incomplete measured native comparison in {path}')
    counts, _, _, stability, ok, verdict, serial = read_sets(path.parent, comparison, 20, data[1])
    samples = read(path.parent / 'diagnose/round-1/full_attribution.json')['samples']
    li, _ = li_times(samples)
    if not isclose(li[0], data[13], abs_tol=0.025):
        raise ValueError(f'G{group}: LI mean differs from the four measured intervals')
    return data, dict(graph=data[3:5], tail=(data[8], data[10]), LI=li), comparison['native_reference'], (
        counts, stability, ok, verdict, serial), path


def report(root, group):
    archive = Path((root / 'early-handoff-reference.txt').read_text().strip())
    if (archive / 'logical-ties.txt').read_text().strip() != '24c619b9581aad43fd53c815928def26a1e19560':
        raise ValueError('Expected the completed 24c619b9 reference')
    if any((r / 'physical_npu.txt').read_text().strip() != '4' for r in (archive, root)):
        raise ValueError('Expected two measurements on physical NPU4')
    if eager_failure(root / f'g{group}/logical-ties/results'):
        return report_li(root, group, 'logical-ties')

    old, old_times, archive_meta, archive_check, path = measurement(archive, group)
    new, new_times, meta, check, candidate_path = measurement(root, group)
    counts, stability, ok, verdict, serial = check
    if (meta['origin'] != 'completed-archive' or Path(meta['archive_path']).resolve() != path.resolve()
            or meta['identity'] != archive_meta['identity'] or not archive_check[2]
            or not np.array_equal(archive_check[4], serial)):
        raise ValueError(f'G{group}: saved strong-serial reference differs from {path}')
    print(f'G{group} CHECK TopK_set_vs_strong_serial={verdict} calls={counts["matched_calls"]}/20 '
          f'queries={counts["matched_queries"]}/1280 serial_repeat_queries={stability}/64 '
          'KV=PASS ordered_Index=PASS rows=320/320')
    if not ok:
        print(f'G{group} TOPK_MISMATCH details={candidate_path}')
        return 3

    if old[0] != new[0] or old[1] != new[1] or old[14] != new[14] or old[5] == new[5]:
        raise ValueError(f'G{group}: replay inputs, TopK digests, KVAF root or binary identity incompatible')
    matched_rows, matched_calls = compare_layout_receipts(
        {'archive': old[11], 'candidate': new[11]}, 'candidate')['archive']
    placement = pair_placement(old[2], new[2], local_baseline=True)
    changed = sorted(key for key in meta['identity'] if meta['identity'][key] != meta['replay_identity'].get(key))
    for label, times in [('24c619b9_archive', old_times), ('early-index-handoff', new_times)]:
        print(f'G{group} {label}', *(f'{k}_mean/P50_us={stats(v)}' for k, v in times.items()))
    detail = (f'placement={placement} runtime_identity_diff={",".join(changed) or "none"} '
              f'same_event_layout_rows={matched_rows}/320 same_event_calls={matched_calls}/20')
    if placement != 'matched' or changed:
        print(f'G{group} vs_24c619b9_archive UNCOMPARABLE {detail}')
    else:
        print(f'G{group} vs_24c619b9_archive historical_unpaired',
              *(f'{k}_saved_mean/P50_us={saved(old_times[k], new_times[k])}' for k in old_times), detail)
    return 0


if __name__ == '__main__':
    sys.exit(report(Path(sys.argv[1]), int(sys.argv[2])))
