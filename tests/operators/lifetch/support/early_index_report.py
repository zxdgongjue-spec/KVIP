"""Compare a normal level-1 handoff run with its completed, uninstrumented parent."""
import sys
from math import isclose
from pathlib import Path
from typing import NamedTuple

import numpy as np

from li_restore_report import eager_failure, li_times, read_sets, report as report_li, saved, stats
from peer_delta import pair_placement, read
from query_owner_report import compare_layout_receipts, load


class Measurement(NamedTuple):
    data: tuple
    times: dict
    metadata: dict
    counts: dict
    stable_queries: int
    ok: bool
    verdict: str
    serial: np.ndarray
    path: Path


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
    times = dict(graph=data[3:5], tail=(data[8], data[10]), LI=li)
    return Measurement(data, times, comparison['native_reference'], counts, stability, ok, verdict, serial, path)


def compare(group, label, reference, candidate):
    old, new = reference.data, candidate.data
    if old[0] != new[0] or old[1] != new[1] or old[14] != new[14] or old[5] == new[5]:
        raise ValueError(f'G{group}: {label} replay inputs, TopK digests, KVAF root or binary identity incompatible')
    matched_rows, matched_calls = compare_layout_receipts(
        {'archive': old[11], 'candidate': new[11]}, 'candidate')['archive']
    placement = pair_placement(old[2], new[2], local_baseline=True)
    previous = reference.metadata.get('replay_identity', reference.metadata['identity'])
    current = candidate.metadata['replay_identity']
    changed = sorted(key for key in previous.keys() | current.keys() if previous.get(key) != current.get(key))
    detail = (f'placement={placement} runtime_identity_diff={",".join(changed) or "none"} '
              f'same_event_layout_rows={matched_rows}/320 same_event_calls={matched_calls}/20')
    comparable = placement == 'matched' and not changed and matched_calls == 20
    status = 'historical_unpaired' if comparable else 'observed_only'
    metric = 'saved' if comparable else 'archive_minus_candidate'
    plans = 'matched' if matched_calls == 20 else 'differ'
    print(f'G{group} vs_{label} {status}',
          *(f'{k}_{metric}_mean/P50_us={saved(reference.times[k], candidate.times[k])}' for k in reference.times),
          detail, f'physical_plans={plans}')


def g7_tail_summary(samples):
    tails = [s['accounting']['stages_us']['resolve_update'] for s in samples]
    q4 = [q for s in samples for r in s['algorithm']['requests']
          for q in r['queries'] if q['query'] == 4]
    if len(tails) != 20 or len(q4) != 320:
        raise ValueError('Expected 20 G7 calls and 320 Q4 descriptor slices')
    # 300 us lies in the gap between the two observed 24c619b9 tail ranges.
    above = sum(t > 300 for t in tails)
    conflicts = sum(q['final_safe'] < q['final_posted'] for q in q4)
    return (f'tail_P95_us={np.percentile(tails, 95):.2f} tail_gt_300us_calls={above}/20 '
            f'Q4_rows_with_prefetch_conflict={conflicts}/320')


def report(root, group):
    archive = Path((root / 'early-handoff-reference.txt').read_text().strip())
    previous_ref = root / 'previous-handoff-reference.txt'
    archives = [(archive, '24c619b9581aad43fd53c815928def26a1e19560')]
    if previous_ref.exists():
        archives.append((Path(previous_ref.read_text().strip()), '67b49a2748fde8f8779c4551410416596b35cf76'))
    for folder, revision in archives:
        if (folder / 'logical-ties.txt').read_text().strip() != revision:
            raise ValueError(f'Expected reference {revision[:8]} at {folder}')
    if any((r / 'physical_npu.txt').read_text().strip() != '4' for r in [root] + [r for r, _ in archives]):
        raise ValueError('Expected measurements on physical NPU4')
    if eager_failure(root / f'g{group}/logical-ties/results'):
        return report_li(root, group, 'logical-ties')

    references = [(rev[:8] + '_archive', measurement(folder, group)) for folder, rev in archives]
    origin = references[0][1]
    candidate = measurement(root, group)
    for run in [r for _, r in references[1:]] + [candidate]:
        meta = run.metadata
        if (meta['origin'] != 'completed-archive' or Path(meta['archive_path']).resolve() != origin.path.resolve()
                or meta['identity'] != origin.metadata['identity'] or not np.array_equal(origin.serial, run.serial)):
            raise ValueError(f'G{group}: saved strong-serial reference differs from {origin.path}')
    if any(not run.ok for _, run in references):
        raise ValueError(f'G{group}: a reference run did not match the strong-serial sets')
    print(f'G{group} CHECK TopK_set_vs_strong_serial={candidate.verdict} calls={candidate.counts["matched_calls"]}/20 '
          f'queries={candidate.counts["matched_queries"]}/1280 serial_repeat_queries={candidate.stable_queries}/64 '
          'KV=PASS ordered_Index=PASS rows=320/320')
    if not candidate.ok:
        print(f'G{group} TOPK_MISMATCH details={candidate.path}')
        return 3

    label = 'prefetch-cq-overlap' if previous_ref.exists() else 'early-index-handoff'
    for name, run in [*references, (label, candidate)]:
        print(f'G{group} {name}', *(f'{k}_mean/P50_us={stats(v)}' for k, v in run.times.items()))
        if group == 7 and previous_ref.exists():
            profile = read(run.path.parent / 'diagnose/round-1/full_attribution.json')
            print(f'G7 {name}', g7_tail_summary(profile['samples']))
    for name, run in reversed(references):
        compare(group, name, run, candidate)
    return 0


if __name__ == '__main__':
    sys.exit(report(Path(sys.argv[1]), int(sys.argv[2])))
