"""Exact per-query native set agreement and the four LI kernels in the measured graph."""
import sys
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median

import numpy as np

from correctness import topk_fingerprint
from diagnostics import union
from li_restore_archive import archive_lines
from peer_delta import pair_placement, read
from query_owner_report import compare_layout_receipts, load


def li_times(samples):
    all_li, final_li = [], []
    for sample in samples:
        accounting = sample['accounting']
        total = accounting['stages_us']['li']
        intervals = sorted(accounting['intervals_us']['li'])
        if len(intervals) != 4 or not all(isfinite(b) and isfinite(e) and e > b for b, e in intervals):
            raise ValueError('Expected four finite LI kernel intervals')
        if not isfinite(total) or total <= 0 or not isclose(
                sum(e - b for b, e in union(intervals)), total, rel_tol=1e-6, abs_tol=0.025):
            raise ValueError('Four LI intervals do not reconcile with the per-call stage time')
        all_li.append(total)
        final_li.append(intervals[-1][1] - intervals[-1][0])
    return (mean(all_li), median(all_li)), (mean(final_li), median(final_li))


def stats(pair):
    return f'{pair[0]:.2f}/{pair[1]:.2f}'


def owner_delta(root, group, current, li_pair, li4_pair):
    previous = root.parent / 'fused.tbs07g'
    if not previous.exists():
        return
    owner = '36d92c3e663e5164af54e7761187f3204e76fe05'
    if ((previous / 'owner-pack.txt').read_text().strip() != owner
            or (previous / 'physical_npu.txt').read_text() != (root / 'physical_npu.txt').read_text()):
        print(f'G{group} vs_owner_pack reference_revision_or_device_differs')
        return
    prior = load(previous, group, 'owner-pack', row_dependencies=True)
    if prior[0] != current[0]:
        print(f'G{group} vs_owner_pack archived_input_metadata_differs')
        return
    profile, = (previous / f'g{group}/owner-pack/results').glob('**/fused/diagnose/round-1/full_attribution.json')
    old_li, old_li4 = li_times(read(profile)['samples'])

    def saved(old, new):
        return '/'.join(f'{a-b:+.2f}' for a, b in zip(old, new))

    placement = pair_placement(prior[2], current[2], local_baseline=True)
    print(f'G{group} vs_owner_pack LI_saved_mean/P50_us={saved(old_li, li_pair)} '
          f'LI4_saved_mean/P50_us={saved(old_li4, li4_pair)} '
          f'graph_saved_mean/P50_us={saved(prior[3:5], current[3:5])} '
          f'tail_saved_mean/P50_us={saved((prior[8], prior[10]), (current[8], current[10]))} '
          f'placement={placement}')


def report(root, group):
    label = 'li-restore'
    data = load(root, group, label, row_dependencies=True)
    compare_layout_receipts({label: data[11]}, label)
    path, = (root / f'g{group}' / label / 'results').glob('**/fused/path_result.json')
    folder = path.parent
    comparison = read(folder / 'native_li_compare.json')
    if (comparison['group'], comparison['measured_calls'], comparison['queries_per_call']) != (group, 20, 64):
        raise ValueError('Native TopK check does not cover the measured group')
    samples = read(folder / 'diagnose/round-1/full_attribution.json')['samples']
    li_pair, li4_pair = li_times(samples)
    if not isclose(li_pair[0], data[13], rel_tol=1e-6, abs_tol=0.025):
        raise ValueError('Profile and graph summary disagree on mean LI')
    with np.load(folder / comparison['raw_ids'], allow_pickle=False) as raw:
        outputs = raw['fused']
        if outputs.shape != (20, 16, 4, 2048) or [topk_fingerprint(call) for call in outputs] != data[1]:
            raise ValueError('Native comparison IDs differ from the measured fused outputs')
        serial_ids = raw['strong_serial']
        if serial_ids.shape != (16, 4, 2048):
            raise ValueError('Incomplete native LI reference')
        correct = comparison['comparisons']['strong_serial']
        physical = comparison['comparisons']['same_layout_native']
        observed_stability = np.all(np.sort(serial_ids, axis=-1) == np.sort(raw['strong_serial_repeat'], axis=-1),
                                    axis=-1).sum()
        sorted_outputs = np.sort(outputs, axis=-1)
        for name, counts in comparison['comparisons'].items():
            reference = raw[name]
            if reference.shape != serial_ids.shape:
                raise ValueError(f'{name} has the wrong TopK shape')
            reference = np.sort(reference, axis=-1)
            if np.any(np.diff(reference, axis=-1) <= 0):
                raise ValueError(f'{name} has duplicate IDs')
            q_match = np.all(sorted_outputs == reference, axis=-1)
            passes = q_match.all(axis=(1, 2)).sum()
            matches = q_match.sum()
            if (counts['matched_calls'], counts['matched_queries'], counts['total_queries']) != (passes, matches, 1280):
                raise ValueError(f'{name} comparison does not agree with the saved IDs')
    domains = comparison['domains']
    stable = comparison['serial_reference_stability']['stable_queries']
    if observed_stability != stable:
        raise ValueError('Serial repeat stability count disagrees with the raw native outputs')
    ok = correct['matched_queries'] == 1280 and stable == 64
    verdict = 'UNSTABLE_REFERENCE' if stable != 64 else ('PASS' if ok else 'FAIL')
    print(f'G{group} CHECK TopK_set_vs_strong_serial={verdict} calls={correct["matched_calls"]}/20 '
          f'queries={correct["matched_queries"]}/1280 serial_repeat_queries={stable}/64 '
          f'KV=PASS ordered_Index=PASS rows=320/320')
    print(f'G{group} ATTRIBUTION same_layout_native_queries={physical["matched_queries"]}/1280 '
          f'strong_domain_match={domains["strong_equals_fused_queries"]}/64 '
          f'same_layout_domain_match={domains["same_layout_equals_fused_queries"]}/64 '
          f'captured_cutoff_only_differences={correct["cutoff_ties_only_mismatches"]}')
    if stable != 64:
        print(f'G{group} NATIVE_REPEAT_DIFFERENCE details={folder / "native_li_compare.json"}')
    if correct['matched_queries'] != 1280:
        first = comparison['samples']['strong_serial'][0]
        print(f'G{group} FIRST_DIFFERENCE call={first["call"]+1} row={first["row"]} Q={first["query"]} '
              f'missing={first["missing"]["examples"][:3]} extra={first["extra"]["examples"][:3]} '
              f'cutoff_key={first["fused_cutoff_key"]} details={folder / "native_li_compare.json"}')
    print(f'G{group} li-restore LI_mean/P50_us={stats(li_pair)} LI4_mean/P50_us={stats(li4_pair)} '
          f'graph_mean/P50_us={stats((data[3], data[4]))} tail_mean/P50_us={stats((data[8], data[10]))}')
    try:
        owner_delta(root, group, data, li_pair, li4_pair)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'G{group} vs_owner_pack archive_unavailable={error}')
    for line in archive_lines(root, group, data[0], serial_ids, li_pair, li4_pair):
        print(line)
    return 0 if ok else 3


if __name__ == '__main__':
    sys.exit(report(Path(sys.argv[1]), int(sys.argv[2])))
