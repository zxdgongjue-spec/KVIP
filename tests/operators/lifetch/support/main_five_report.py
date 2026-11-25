"""Compare the B24 parallel terminal with the same captured inputs on the per-Q TopK archive."""
import argparse
import hashlib
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median

import numpy as np

MAIN = '609fa26fd715f9409bbe74e1b2d3685c58dfb0cf'
OWNER = '888db6b1603d22f4142bd2ccffe0518611856a31'
GROUPS = (4, 10, 17, 7, 2)


def read(path):
    return json.loads(path.read_text())


def case_folder(root, group, arm):
    paths = list((root / f'g{group}' / arm / 'results').glob('*/cases/diagnose/fused/path_result.json'))
    if len(paths) != 1:
        raise ValueError(f'G{group} {arm}: expected one completed path in {root}, found {len(paths)}')
    return paths[0].parent


def archived_oracle(owner_root, group):
    """Load the saved native IDs, including G2's self-contained serial reference."""
    owner_root = Path(owner_root)
    if (owner_root / 'native_score_oracle.json').is_file():
        from feature_score_oracle import load_oracle
        return load_oracle(owner_root, group)
    if ((owner_root / 'logical-ties.txt').read_text().strip() != OWNER
            or (owner_root / 'physical_npu.txt').read_text().strip() != '4'):
        raise ValueError('Expected the archived per-Q final TopK version measured on physical NPU4')
    folder = case_folder(owner_root, group, 'logical-ties')
    receipt = read(folder / 'score_topk_check.json')
    src = receipt['native_reference']
    if ((receipt['group'], receipt['calls'], receipt['queries']) != (group, 20, 1280)
            or src['identity']['visible'] != '4'
            or src['identity']['operator'] != 'torch_npu.npu_quant_lightning_indexer'):
        raise ValueError(f'G{group}: wrong native TopK oracle or incomplete owner archive')
    if group == 2:
        if src['origin'] not in ('fresh', 'validated-cache'):
            raise ValueError('G2: missing captured native reference')
        ids_path = folder / 'score_topk_ids.npz'
    else:
        path = Path(src['archive_path'])
        archived = read(path)
        if (src['origin'] != 'completed-archive' or src['identity'] != archived['native_reference']['identity']
                or (archived['group'], archived['phase'], archived['measured_calls']) != (group, 'measured', 20)):
            raise ValueError(f'G{group}: serial source does not match the archived owner receipt')
        ids_path = path.parent / archived['raw_ids']
    with np.load(ids_path, allow_pickle=False) as raw:
        ids = raw['strong_serial']
    ends = np.asarray(src['effective_native_ends'])
    if (ids.shape != (16, 4, 2048) or ids.dtype.kind not in 'iu' or ends.shape != (16, 4)
            or (ids < 0).any() or (ids >= ends[..., None]).any()):
        raise ValueError(f'G{group}: invalid native TopK dimensions or token bounds')
    ordered = np.sort(ids, axis=-1)
    # Same canonical little-endian set fingerprint as the archive writer. The worker
    # imports this helper alongside mainline's own module named "correctness".
    digest = hashlib.sha256(ordered.astype('<i8').tobytes()).hexdigest()
    if digest != receipt['serial_set_digest'] or (np.diff(ordered, axis=-1) == 0).any():
        raise ValueError(f'G{group}: native TopK IDs differ from the completed owner archive')
    return ids, receipt


def main_times(folder, timing, *, expected_calls=20):
    profile = read(folder / 'diagnose/round-1/full_attribution.json')
    samples = profile['samples']
    if (profile['graph_calls'] != expected_calls or len(samples) != expected_calls
            or timing['graph_calls'] != expected_calls or len(timing['samples_us']) != expected_calls
            or timing['timing_protocol'] != 'one_profiled_graph' or timing['measured_graphs'] != 1):
        raise ValueError(f'Expected one graph with {expected_calls} independent measured calls')
    before, after, spans = [], [], []
    for i, sample in enumerate(samples):
        item = sample['accounting']
        groups = item['intervals_us']
        if (item['invocation'] != i or set(groups) != {'li', 'topk_resolve_update'}
                or len(groups['li']) != 3 or len(groups['topk_resolve_update']) != 1):
            raise ValueError(f'Call {i}: expected three LI launches and one mainline parallel terminal')
        intervals = sorted(groups['li']) + groups['topk_resolve_update']
        if (any(not (isfinite(a) and isfinite(b) and a < b) for a, b in intervals)
                or any(b > c for (_, b), (c, _) in zip(intervals, intervals[1:]))):
            raise ValueError(f'Call {i}: invalid LI or terminal interval')
        prefix = sum(b-a for a, b in intervals[:3])
        tail = intervals[-1][1] - intervals[-1][0]
        span = intervals[-1][1] - intervals[0][0]
        stages = item['stages_us']
        if not all(isclose(x, y, abs_tol=0.025) for x, y in (
                (prefix, stages['li']), (tail, stages['topk_resolve_update']), (span, sample['us']))):
            raise ValueError(f'Call {i}: stage totals differ from the trace')
        before.append(prefix)
        after.append(tail)
        spans.append(span)
    graph = (timing['event_mean_us'], median(spans))
    if (not isfinite(graph[0]) or graph[0] <= 0 or not isclose(graph[0], timing['mean_us'])
            or not isclose(graph[1], timing['p50_us'], abs_tol=0.025)
            or not isclose(graph[0]*expected_calls, profile['graph_accounting']['event_us'], abs_tol=0.025)
            or not np.allclose(spans, timing['samples_us'], rtol=1e-6, atol=0.025)):
        raise ValueError('Graph event mean or per-call spans disagree with the measured profile')
    return dict(graph=graph, LI1_to_LI3=(mean(before), median(before)),
                final_topk_through_tail=(mean(after), median(after)))


def measured_parallel(root, group, arm, revision, owner, native, old_receipt, archived_inputs, *,
                      verify_library, timing_reader=None, capture_meta_name=None, expected_calls=20):
    from li_restore_archive import check_snapshot
    from ranked_integration_sets import INPUT_AND_RESULTS
    from score_equivalence_report import measured_sets

    folder = case_folder(root, group, arm)
    out = read(folder / 'path_result.json')
    config = read(folder.parent / 'config.json')
    inputs = read(folder.parent / 'replay_input.json')
    capture = read(folder.parent / capture_meta_name) if capture_meta_name else inputs
    if capture != archived_inputs:
        check_snapshot(group, INPUT_AND_RESULTS[group][0], archived_inputs, capture)
    if out['input_paths'] != inputs['source_paths'] or inputs['geometry'] != out['geometry']:
        raise ValueError(f'G{group}: output capture metadata mismatch')
    expected = dict(group=group, transport='urma', input='real', device=0, physical_npu=4, warmup=4,
                    iterations=expected_calls, intermediate_fetches=4, prefetch_tokens=1024, prefetch_rank_start=1024,
                    fused_only=True, production_revision=revision)
    differences = [name for name, value in expected.items() if config.get(name) != value]
    if differences:
        raise ValueError(f'G{group}: unexpected replay settings: {differences}')
    if ((out['path'], out['transport'], out['runtime_workers']) != ('fused', 'urma', 40)
            or out['correctness'] != 'PASS' or out['failures']):
        raise ValueError(f'G{group}: mainline device validation or transport setup failed')
    proof = read(folder / 'mainline_checks.json')
    expected_counts = (expected_calls, expected_calls * 16)
    if (proof['production_revision'] != revision or proof['production_terminal'] is not True
            or proof['profiled_terminal_kernel'] != 'v9_parallel_terminal'
            or (proof['measured_calls'], proof['measured_kv_index_rows']) != expected_counts
            or (proof['ordered_diagnostic_calls'], proof['ordered_diagnostic_rows']) != expected_counts):
        raise ValueError(f'G{group}: incomplete mainline kernel, measured outputs or ordered Index evidence')
    sha = proof['library_sha256']
    if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError(f'G{group}: missing build digest')
    if verify_library:
        digest = hashlib.sha256()
        with Path(proof['loaded_library']).open('rb') as library:
            for block in iter(lambda: library.read(2 ** 20), b''):
                digest.update(block)
        if digest.hexdigest() != sha:
            raise ValueError(f'G{group}: loaded binary differs from the recorded build')
    receipt = read(folder / 'score_topk_check.json')
    source = receipt['native_reference']
    expected_origin = 'native-only-oracle' if old_receipt.get('reference_kind') == 'native-only' else 'completed-owner-archive'
    if (source['origin'] != expected_origin or Path(source['archive_path']).resolve() != owner.resolve()
            or source['identity'] != old_receipt['native_reference']['identity']):
        raise ValueError(f'G{group}: mainline and owner archive used different native oracles')
    if proof['capture_operand_hashes'] != source['identity']['inputs']:
        raise ValueError(f'G{group}: measured input differs from the archived native operands')
    exact = measured_sets(folder, group, (inputs, out['topk_set_digests']), native, receipt,
                          expected_calls=expected_calls)
    if timing_reader is None:
        new_times = main_times(folder, out['all_rounds'], expected_calls=expected_calls)
    else:
        new_times = timing_reader(folder, out['all_rounds'])
    thp = read(root / f'g{group}/{arm}/thp.json')
    if (thp.get('status') != 'measured' or thp.get('page_mode') != 'thp'
            or thp['bytes'] <= 0):
        raise ValueError(f'G{group}: THP or registered DDR range differs from the candidate archive')
    return new_times, exact, thp, proof, config


def report(root, group, owner):
    # Delay reader imports: the upstream device worker uses archived_oracle alone,
    # and must import upstream modules under their native Python names.
    from li_restore_report import saved, serial_graph_line, stats
    from owner_topk_report import times
    from peer_delta import pair_placement
    from query_owner_report import load
    from score_equivalence_report import measured_sets
    from strong_serial_g10 import comparison_line

    if ((root / 'main.txt').read_text().strip() != MAIN or (root / 'physical_npu.txt').read_text().strip() != '4'
            or (root / 'kvaf.txt').read_text().strip() != 'a2857b460a74704114fd30a849ae38666cc2c372'):
        raise ValueError('Expected the B24 NPU4 upstream parallel terminal with the shared KVAF revision')
    native, old_receipt = archived_oracle(owner, group)
    old = load(owner, group, 'logical-ties', row_dependencies=True)
    old_folder = case_folder(owner, group, 'logical-ties')
    measured_sets(old_folder, group, old, native, old_receipt)
    old_times = times(old_folder, old[3:5], joined=True)
    new_times, exact, thp, proof, config = measured_parallel(
        root, group, 'mainline', MAIN, owner, native, old_receipt, old[0], verify_library=True)
    if proof['library_sha256'] == old[5] or thp['bytes'] != old[2]['bytes'] or old[2]['page_mode'] != 'thp':
        raise ValueError(f'G{group}: expected different native operators and a matched registered DDR range')
    inputs = read(case_folder(root, group, 'mainline').parent/'replay_input.json')
    placement = pair_placement(old[2], thp, local_baseline=True)
    print(f'G{group} CHECK TopK_score_vs_strong_serial=PASS calls=20/20 queries=1280/1280 '
          f'exact_ID_sets={exact.sum()}/1280 cutoff_tie_only_sets={exact.size-exact.sum()}/1280 '
          f'KV_and_Index_state=PASS measured_rows=320/320 ordered_Index=PASS '
          f'separate_diagnostic_calls={proof["ordered_diagnostic_calls"]} '
          f'separate_diagnostic_rows={proof["ordered_diagnostic_rows"]}')
    for name, values in [('mainline_parallel_terminal', new_times), ('archived_per_Q_final_TopK', old_times)]:
        print(f'G{group} {name}', *(f'{k}_mean/P50_us={stats(v)}' for k, v in values.items()))
    archive_ref = old_receipt['native_reference']
    previous_runtime = archive_ref.get('replay_identity', archive_ref['identity'])
    changed = [name for name in ('host', 'npu', 'visible', 'torch', 'torch_npu', 'cann')
               if proof['runtime'][name] != previous_runtime[name]]
    if Path(config['kvaf_root']).resolve() != Path(old[14]).resolve():
        changed.append('kvaf_root')
    if placement != 'matched' or changed:
        print(f'G{group} vs_per_Q_final_TopK UNVERIFIED placement={placement} '
              f'runtime_changed={",".join(changed) or "none"}')
    else:
        gains = tuple(100*(a-b)/a for a, b in zip(old_times['graph'], new_times['graph']))
        print(f'G{group} vs_per_Q_final_TopK historical_unpaired_host24_npu4',
              *(f'{k}_saved_mean/P50_us={saved(old_times[k], v)}' for k, v in new_times.items()),
              f'graph_gain_mean/P50_pct={gains[0]:+.2f}/{gains[1]:+.2f} placement={placement} archive={owner.name}')
    # Strong serial is a separate historical comparison. Its source archive for
    # G10 resides in the same baseline area and may use the merged trace export.
    print(comparison_line(owner.parent, inputs, new_times['graph']) if group == 10
          else serial_graph_line(group, inputs, new_times['graph']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('group', type=int, choices=GROUPS)
    parser.add_argument('owner_archive', type=Path)
    args = parser.parse_args()
    report(args.root, args.group, args.owner_archive)
