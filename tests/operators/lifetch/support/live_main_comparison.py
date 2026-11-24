"""Compare complete fusion stacks, freshly measured in two separate runs on B24 NPU4."""
import argparse
from pathlib import Path
from statistics import mean

from li_restore_report import saved, stats
from main_five_report import archived_oracle, case_folder, measured_parallel, read
from peer_delta import pair_placement
from reporting import atomic_json


CANDIDATE = '20915a78b682378ea60889afbcd14b9a21123913'
UPSTREAM = 'd6cfd392f722dd3f96ef6e05175f503aac511dbb'
LEGACY_SDK = 'a2857b460a74704114fd30a849ae38666cc2c372'
UNIFIED_SDK = '813f1750a50b7e6c92d5f5397f2fbc05befef6da'
GROUPS = (10, 4, 17, 7, 2)


def marker(root, name):
    return (root / f'{name}.txt').read_text().strip()


def gain_pct(reference, candidate):
    return [100 * (a-b) / a for a, b in zip(reference, candidate)]


def compare(candidate, main, reader, output):
    for root, tag, revision, sdk in ((candidate, 'candidate', CANDIDATE, LEGACY_SDK),
                                     (main, 'main', UPSTREAM, UNIFIED_SDK)):
        expected = {tag: revision, 'reader': reader, 'physical_npu': '4', 'kvaf': sdk}
        for name, value in expected.items():
            if marker(root, name) != value:
                raise ValueError(f'{root}: wrong {name} for this live comparison')
    for name in ('host', 'boot_id'):
        if not marker(main, name) or marker(main, name) != marker(candidate, name):
            raise ValueError(f'The live runs have different {name}')
    area = candidate.parent
    results = []
    for group in GROUPS:
        owner = area / ('fused.gbxKC9' if group == 2 else 'fused.N7n8ZP')
        ids, receipt = archived_oracle(owner, group)
        inputs = read(case_folder(owner, group, 'logical-ties').parent / 'replay_input.json')
        arms = []
        for root, arm, revision in ((main, 'mainline', UPSTREAM), (candidate, 'li-window', CANDIDATE)):
            arms.append(measured_parallel(root, group, arm, revision, owner, ids, receipt, inputs,
                                          verify_library=True))
        (old, _, main_ddr, main_proof, main_cfg), (new, _, cand_ddr, cand_proof, cand_cfg) = arms
        if (main_proof['capture_operand_hashes'] != cand_proof['capture_operand_hashes']
                or main_proof['runtime'] != cand_proof['runtime']
                or main_ddr['bytes'] != cand_ddr['bytes']
                or pair_placement(main_ddr, cand_ddr, local_baseline=True) != 'matched'):
            raise ValueError(f'G{group}: inputs, NPU stack or DDR placement differ between the two live runs')
        if (main_proof['library_sha256'] == cand_proof['library_sha256'] or
                Path(main_cfg['kvaf_root']).resolve() == Path(cand_cfg['kvaf_root']).resolve()):
            raise ValueError(f'G{group}: both runs used the same fusion binary or KVAF checkout')
        print(f'G{group} BOTH_CHECK TopK_score_vs_strong_serial=PASS queries_per_arm=1280/1280 '
              'KV_and_Index_state=PASS ordered_Index=PASS rows_per_arm=320/320')
        for label, times in (('upstream_unified_KVAF', old), ('last_window_and_four_batches', new)):
            print(f'G{group} {label}', *(f'{k}_mean/P50_us={stats(v)}' for k, v in times.items()))
        gains = gain_pct(old['graph'], new['graph'])
        print(f'G{group} vs_live_main', *(f'{k}_saved_mean/P50_us={saved(old[k], new[k])}' for k in new),
              'graph_gain_mean/P50_pct=' + '/'.join(f'{x:+.2f}' for x in gains), flush=True)
        results.append(dict(group=group, main=old, candidate=new, graph_gain_pct=gains))
    baseline = mean(r['main']['graph'][0] for r in results)
    actual = mean(r['candidate']['graph'][0] for r in results)
    gain = 100 * (baseline - actual) / baseline
    count = sum(r['graph_gain_pct'][0] > 10 for r in results)
    print(f'FIVE_GROUP_VS_LIVE_MAIN equal_weight_graph_mean_us main={baseline:.3f} candidate={actual:.3f} '
          f'saved_us={baseline-actual:+.3f} gain_pct={gain:+.3f} groups_over_10pct={count}/5')
    atomic_json(output, dict(candidate_root=str(candidate), main_root=str(main), reader=reader,
        candidate=CANDIDATE, upstream=UPSTREAM, kvaf_candidate=LEGACY_SDK, kvaf_main=UNIFIED_SDK,
        groups=results, equal_weight_graph_mean_us=dict(main=baseline, candidate=actual), gain_pct=gain))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate', type=Path)
    p.add_argument('main', type=Path)
    p.add_argument('reader')
    p.add_argument('output', type=Path)
    args = p.parse_args()
    compare(args.candidate.resolve(), args.main.resolve(), args.reader, args.output)
