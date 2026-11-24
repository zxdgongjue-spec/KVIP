"""Compare complete fusion stacks, freshly measured on the same physical NPU4."""
import argparse
from pathlib import Path
from statistics import mean

from last_window_borrow import COMBINED as CANDIDATE, expected_text
from li_restore_report import saved, stats
from main_five_report import archived_oracle, case_folder, measured_parallel, oracle_case_folder, read
from peer_delta import pair_placement
from reporting import atomic_json
from terminal_batch_checks import totals
from unified_main_device import EXPECTED_TEXT as MAIN_TEXT, REVISION as UPSTREAM


LEGACY_SDK = 'a2857b460a74704114fd30a849ae38666cc2c372'
UNIFIED_SDK = '813f1750a50b7e6c92d5f5397f2fbc05befef6da'
GROUPS = (10, 4, 17, 7, 2)


def marker(root, name):
    return (root / f'{name}.txt').read_text().strip()


def gain_pct(reference, candidate):
    return [100 * (a-b) / a for a, b in zip(reference, candidate)]


def compare(candidate, main, reader, output, site='host24'):
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
    local_owner = None
    if site == 'host20':
        local_owner = (area / 'host20-npu4-post-before-output.tw7TkM').resolve()
        for root in (candidate, main):
            if marker(root, 'site') != site or Path(marker(root, 'oracle_archive')).resolve() != local_owner:
                raise ValueError(f'{root}: expected B20 NPU4 native TopK archive {local_owner}')
    results = []
    for group in GROUPS:
        owner = local_owner or area / ('fused.gbxKC9' if group == 2 else 'fused.N7n8ZP')
        ids, receipt = archived_oracle(owner, group)
        inputs = read(oracle_case_folder(owner, group).parent / 'replay_input.json')
        arms = []
        for root, arm, revision in ((main, 'mainline', UPSTREAM), (candidate, 'li-window', CANDIDATE)):
            arms.append(measured_parallel(root, group, arm, revision, owner, ids, receipt, inputs,
                                          verify_library=True))
        (old, _, main_ddr, main_proof, main_cfg), (new, _, cand_ddr, cand_proof, cand_cfg) = arms
        combined_folder = case_folder(candidate, group, 'li-window')
        combined_proof = read(combined_folder / 'borrow_checks.json')
        if (main_proof.get('device_text_sha256') != MAIN_TEXT or
                combined_proof.get('device_text_sha256') != expected_text(CANDIDATE)):
            raise ValueError(f'G{group}: saved kernel evidence differs from independently built LI and terminal')
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
        cq = read(combined_folder / 'final_cq_checks.json')
        if cq['independent_status_allocations'] != 20:
            raise ValueError(f'G{group}: missing independent terminal completion records')
        batches = totals(cq['measured'])
        print(f'G{group} measured_private_final_CQ batches_0/1/2/3/4=' + '/'.join(map(str, batches)))
        print(f'G{group} vs_live_main', *(f'{k}_saved_mean/P50_us={saved(old[k], new[k])}' for k in new),
              'graph_gain_mean/P50_pct=' + '/'.join(f'{x:+.2f}' for x in gains), flush=True)
        results.append(dict(group=group, main=old, candidate=new, graph_gain_pct=gains, candidate_cq_batches=batches))
    baseline = mean(r['main']['graph'][0] for r in results)
    actual = mean(r['candidate']['graph'][0] for r in results)
    gain = 100 * (baseline - actual) / baseline
    count = sum(r['graph_gain_pct'][0] > 10 for r in results)
    print(f'FIVE_GROUP_VS_LIVE_MAIN equal_weight_graph_mean_us main={baseline:.3f} candidate={actual:.3f} '
          f'saved_us={baseline-actual:+.3f} gain_pct={gain:+.3f} groups_over_10pct={count}/5')
    atomic_json(output, dict(site=site, candidate_root=str(candidate), main_root=str(main), reader=reader,
        candidate=CANDIDATE, upstream=UPSTREAM, kvaf_candidate=LEGACY_SDK, kvaf_main=UNIFIED_SDK,
        groups=results, equal_weight_graph_mean_us=dict(main=baseline, candidate=actual), gain_pct=gain))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate', type=Path)
    p.add_argument('main', type=Path)
    p.add_argument('reader')
    p.add_argument('output', type=Path)
    p.add_argument('--site', choices=('host20', 'host24'), default='host24')
    args = p.parse_args()
    compare(args.candidate.resolve(), args.main.resolve(), args.reader, args.output, args.site)
