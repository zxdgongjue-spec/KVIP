"""Report the paced terminal on the selected B20 NPU with explicit timing provenance."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from li_restore_archive import dp_capture_id, normalized_inputs
from li_restore_report import SERIAL_NPU4, saved, stats
from main_five_report import archived_oracle, case_folder, main_times, read
from peer_delta import pair_placement, single_numa_node
from ranked_integration_sets import INPUT_AND_RESULTS
from reporting import atomic_json
from score_equivalence_report import measured_sets
from strong_serial_g10 import from_this_host, serial_measurement

MAIN = '609fa26fd715f9409bbe74e1b2d3685c58dfb0cf'
PACED = '6d7f257b3d664a8b4991cfd3ebd12da4f2038d04'
GROUPS = (10, 4, 17, 7, 2)
B24_MAIN = {10: (297.10, 289.57), 4: (315.79, 304.59), 17: (327.49, 314.24),
            7: (580.87, 587.58), 2: (284.11, 276.89)}
B24_SERIAL = {**SERIAL_NPU4, 10: (356.98, 348.26)}
ARCHIVE_ERRORS = (OSError, KeyError, ValueError, TypeError, IndexError, AssertionError)


def run_device(root):
    physical = (root/'physical_npu.txt').read_text().strip()
    if physical not in ('0', '5'):
        raise ValueError(f'Unexpected B20 physical device {physical}')
    return physical


def local_archives(area, group, arm, physical):
    # Prefer the newest compatible archive, independently of the measured latency.
    files = sorted(area.glob(f'*/g{group}/{arm}/thp.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for f in files:
        root = f.parents[2]
        try:
            if (root/'physical_npu.txt').read_text().strip() != physical or not from_this_host(root, physical):
                continue
            if arm == 'mainline' and ((root/'candidate.txt').exists() or (root/'main.txt').read_text().strip() != MAIN):
                continue
            yield root, read(f)
        except ARCHIVE_ERRORS:
            continue


def target_node(root, group):
    physical = run_device(root)
    for arm in ('mainline', 'serial2'):
        for _, thp in local_archives(root.parent, group, arm, physical):
            if thp.get('status') != 'measured' or 'numa_pages_overlapping_vmas' not in thp:
                continue
            node = single_numa_node(thp)
            if node is not None:
                return node
    cpu = min(os.sched_getaffinity(0))
    nodes = list(Path(f'/sys/devices/system/cpu/cpu{cpu}').glob('node[0-9]*'))
    return nodes[0].name[4:] if len(nodes) == 1 else ''


def verified_input(group, old, new):
    if any(old[k] != new[k] for k in ('geometry', 'identities', 'rounds', 'page_count')):
        return False
    a, b = dp_capture_id(group, old), dp_capture_id(group, new)
    if a is not None and b is not None:
        return a == b == INPUT_AND_RESULTS[group][0]
    return normalized_inputs(group, old) == normalized_inputs(group, new)


def main_archive(root, group, current_proof, current_cfg):
    folder = case_folder(root, group, 'mainline')
    proof = read(folder/'mainline_checks.json')
    cfg = read(folder.parent/'config.json')
    result = read(folder/'path_result.json')
    if (cfg['physical_npu'] != current_cfg['physical_npu'] or proof['runtime'] != current_proof['runtime']
            or cfg['kvaf_root'] != current_cfg['kvaf_root'] or proof['production_revision'] != MAIN
            or cfg['production_revision'] != MAIN or proof['capture_operand_hashes'] != current_proof['capture_operand_hashes']
            or not proof['production_terminal'] or proof['ordered_diagnostic_rows'] != 320
            or proof['measured_kv_index_rows'] != 320 or result['correctness'] != 'PASS' or result['failures']):
        raise ValueError('incompatible mainline replay')
    return main_times(folder, result['all_rounds'])['graph'], read(folder.parent/'replay_input.json')


def compare_local(area, group, arm, new, thp, inputs, proof, cfg):
    physical = str(cfg['physical_npu'])
    skipped = []
    for root, old_thp in local_archives(area, group, arm, physical):
        try:
            if old_thp['bytes'] != thp['bytes'] or pair_placement(old_thp, thp, local_baseline=True) != 'matched':
                raise ValueError('DDR placement differs')
            if arm == 'serial2':
                archive, old, _ = serial_measurement(root, group, physical)
                old_inputs = archive['inputs']
            else:
                old, old_inputs = main_archive(root, group, proof, cfg)
            if not verified_input(group, old_inputs, inputs):
                raise ValueError('different capture inputs')
        except ARCHIVE_ERRORS as exc:
            skipped.append(f'{root.name} ({exc})')
            continue
        print(f'G{group} vs_B20_NPU{physical}_{arm} archived_graph_mean/P50_us={stats(old)} '
              f'graph_saved_mean/P50_us={saved(old,new)} gain_mean/P50_pct={percent(old,new)} source={root}')
        return list(old)
    why = '; '.join(skipped[:2]) if skipped else 'no_matching_local_archive'
    print(f'G{group} vs_B20_NPU{physical}_{arm} UNAVAILABLE reason={why}')
    return None


def percent(old, new):
    return '/'.join(f'{100*(1-y/x):+.2f}' for x,y in zip(old,new))


def report(root, group):
    physical = run_device(root)
    folder = case_folder(root, group, 'paced-tail')
    proof = read(folder/'mainline_checks.json')
    cfg = read(folder.parent/'config.json')
    receipt = read(folder/'score_topk_check.json')
    result = read(folder/'path_result.json')
    inputs = read(folder.parent/'replay_input.json')
    if ((root/'candidate.txt').read_text().strip() != PACED or cfg['production_revision'] != PACED
            or proof['production_revision'] != PACED or cfg['physical_npu'] != int(physical) or cfg['device'] != 0
            or proof['runtime']['visible'] != physical or result['correctness'] != 'PASS' or result['failures']
            or not proof['production_terminal'] or proof['profiled_terminal_kernel'] != 'v9_parallel_terminal'
            or [proof[k] for k in ('measured_calls','measured_kv_index_rows','ordered_diagnostic_calls','ordered_diagnostic_rows')]
               != [20,320,20,320]
            or proof['capture_operand_hashes'] != receipt['native_reference']['identity']['inputs']
            or hashlib.sha256(Path(proof['loaded_library']).read_bytes()).hexdigest() != proof['library_sha256']):
        raise ValueError(f'G{group} incomplete candidate, device, build, or correctness evidence')
    source = receipt['native_reference']
    if source['origin'] == 'completed-owner-archive':
        serial, _ = archived_oracle(Path(source['archive_path']),group)
    else:
        if source['origin'] not in ('fresh', 'validated-cache') or source['identity']['visible'] != physical:
            raise ValueError('unverified native LI reference')
        with np.load(folder/'score_topk_ids.npz', allow_pickle=False) as ids:
            serial = ids['strong_serial']
    exact = measured_sets(folder, group, (inputs, result['topk_set_digests']), serial, receipt)
    times = main_times(folder, result['all_rounds'])
    thp = read(root/f'g{group}/paced-tail/thp.json')
    if thp['status'] != 'measured' or thp['page_mode'] != 'thp' or thp['bytes'] <= 0:
        raise ValueError('missing measured DDR placement')
    print(f'G{group} CHECK TopK_score_vs_strong_serial=PASS calls=20/20 queries=1280/1280 '
          f'exact_ID_sets={int(exact.sum())}/1280 KV=PASS ordered_Index=PASS rows=320/320 oracle={source["origin"]}')
    print(f'G{group} paced-tail graph_mean/P50_us={stats(times["graph"])} '
          f'LI_mean/P50_us={stats(times["LI1_to_LI3"])} tail_mean/P50_us={stats(times["final_topk_through_tail"])}')
    local = {}
    for arm, historical in [('mainline', B24_MAIN), ('serial2', B24_SERIAL)]:
        local[arm] = compare_local(root.parent,group,arm,times['graph'],thp,inputs,proof,cfg)
        if local[arm] is None:
            old = historical[group]
            # The four early serial archives identify physical device 4 only.
            provenance = 'B24_NPU4' if arm == 'mainline' or group == 10 else 'archived_NPU4'
            print(f'G{group} vs_{provenance}_{arm} historical_other_device '
                  f'archive_minus_candidate_graph_mean/P50_us={saved(old,times["graph"])} '
                  f'difference_mean/P50_pct={percent(old,times["graph"])}')
    atomic_json(root/f'g{group}/paced_comparison.json',dict(group=group, graph=times['graph'], local=local))


def gate(values):
    if False in values:
        return 'NOT_MET'
    return 'UNAVAILABLE' if None in values else 'PASS'


def summary(root):
    physical = run_device(root)
    rows = {g: read(root/f'g{g}/paced_comparison.json') for g in GROUPS}
    def compare(groups, arm, i, factor):
        ok = []
        for g in groups:
            reference = rows[g]['local'][arm]
            if reference is None:
                ok.append(None)
            else:
                value, target = rows[g]['graph'][i], reference[i]*factor
                ok.append(value < target if factor < 1 else value <= target)
        return gate(ok)
    for name, groups, arm, factor in [('G7_over_20pct',(7,),'serial2',.8),
                                      ('other_four_no_regression',(10,4,17,2),'mainline',1)]:
        print(f'B20_NPU{physical}_GATE {name}_mean/P50=' +
              '/'.join(compare(groups,arm,i,factor) for i in range(2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('oracle','node','report','summary'))
    parser.add_argument('root',type=Path)
    parser.add_argument('group',nargs='?',type=int,choices=GROUPS)
    args = parser.parse_args()
    if args.action == 'oracle':
        try:
            archived_oracle(args.root,args.group)
        except ARCHIVE_ERRORS:
            raise SystemExit(1)
    elif args.action == 'node':
        print(target_node(args.root,args.group))
    elif args.action == 'report':
        report(args.root,args.group)
    else:
        summary(args.root)
