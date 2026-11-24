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
CREDIT_TRACE = '957f59ace1520d3da8ef839a4d341902f62f8f66'
POST_BEFORE_OUTPUT = 'c26c39e27730c10d33c82bf996807c4ef9d4952b'
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


def four_batch_archive(root, group):
    name = 'host20-npu5-paced-tail.usoCcG' if group == 10 else 'host20-npu5-paced-tail.eGdWHC'
    archive = root.parent / name
    if ((archive/'candidate.txt').read_text().strip() != PACED or run_device(archive) != '5'
            or not from_this_host(archive, '5')):
        raise ValueError(f'G{group}: expected the completed four-batch archive on this host and NPU5 at {archive}')
    return archive


def four_batch_trace(root, group):
    archive = four_batch_archive(root, group)
    folder = case_folder(archive, group, 'paced-tail')
    trace = Path(read(folder.parent/'config.json')['trace_root'])
    if not trace.is_dir():
        raise FileNotFoundError(f'G{group}: archived replay input directory {trace}')
    return trace


def host16_run(root):
    marker = root / 'site.txt'
    return marker.is_file() and marker.read_text().strip() == 'host16'


def host16_four_batch_archives(root, group):
    # Containers on different machines can share a hostname and machine ID.
    # Accept a legacy archive with no site marker only when the boot UUID matches.
    boot = (root / 'boot_id.txt').read_text().strip()
    for archive, thp in local_archives(root.parent, group, 'paced-tail', '5'):
        try:
            if ((archive/'candidate.txt').read_text().strip() == PACED
                    and (host16_run(archive) or (boot and (archive/'boot_id.txt').read_text().strip() == boot))
                    and thp['status'] == 'measured' and thp['page_mode'] == 'thp'
                    and 'numa_pages_overlapping_vmas' in thp):
                yield archive, thp
        except ARCHIVE_ERRORS:
            continue


def four_batch_times(archive, group, thp, inputs, proof, cfg):
    folder = case_folder(archive, group, 'paced-tail')
    old_proof = read(folder/'mainline_checks.json')
    old_cfg = read(folder.parent/'config.json')
    old_result = read(folder/'path_result.json')
    old_thp = read(archive/f'g{group}/paced-tail/thp.json')
    if (old_proof['production_revision'] != PACED or old_cfg['production_revision'] != PACED
            or old_cfg['physical_npu'] != cfg['physical_npu'] or old_cfg['kvaf_root'] != cfg['kvaf_root']
            or old_proof['runtime'] != proof['runtime']
            or old_proof['capture_operand_hashes'] != proof['capture_operand_hashes']
            or not verified_input(group, read(folder.parent/'replay_input.json'), inputs)
            or not old_proof['production_terminal'] or old_result['correctness'] != 'PASS' or old_result['failures']
            or [old_proof[k] for k in ('measured_calls', 'measured_kv_index_rows', 'ordered_diagnostic_rows')] != [20, 320, 320]):
        raise ValueError(f'G{group}: archive inputs, runtime or four-batch correctness differs at {folder}')
    old = main_times(folder, old_result['all_rounds'])
    placement = pair_placement(old_thp, thp, local_baseline=True)
    if old_thp['bytes'] != thp['bytes'] or old_thp['page_mode'] != thp['page_mode']:
        placement = 'different_memory'
    return old, placement


def compare_four_batch(root, group, times, thp, inputs, proof, cfg):
    if host16_run(root):
        reason = 'no_matching_host16_npu5_four_batch_archive'
        for archive, _ in host16_four_batch_archives(root, group):
            try:
                old, placement = four_batch_times(archive, group, thp, inputs, proof, cfg)
            except ARCHIVE_ERRORS as exc:
                reason = f'{archive.name}: {exc}'
                continue
            if placement == 'matched':
                break
            reason = f'{archive.name}: {placement}'
        else:
            print(f'G{group} vs_four_batch UNAVAILABLE reason={reason}')
            atomic_json(root/f'g{group}/four_batch_comparison.json',
                        dict(archive=None, new=times, reason=reason, passed=[None, None]))
            return
    else:
        archive = four_batch_archive(root, group)
        old, placement = four_batch_times(archive, group, thp, inputs, proof, cfg)
    print(f'G{group} vs_four_batch graph_saved_mean/P50_us={saved(old["graph"], times["graph"])} '
          f'tail_saved_mean/P50_us={saved(old["final_topk_through_tail"], times["final_topk_through_tail"])} '
          f'gain_mean/P50_pct={percent(old["graph"], times["graph"])} placement={placement} archive={archive.name}')
    passed = [new <= 0.97 * ref if placement == 'matched' else None
              for ref, new in zip(old['graph'], times['graph'])]
    atomic_json(root/f'g{group}/four_batch_comparison.json', dict(archive=str(archive), old=old, new=times,
                                                                  placement=placement, passed=passed))


def target_node(root, group):
    physical = run_device(root)
    if host16_run(root):
        for _, thp in host16_four_batch_archives(root, group):
            node = single_numa_node(thp)
            if node is not None:
                return node
    elif (root/'candidate.txt').read_text().strip() == POST_BEFORE_OUTPUT:
        archive = four_batch_archive(root, group)
        node = single_numa_node(read(archive/f'g{group}/paced-tail/thp.json'))
        if node is not None:
            return node
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
    expected = (root/'candidate.txt').read_text().strip()
    arm = 'post-before-output' if expected == POST_BEFORE_OUTPUT else 'paced-tail'
    folder = case_folder(root, group, arm)
    proof = read(folder/'mainline_checks.json')
    cfg = read(folder.parent/'config.json')
    receipt = read(folder/'score_topk_check.json')
    result = read(folder/'path_result.json')
    inputs = read(folder.parent/'replay_input.json')
    allowed = (PACED, CREDIT_TRACE) if group == 7 and physical == '5' else (PACED,)
    if physical == '5':
        allowed += (POST_BEFORE_OUTPUT,)
    if (expected not in allowed or cfg['production_revision'] != expected
            or proof['production_revision'] != expected or cfg['physical_npu'] != int(physical) or cfg['device'] != 0
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
    thp = read(root/f'g{group}/{arm}/thp.json')
    if thp['status'] != 'measured' or thp['page_mode'] != 'thp' or thp['bytes'] <= 0:
        raise ValueError('missing measured DDR placement')
    print(f'G{group} CHECK TopK_score_vs_strong_serial=PASS calls=20/20 queries=1280/1280 '
          f'exact_ID_sets={int(exact.sum())}/1280 KV=PASS ordered_Index=PASS rows=320/320 oracle={source["origin"]}')
    if expected == CREDIT_TRACE:
        return
    print(f'G{group} {arm} graph_mean/P50_us={stats(times["graph"])} '
          f'LI_mean/P50_us={stats(times["LI1_to_LI3"])} tail_mean/P50_us={stats(times["final_topk_through_tail"])}')
    if expected == POST_BEFORE_OUTPUT:
        compare_four_batch(root, group, times, thp, inputs, proof, cfg)
        return
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
    if (root/'candidate.txt').read_text().strip() == POST_BEFORE_OUTPUT:
        rows = [read(root/f'g{g}/four_batch_comparison.json')['passed'] for g in GROUPS]
        print('ALL_FIVE_3PCT graph_mean/P50=' + '/'.join(gate([r[i] for r in rows]) for i in range(2)))
        return
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
    parser.add_argument('action',choices=('oracle','node','report','summary','trace'))
    parser.add_argument('root',type=Path)
    parser.add_argument('group',nargs='?',type=int,choices=GROUPS)
    args = parser.parse_args()
    if args.action == 'oracle':
        try:
            archived_oracle(args.root,args.group)
        except ARCHIVE_ERRORS:
            raise SystemExit(1)
    elif args.action == 'trace':
        print(four_batch_trace(args.root,args.group))
    elif args.action == 'node':
        print(target_node(args.root,args.group))
    elif args.action == 'report':
        report(args.root,args.group)
    else:
        summary(args.root)
