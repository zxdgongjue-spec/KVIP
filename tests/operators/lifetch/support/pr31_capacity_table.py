# SPDX-License-Identifier: Apache-2.0
"""Build the PR31 performance table from the archived narrow graph and two wide graphs."""
import argparse
import hashlib
from math import isclose
from pathlib import Path
from statistics import mean

import numpy as np

from cap_pair_report import SDK, SDK_TREE, inspect_cap, oracle_for
from feature_host_reference import same_local_device
from main_five_report import case_folder, main_times, measured_parallel, read
from peer_delta import pair_placement
from pr30_archive_compare import marker
from pr30_feature_steps import FRESH_CALLS, GROUPS, receipt_summary, text_hashes
from pr_head_parity_report import REVISIONS, identity
from reporting import atomic_json

NATIVE = REVISIONS['pr31']
SHAPES = {'wide128': 'wide-short', 'near512': 'wide-long'}
METRICS = ('graph', 'LI1', 'LI2', 'LI3', 'LI', 'tail')
CAPACITY = 524544


def stages(folder, times):
    raw = read(folder / 'diagnose/round-1/full_attribution.json')
    spans = [[end - start for start, end in sorted(s['accounting']['intervals_us']['li'])]
             for s in raw['samples']]
    if len(spans) != FRESH_CALLS or any(len(row) != 3 for row in spans):
        raise ValueError(f'Incomplete LI profile in {folder}')
    kernels = [mean(part) for part in zip(*spans)]
    li = times['LI1_to_LI3'][0]
    if not isclose(sum(kernels), li, abs_tol=.025):
        raise ValueError(f'LI stages differ from the measured intervals in {folder}')
    return dict(zip(METRICS, (times['graph'][0], *kernels, li, times['final_topk_through_tail'][0])))


def check_machine(root, archive):
    source = identity(archive)
    for field, value in (('main', NATIVE), ('kvaf', SDK), ('kvaf_tree', SDK_TREE),
                         ('physical_npu', '4'), ('host', source['host'])):
        if marker(root, field) != value:
            raise ValueError(f'{root}: {field} differs from the narrow run')
    machine = tuple(marker(root, field) for field in ('host', 'machine_id', 'boot_id'))
    if not same_local_device(archive, machine):
        raise ValueError('Unable to identify the same machine as the narrow run')
    if Path(marker(root, 'reference')).resolve() != archive:
        raise ValueError('Unexpected reference archive')


def same_binary(proof, earlier):
    if proof['library_sha256'] == earlier['library_sha256']:
        return True
    # A rebuild in a new checkout can change host build metadata. Compare the
    # executable device sections only if the previous artifact still matches its receipt.
    old = Path(earlier['loaded_library'])
    if not old.is_file() or hashlib.sha256(old.read_bytes()).hexdigest() != earlier['library_sha256']:
        return False
    return text_hashes(proof['loaded_library']) == text_hashes(old)


def same_long_input(first, second):
    # The native TopK may choose different IDs at a tied cutoff. The resulting
    # number of distinct winners per hot/cold region is a measurement, not an input.
    return all(first[key] == second[key] for key in (
        'source_kind', 'source_oracle_digest', 'source_page_map_sha256',
        'long_operand_sha256', 'long_state_sha256', 'short_unique', 'native_queries', 'new_winners'))


def wide_block(root, group, label, owner, oracle, narrow_proof, placement):
    folder = case_folder(root, group, label)
    times, exact, thp, proof, cfg = measured_parallel(
        root, group, label, NATIVE, owner, *oracle, verify_library=True,
        capture_meta_name='capture_input.json', expected_calls=FRESH_CALLS)
    if (proof['runtime'] != narrow_proof['runtime'] or
            not same_binary(proof, narrow_proof) or
            proof['capture_operand_hashes'] != narrow_proof['capture_operand_hashes'] or
            proof['pr31_shape'] != SHAPES[label] or cfg['history'] != CAPACITY or
            proof['independent_buffers'] != dict.fromkeys(('kv', 'index', 'winners', 'window'), FRESH_CALLS) or
            pair_placement(placement, thp, local_baseline=True) != 'matched'):
        raise ValueError(f'G{group} {label}: version, runtime, capture, private state or placement mismatch')
    captured = read(folder.parent / 'capture_input.json')
    replay = read(folder.parent / 'replay_input.json')
    ref_cap = narrow_proof['history_cap']
    cap = proof['history_cap']
    for key in ('capture_history', 'capture_state_sha256', 'captured_page_count', 'capture_page_table_sha256'):
        if cap[key] != ref_cap[key]:
            raise ValueError(f'G{group} {label}: donor {key} changed')
    if (cap['bound_history'] != CAPACITY or replay['geometry'] != {**captured['geometry'], 'history': CAPACITY}
            or cap['bound_page_count'] != replay['page_count']
            or thp['bytes'] != (cap['bound_page_count'] + 2) * 128 * captured['geometry']['token_bytes']):
        raise ValueError(f'G{group} {label}: workspace or page binding differs from the recorded input')
    if label == 'wide128':
        inspect_cap('cap512k', captured, replay, cap, oracle[1]['native_reference'], expected_calls=FRESH_CALLS)
        if thp['bytes'] != placement['bytes']:
            raise ValueError(f'G{group}: padding short input changed the registered DDR size')
        workload = None
    else:
        ends = np.asarray(cap['effective_ends'])
        expected = np.asarray(ref_cap['effective_ends']) + CAPACITY - ref_cap['bound_history']
        workload = proof['long_workload']
        if (not np.array_equal(ends, expected) or ends.shape != (16, 4)
                or ends[:, 0].min() < 510 * 1024 or ends[:, -1].max() > 512 * 1024 + 3
                or cap['bound_page_count'] <= ref_cap['bound_page_count']
                or workload['source_kind'] != 'rank_preserving_short_capture'
                or workload['native_queries'] != 64 or workload['new_winners'] != 0):
            raise ValueError(f'G{group}: invalid near-512K geometry, source mapping or native TopK receipt')
    cq = read(folder / 'ablation_receipts.json')
    if cq['policy'] != 'paced' or cq['independent_status_allocations'] != FRESH_CALLS:
        raise ValueError(f'G{group} {label}: independent CQ status missing')
    for phase in ('measured', 'detail'):
        receipt_summary(cq[phase], True)
    measured_requests = cap['measured_requests']
    counts = [(sum(r['prefetch']), sum(map(sum, r['final']))) for r in cq['measured']]
    if counts != list(zip(measured_requests['prefetch'], measured_requests['final'])):
        raise ValueError(f'G{group} {label}: GET counts disagree with the CQ receipts')
    if exact.shape != (FRESH_CALLS, 16, 4):
        raise ValueError(f'G{group} {label}: incomplete TopK checks')
    return stages(folder, times), workload


def averaged(blocks):
    return {key: mean(b[key] for b in blocks) for key in METRICS}


def summary_row(control, narrow, wide, long):
    return (control, narrow['graph'], wide['graph'], long['graph'], wide['graph'] - narrow['graph'],
            long['graph'] - wide['graph'], long['LI3'] - wide['LI3'])


def markdown(rows, narrow_archive, wide_archive):
    avg = [mean(col) for col in zip(*rows.values())]
    lines = ['## B20 NPU4 性能', '',
             '五组 BS16、Q4，每个 KV token 656 B。一个设备事件包住十次独立状态的调用，'
             '耗时除以十。每组预热四次，测两轮，结果等权平均。单位均为 µs。',
             '两种 128K 上限配置取自前次同机实测。512K 上限下的长、短输入新测两轮，第二轮调换顺序。', '',
             '| 负载 | PR30 对照 · 128K/128K | PR31 · 128K/128K | PR31 · 512K/128K | '
             'PR31 · 512K/近512K | 只提高上限的整图增量 | 输入变长的整图增量 | 其中 LI 第三段增量 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    table_rows = [(f'G{g}', values) for g, values in rows.items()]
    table_rows.append(('**五组等权**', avg))
    for group, values in table_rows:
        cells = [f'{x:.2f}' for x in values[:4]] + [f'{x:+.2f}' for x in values[4:]]
        lines.append('| ' + ' | '.join([group, *cells]) + ' |')
    lines += ['', '表头中斜线左侧是模型上限，右侧是有效输入长度。差值用未舍入的均值计算，正值表示增加。',
              '近 512K 输入从本组短 trace 扩展，保留原始赢家和初始冷热关系。长输入重新核对原生 TopK，'
              '允许截止同分替换。三种 PR31 配置均通过 KV、Index 和每个 Q 的完成回执校验。',
              f'128K 归档为 `{narrow_archive.name}`，512K 归档为 `{wide_archive.name}`。']
    return '\n'.join(lines) + '\n'


def report(out, archive):
    check_machine(out, archive)
    raw_narrow = read(archive / 'pr_head_parity.json')
    if raw_narrow['identity'] != identity(archive):
        raise ValueError('Invalid archived narrow identity')
    groups, rows = {}, {}
    for g in GROUPS:
        blocks = []
        orders = []
        workloads = []
        for number in (1, 2):
            narrow_root = archive / f'block{number}'
            root = out / f'block{number}'
            order = (root / f'g{g}/order.txt').read_text().split()
            if sorted(order) != sorted(SHAPES) or (orders and order != orders[0][::-1]):
                raise ValueError(f'G{g}: expected two reverse-order blocks')
            orders.append(order)
            narrow = case_folder(narrow_root, g, 'pr31')
            narrow_proof = read(narrow / 'mainline_checks.json')
            narrow_row = read(narrow_root / f'g{g}/pr_head_parity.json')
            placement = read(narrow_root / f'g{g}/pr31/thp.json')
            archive_path = read(narrow / 'score_topk_check.json')['native_reference']['archive_path']
            owner = Path(archive_path).resolve()
            oracle = oracle_for(owner, g)
            narrow_times = main_times(narrow, read(narrow / 'path_result.json')['all_rounds'], expected_calls=FRESH_CALLS)
            values = {'pr31': stages(narrow, narrow_times)}
            if (narrow_row['identity'] != raw_narrow['identity'] or
                    not isclose(values['pr31']['graph'], narrow_row['arms']['pr31']['means']['event_per_call'], abs_tol=.001)):
                raise ValueError(f'G{g}: narrow receipt and raw timings disagree')
            for label in order:
                values[label], workload = wide_block(root, g, label, owner, oracle, narrow_proof, placement)
                if label == 'near512':
                    if workloads and not same_long_input(workloads[0], workload):
                        raise ValueError(f'G{g}: long input differs between blocks')
                    workloads.append(workload)
            blocks.append(values)
        group = {label: averaged([b[label] for b in blocks]) for label in ('pr31', *SHAPES)}
        narrow_means = raw_narrow['groups'][str(g)]['means']
        if not isclose(group['pr31']['graph'], narrow_means['pr31']['event_per_call'], abs_tol=.001):
            raise ValueError(f'G{g}: narrow summary differs from the checked blocks')
        control = narrow_means['pr30']['event_per_call']
        rows[g] = summary_row(control, group['pr31'], group['wide128'], group['near512'])
        groups[g] = dict(times=group, pr30_graph=control, near512_workloads=workloads, blocks=blocks)
    atomic_json(out / 'pr31_capacity_table.json', dict(reference=str(archive), native=NATIVE, groups=groups))
    table = markdown(rows, archive, out)
    path = out / 'pr31-performance.md'
    path.write_text(table, encoding='utf-8')
    print(table, end='')
    print(f'TABLE_READY={path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    report(args.out.resolve(), args.archive.resolve())
