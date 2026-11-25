# SPDX-License-Identifier: Apache-2.0
"""Find reusable, measured NPU4 evidence on the machine running the replay."""
import argparse
from contextlib import redirect_stdout
import os
from pathlib import Path
import sys
from uuid import UUID
from zipfile import BadZipFile

from cap_pair_report import SDK, SDK_TREE, oracle_for
from main_five_report import GROUPS, OWNER, case_folder, read
from peer_delta import pair_placement
from pr30_archive_compare import marker
from pr30_feature_steps import collect_arm
from unified_main_device import REVISION

INVALID = (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, BadZipFile)


def file_value(path):
    return path.read_text().strip() if path.is_file() else ''


def valid_id(value):
    try:
        return bool(UUID(value).int)
    except ValueError:
        return False


def current_runtime(host):
    import torch
    import torch_npu

    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '4':
        raise RuntimeError('Expected physical NPU4 for the runtime comparison')
    return dict(host=host, visible='4', npu=torch.npu.get_device_name(torch.device('npu:0')),
                torch=str(torch.__version__), torch_npu=str(torch_npu.__version__),
                cann=str(getattr(torch.version, 'cann', '')))


def same_local_device(root, identity):
    host, machine, boot = identity
    if marker(root, 'physical_npu') != '4' or not host or marker(root, 'host') != host:
        return False
    saved = marker(root, 'machine_id')
    if valid_id(machine) and valid_id(saved):
        return machine == saved
    return valid_id(boot) and marker(root, 'boot_id') == boot


def usable_placement(path):
    thp = read(path)
    return (thp['status'] == 'measured' and thp['page_mode'] == 'thp' and thp['bytes'] > 0
            and 99 <= thp['coverage_min_pct'] <= 100 and isinstance(thp['ub_device'], str)
            and thp['ub_device'].strip() and pair_placement(thp, thp, local_baseline=True) == 'matched')


def owner_data(root, group):
    ids, proof, snapshot = oracle_for(root, group)
    if not (snapshot['geometry'] and snapshot['identities'] and snapshot['source_paths'] and
            snapshot['rounds'] == 1 and snapshot['page_count'] > 0 and snapshot['kv_payload']):
        raise ValueError('incomplete owner input')
    return ids, proof, snapshot


def newest(paths):
    return sorted(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)


def find_reference(area, group, kind, identity):
    if kind == 'placement':
        paths = newest([*area.glob('*/g*/*/thp.json'), *area.glob('*/block[12]/g*/*/thp.json')])
        # Another trace on this card can supply its CPU adapter and NUMA node.
        paths.sort(key=lambda path: path.parents[1].name != f'g{group}')
        for path in paths:
            try:
                if same_local_device(path.parents[2], identity) and usable_placement(path):
                    return path
            except INVALID:
                continue
    elif kind == 'owner':
        for proof in newest(area.glob('*/logical-ties.txt')):
            try:
                if file_value(proof) == OWNER:
                    owner_data(proof.parent, group)
                    return proof.parent
            except INVALID:
                continue
    else:
        paths = []
        for proof in area.glob('*/main.txt'):
            root = proof.parent
            try:
                if (file_value(proof) != REVISION or marker(root, 'kvaf') != SDK or
                        marker(root, 'kvaf_tree') != SDK_TREE or not same_local_device(root, identity)):
                    continue
                folder = case_folder(root, group, 'mainline')
                # Prefer a full five-case archive over a more recent partial run.
                full = all(any((root/f'g{g}/mainline/results').glob('*/cases/diagnose/fused/path_result.json'))
                           for g in GROUPS)
                paths.append((full, (folder/'path_result.json').stat().st_mtime_ns, root, folder))
            except INVALID:
                continue
        runtime = current_runtime(identity[0]) if paths else None
        for _, _, root, folder in sorted(paths, reverse=True):
            try:
                if read(folder/'mainline_checks.json')['runtime'] != runtime:
                    print(f'G{group} skipping main archive with a different runtime: {root}', file=sys.stderr)
                    continue
                if not usable_placement(root/f'g{group}/mainline/thp.json'):
                    continue
                owner = Path(read(folder/'score_topk_check.json')['native_reference']['archive_path'])
                collect_arm(root, group, 'mainline', owner, owner_data(owner, group), identity[0])
                return root
            except INVALID as error:
                print(f'G{group} skipping unusable main archive {root}: {error}', file=sys.stderr)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('area', type=Path)
    parser.add_argument('group', type=int, choices=GROUPS)
    parser.add_argument('kind', choices=('baseline', 'placement', 'owner'))
    args = parser.parse_args()
    identity = tuple(file_value(Path(p)) for p in
                     ('/proc/sys/kernel/hostname', '/etc/machine-id', '/proc/sys/kernel/random/boot_id'))
    with redirect_stdout(sys.stderr):
        result = find_reference(args.area.resolve(), args.group, args.kind, identity)
    if result is not None:
        print(result)
