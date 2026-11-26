# SPDX-License-Identifier: Apache-2.0
"""Reuse this card's URMA placement and compare the actual registered regions."""
import argparse
import json
from pathlib import Path


def placement(path):
    info = json.loads(path.read_text())
    nodes = info['numa_pages_overlapping_vmas']
    if (info['status'] != 'measured' or info['page_mode'] != 'thp' or not nodes
            or not info['ub_device'] or info['coverage_min_pct'] < 99):
        raise ValueError(f'Incomplete CPU memory placement: {path}')
    populated = [str(node) for node, pages in nodes.items() if int(pages) > 0]
    if len(populated) != 1:
        raise ValueError(f'CPU region spans multiple NUMA nodes: {path}: {nodes}')
    return populated[0].removeprefix('N'), info['ub_device']


def field(root, name):
    path = root / (name + '.txt')
    return path.read_text().strip() if path.is_file() else ''


def same_card(root, out):
    """Containers can share hostnames. Require a machine or boot identity too."""
    if not all(field(out, k) and field(out, k) == field(root, k) for k in ('host', 'physical_npu')):
        return False
    machine, other = field(out, 'machine_id'), field(root, 'machine_id')
    if machine and other:
        return machine == other
    return bool(field(out, 'boot_id') and field(out, 'boot_id') == field(root, 'boot_id'))


def find_archive(out, explicit=None):
    """Select only complete same-card archives accepted by compute_reference."""
    from large_batch_reference import COMPUTE_BASELINE
    compatible = {COMPUTE_BASELINE}
    if field(out, 'native'):
        compatible.add(field(out, 'native'))
    roots = (p for prefix in ('largebs-q4.', 'fixed-q4.') for p in out.parent.glob(prefix + '*')
             if p != out and p.is_dir())
    candidates = [explicit] if explicit else sorted(roots, key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for root in candidates:
        if (not same_card(root, out) or not field(out, 'machine_id') or
                field(root, 'machine_id') != field(out, 'machine_id') or field(root, 'native') not in compatible):
            continue
        pairs = [root / f'g{g}/bs{b}' for g in (10, 4, 17, 7, 2) for b in (20, 24, 28, 32)]
        required = ('serial/selected.npy', 'serial/ARM_RESULT.json', 'serial/path/path_result.json',
                    'serial/thp.json', 'fused/ARM_RESULT.json', 'fused/path/path_result.json', 'fused/thp.json')
        if all((pair / name).is_file() for pair in pairs for name in required):
            return root.resolve()
    return None


def find(out):
    area = out.parent
    roots = (p for prefix in ('pr30-feature', 'unified-main-five', 'main-five', 'li-window-five', 'fused',
                               'fixed-q4', 'largebs-q4') for p in area.glob(prefix + '*.*')
             if p.is_dir() and p != out)
    for root in sorted(roots, key=lambda p: p.stat().st_mtime_ns, reverse=True):
        if not same_card(root, out):
            continue
        for pattern in ('g10/*/thp.json', 'block1/g10/*/thp.json', 'g*/bs*/serial/thp.json',
                        'g*/bs*/cohort*/serial/thp.json'):
            for p in sorted(root.glob(pattern)):
                try:
                    return placement(p)
                except (KeyError, OSError, ValueError):
                    continue
    return 'auto', 'auto'


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cmds = cli.add_subparsers(dest='mode', required=True)
    cmds.add_parser('find').add_argument('out', type=Path)
    archive = cmds.add_parser('archive')
    archive.add_argument('out', type=Path)
    archive.add_argument('root', nargs='?', type=Path)
    cmds.add_parser('get').add_argument('thp', type=Path)
    cmp = cmds.add_parser('compare')
    cmp.add_argument('serial', type=Path)
    cmp.add_argument('fused', type=Path)
    args = cli.parse_args()
    if args.mode == 'find':
        print(*find(args.out))
    elif args.mode == 'archive':
        root = find_archive(args.out, args.root)
        if root is not None:
            print(root)
    elif args.mode == "get":
        print(*placement(args.thp))
    else:
        first, second = placement(args.serial), placement(args.fused)
        if first != second:
            raise SystemExit(f'CPU placement differs between serial {first} and fused {second}')
