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


def find(out):
    area = out.parent
    roots = (p for prefix in ('pr30-feature', 'unified-main-five', 'main-five', 'li-window-five', 'fused')
             for p in area.glob(prefix + '*.*') if p.is_dir())
    for root in sorted(roots, key=lambda p: p.stat().st_mtime_ns, reverse=True):
        try:
            fields = ['host.txt', 'physical_npu.txt']
            if (out/'machine_id.txt').is_file():
                fields.append('machine_id.txt')
            if any((root/f).read_text().strip() != (out/f).read_text().strip() for f in fields):
                continue
        except OSError:
            continue
        for pattern in ('g10/*/thp.json', 'block1/g10/*/thp.json'):
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
    cmds.add_parser('get').add_argument('thp', type=Path)
    cmp = cmds.add_parser('compare')
    cmp.add_argument('serial', type=Path)
    cmp.add_argument('fused', type=Path)
    args = cli.parse_args()
    if args.mode == 'find':
        print(*find(args.out))
    elif args.mode == "get":
        print(*placement(args.thp))
    else:
        first, second = placement(args.serial), placement(args.fused)
        if first != second:
            raise SystemExit(f'CPU placement differs between serial {first} and fused {second}')
