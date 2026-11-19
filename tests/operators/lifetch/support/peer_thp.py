# SPDX-License-Identifier: Apache-2.0
"""Measure the KVAF CPU service's registered URMA region while it is alive."""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HEADER = re.compile(r'^([0-9a-fA-F]+)-([0-9a-fA-F]+)\s')
PLACEMENT = re.compile(r'\bN(\d+)=(\d+)\b')


def huge_bounds(smaps, base, size):
    """Return provable bounds, including for VMAs that straddle an arena edge."""
    selected = []
    current = None
    for line in smaps.splitlines():
        match = HEADER.match(line)
        if match:
            left, right = (int(value, 16) for value in match.groups())
            current = None
            if left < base + size and right > base:
                current = {'start': left, 'end': right}
                selected.append(current)
        elif current is not None and line.startswith('AnonHugePages:'):
            current['huge_bytes'] = int(line.split()[1]) * 1024

    low = high = covered = 0
    for row in selected:
        left, right = row['start'], row['end']
        inside = min(right, base + size) - max(left, base)
        if 'huge_bytes' not in row or not 0 <= row['huge_bytes'] <= right - left:
            raise ValueError('AnonHugePages unavailable for registered VMA')
        covered += inside
        low += max(0, row['huge_bytes'] - (right - left - inside))
        high += min(inside, row['huge_bytes'])
    if covered != size:
        raise ValueError('registered span not fully mapped when sampled')
    return low, high, selected


def numa_placement(numa_maps, selected):
    # The kernel charges entire VMAs here, so label the scope even on partial overlap.
    starts = {row['start'] for row in selected}
    nodes = {}
    for line in numa_maps.splitlines():
        words = line.split(maxsplit=1)
        if not words or int(words[0], 16) not in starts:
            continue
        for node, pages in PLACEMENT.findall(line):
            nodes[node] = nodes.get(node, 0) + int(pages)
    return nodes


def observe(args):
    deadline = time.monotonic() + args.timeout
    if args.pid <= 0 or args.timeout <= 0:
        raise ValueError('pid and timeout must be positive')
    # Hold a reference to this process, not a potentially recycled numeric PID.
    fd = os.open(f'/proc/{args.pid}', os.O_RDONLY | os.O_DIRECTORY)
    try:
        proc = Path(f'/proc/self/fd/{fd}')
        with args.log.open(errors='replace') as log:
            while time.monotonic() < deadline:
                for line in log:
                    if '[KVAF_CPU] REGION_ALLOCATED ' not in line:
                        continue
                    fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
                    size = int(fields['bytes'], 0)
                    base = int(fields['base'], 0)
                    if size <= 0 or base <= 0 or fields.get('page_mode') != 'thp':
                        raise ValueError(f'unexpected CPU region: {line.strip()}')
                    low, high, selected = huge_bounds((proc / 'smaps').read_text(), base, size)
                    try:
                        nodes = numa_placement((proc / 'numa_maps').read_text(), selected)
                    except OSError:
                        nodes = {}
                    return dict(status='measured', pid=args.pid, page_mode='thp',
                                base=hex(base), bytes=size, session=fields.get('session'),
                                ub_device=fields.get('ub_device'), huge_bytes_min=low,
                                huge_bytes_max=high, vma_count=len(selected),
                                coverage_min_pct=100 * low / size, coverage_max_pct=100 * high / size,
                                numa_pages_overlapping_vmas=nodes)
                if not (proc / 'exe').exists():
                    raise RuntimeError('CPU service exited before region registration')
                time.sleep(0.10)
        raise TimeoutError(f'no CPU region registered within {args.timeout}s')
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', required=True, type=int)
    parser.add_argument('--log', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout', default=900, type=float)
    args = parser.parse_args()
    try:
        result = observe(args)
    except (OSError, RuntimeError, KeyError, ValueError) as error:
        result = dict(status='unavailable', reason=str(error), pid=args.pid)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'measured':
        print(f"THP_UNVERIFIED {result['reason']}", file=sys.stderr)
        return 1
    print(f"CPU_THP page_mode=thp region_bytes={result['bytes']} "
          f"coverage_pct={result['coverage_min_pct']:.3f}..{result['coverage_max_pct']:.3f} "
          f"ub_device={result['ub_device']} "
          f"numa_pages_overlapping_vmas={result['numa_pages_overlapping_vmas']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
