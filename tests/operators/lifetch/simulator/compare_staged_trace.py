#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read CANN per-core output-copy completion times for the standalone test."""
import json
import re
from pathlib import Path


def copy_completions(path, expected):
    trace = json.loads(Path(path).read_text())
    events = trace['traceEvents']
    mte3 = [event['pid'] for event in events if event.get('name') == 'process_name'
            and event.get('args', {}).get('name') == 'MTE3']
    if len(mte3) != 1:
        raise ValueError(f'{path}: expected one MTE3 track')
    complete = [event for event in events if event.get('ph') == 'X']
    if not complete:
        raise ValueError(f'{path}: empty device trace')
    start = min(float(event['ts']) for event in complete)
    copies = sorted(float(event['ts']) + float(event['dur']) - start
                    for event in complete if event.get('pid') == mte3[0]
                    and event.get('name') == 'MOV_SRC_TO_DST_ALIGNv2')
    if len(copies) != expected:
        raise ValueError(f'{path}: expected {expected} output copies, got {len(copies)}')
    return copies


def output_maxima(root, mode, expected):
    paths = sorted(path for path in Path(root, mode).rglob('trace.json')
                   if re.fullmatch(r'core\d+\.veccore\d+', path.parent.name)
                   and path.parent.parent.name == 'simulator')
    if len(paths) != 16:
        raise ValueError(f'{mode}: expected 16 per-core traces, got {len(paths)}')
    values = [copy_completions(path, expected) for path in paths]
    return [max(row[i] for row in values) for i in range(expected)]
