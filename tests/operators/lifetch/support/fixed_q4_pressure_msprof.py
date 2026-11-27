# SPDX-License-Identifier: Apache-2.0
"""Collect two checked fixed-Q4 replays with the CANN msprof timeline."""

from contextlib import ExitStack, contextmanager
from decimal import Decimal, InvalidOperation
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


# AscendCL task timing, msproftx marks, runtime calls, and ACL calls.
DATA_TYPES = 0x0002 | 0x0080 | 0x0100 | 0x0001
STEPS = ('init', 'start', 'stop', 'destroy_config', 'finalize')
PHASES = (
    ('HOT', 'v9_li_fixed_hot', 'Hot Score'),
    ('THRESHOLD', 'v9_li_kernel_1', 'Resident Threshold'),
    ('COLD_SERVICE', 'v9_li_fixed_cold', 'Cold Score + Service'),
    ('TERMINAL', 'v9_parallel_terminal', 'Parallel Terminal'),
)
TOLERANCE = Decimal('0.001')
COMPUTE_TYPES = {'AI_CORE', 'AI_VECTOR_CORE', 'AIC', 'AIV', 'AIV_SQE', 'FUSION',
                 'MIX_AIC', 'MIX_AIV', 'KERNEL_MIX_AIC', 'KERNEL_MIX_AIV'}


@contextmanager
def scoped_msprof(prof, output, device_id):
    """Use the live ACL context and collect one already-prepared replay."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    capture = dict(api='acl.prof', output=str(output), device_id=device_id,
                   data_types=DATA_TYPES, status={})

    def checked(operation, *args):
        code = getattr(prof, operation)(*args)
        capture['status'][operation] = code
        if code != 0:
            raise RuntimeError(f'acl.prof.{operation} failed with status {code}')

    with ExitStack() as cleanup:
        checked('init', str(output))
        cleanup.callback(checked, 'finalize')
        handle = prof.create_config([device_id], 0, 0, DATA_TYPES)
        if not handle:
            raise RuntimeError('acl.prof.create_config returned no configuration')
        cleanup.callback(checked, 'destroy_config', handle)
        checked('start', handle)
        cleanup.callback(checked, 'stop', handle)
        yield capture


def _events(document):
    events = document.get('traceEvents') if isinstance(document, dict) else document
    if not isinstance(events, list):
        raise ValueError('CANN export does not contain a trace event list')
    return events


def _interval(event):
    try:
        start, duration = Decimal(str(event['ts'])), Decimal(str(event.get('dur', 0)))
    except (KeyError, InvalidOperation):
        return None
    if not start.is_finite() or not duration.is_finite() or duration < 0:
        return None
    return start, start + duration


def _launch(tasks, label):
    if not tasks:
        raise ValueError(f'Missing {label} device task')
    bounds = sorted({(begin, end) for _name, _stream, begin, end in tasks})
    begin, end = bounds[0]
    for following, finish in bounds[1:]:
        if following >= end:
            raise ValueError(f'{label} contains multiple launches')
        end = max(end, finish)
    return begin, end


def _stage_intervals(document, cold_kernel='v9_li_fixed_cold'):
    phases = tuple((stage, cold_kernel if stage == 'COLD_SERVICE' else kernel, label)
                   for stage, kernel, label in PHASES)
    pattern = {kernel: re.compile(re.escape(kernel) + r'(?:(?:_\d+)|(?:_mix_ai[cv])|(?:_ai[cv]))*', re.I)
               for _, kernel, _ in phases}
    mark_names = {f'KVIP_LIFETCH_{stage}_{side}' for stage, _, _ in phases for side in ('BEGIN', 'END')}
    marks, tasks = {}, []
    for event in _events(document):
        if not isinstance(event, dict):
            continue
        args = event.get('args')
        name = event.get('name')
        if not isinstance(args, dict) or not isinstance(name, str):
            continue
        device = {'Physic Stream Id', 'Task Id'}.issubset(args)
        stream = str(args.get('Physic Stream Id', event.get('tid')))
        interval = _interval(event)
        if interval is None:
            continue
        begin, end = interval
        if device and name in mark_names:
            marks.setdefault(name, set()).add((stream, str(args['Task Id']), begin))
        named = next((kernel for kernel, regex in pattern.items() if regex.fullmatch(name)), '')
        kernel_event = ('kernel' in str(event.get('cat', '')).lower()
                        or str(args.get('Task Type', '')).upper() in COMPUTE_TYPES)
        if event.get('ph') == 'X' and end > begin and (device or kernel_event):
            if 'batchput' in re.sub(r'[^a-z]', '', name.lower()):
                raise ValueError('Batch Put leaked into the scoped capture')
            if named or kernel_event:
                tasks.append((named, stream, begin, end))

    marked = mark_names == marks.keys()
    if marks and not marked:
        raise ValueError('Missing one or more device fusion marks')
    if marked and (any(len(copies) != 1 for copies in marks.values())
                   or len({stream for copies in marks.values() for stream, _task, _ts in copies}) != 1):
        raise ValueError('Device fusion marks do not identify one replay stream')
    intervals = []
    last_end = Decimal('-Infinity')
    for stage, kernel, label in phases:
        named = [row for row in tasks if row[0] == kernel]
        if marked:
            stream, _task, opening = next(iter(marks[f'KVIP_LIFETCH_{stage}_BEGIN']))
            _stream, _task, closing = next(iter(marks[f'KVIP_LIFETCH_{stage}_END']))
            if closing <= opening or opening < last_end - TOLERANCE:
                raise ValueError(f'{label} device marks are out of order')
            enclosed = [row for row in tasks if row[1] == stream
                        and opening <= row[2] + TOLERANCE and row[3] <= closing + TOLERANCE]
            if any(row not in enclosed for row in named) or any(row[0] not in ('', kernel) for row in enclosed):
                raise ValueError(f'{label} device marks enclose the wrong task')
            begin, end = _launch(named or enclosed, label)
            last_end = closing
        else:
            begin, end = _launch(named, label)
            if begin < last_end - TOLERANCE:
                raise ValueError(f'{label} precedes the previous fusion stage')
            last_end = end
        intervals.append(dict(stage=stage, kernel=kernel, name=label, begin_us=float(begin),
                              end_us=float(end), duration_us=float(end - begin)))
    return intervals


def _export(profiler, directory, log):
    profiles = sorted(path for path in directory.glob('PROF_*') if path.is_dir())
    with log.open('w') as stream:
        for profile in profiles:
            if not any(profile.rglob('msprof_*.json')):
                subprocess.run([profiler, '--export=on', f'--output={profile}'], check=True,
                               stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    paths = sorted(path for profile in profiles for path in profile.rglob('msprof_*.json'))
    if not paths:
        raise RuntimeError(f'No CANN timeline was exported, see {log}')
    return paths


def _overlay(events, intervals, caption, pid, offset):
    events.append(dict(name='process_name', ph='M', pid=pid, tid=0, args={'name': caption}))
    for row in intervals:
        events.append(dict(name=row['name'], cat='KVIP LIFetch', ph='X', pid=pid, tid=0,
                           ts=row['begin_us'] - offset, dur=row['duration_us'], args=row))


def _label_capture(paths, route, caption, case, save):
    failures = {}
    for path in paths:
        try:
            vendor = json.loads(path.read_text())
            intervals = _stage_intervals(vendor, case.get('cold_kernel', 'v9_li_fixed_cold'))
        except (ValueError, OSError) as error:
            failures[str(path)] = str(error)
            continue
        events = list(_events(vendor))
        used = {str(row.get('pid')) for row in events if isinstance(row, dict)}
        pid = 2000000
        while str(pid) in used:
            pid += 1
        _overlay(events, intervals, caption, pid, 0)
        view = dict(vendor) if isinstance(vendor, dict) else {}
        view['traceEvents'] = events
        trace = route / 'msprof_trace.json'
        save(trace, view)
        return dict(attribution='VERIFIED', trace=str(trace), intervals=intervals)
    # Preserve the vendor trace even if CANN did not expose attributable names.
    return dict(attribution='UNATTRIBUTED', errors=failures)


def collect_pair(output, baseline_case, optimized_case, *, launch_case, profiler=None):
    """Run one selected case per route, then publish raw and compared traces."""
    from fixed_q4_combo_a5 import match_counts
    from reporting import atomic_json

    profiler = profiler or shutil.which(os.environ.get('KVIP_MSPROF', 'msprof'))
    if not profiler:
        raise FileNotFoundError('msprof is missing from the active CANN environment')
    geometry = ('batch', 'miss_percent', 'windows', 'score_groups', 'cold_score_groups',
                'service_lanes', 'q_workers')
    if any(baseline_case[key] != optimized_case[key] for key in geometry):
        raise ValueError('msprof pair requires the same fixture and launch geometry')
    pair = Path(output).resolve() / 'msprof_pair'
    pair.mkdir()
    captures, rows = {}, []
    for label, case in (('baseline', baseline_case), ('optimized', optimized_case)):
        route = pair / label
        raw = route / 'msprof'
        raw.mkdir(parents=True)
        print(f'Collecting one msprof replay for {label} {case["arm"]}', flush=True)
        row = launch_case(case, route / 'worker', env={'KVIP_FIXED_Q4_PRESSURE_PROFILE': str(raw)})
        rows.append(row)
        receipt = raw / 'capture.json'
        check = json.loads(receipt.read_text())
        if check.get('correctness') != 'PASS' or check.get('status') != dict.fromkeys(STEPS, 0):
            raise ValueError(f'Unchecked scoped replay for {label}, see {receipt}')
        timelines = _export(profiler, raw, route / 'msprof-export.log')
        capture = dict(arm=case['arm'], receipt=str(receipt), raw=[str(path) for path in timelines])
        cycle_report = raw / 'window_cycles.json'
        if cycle_report.is_file():
            capture['window_cycles'] = str(cycle_report)
        capture.update(_label_capture(timelines, route, label, case, atomic_json))
        captures[label] = capture
        atomic_json(route / 'capture_summary.json', capture)
        path = capture.get('trace', capture['raw'][0])
        print(f'{label} msprof trace: {path}', flush=True)

    for field in ('fixture_sha256', 'library_sha256'):
        if rows[0][field] != rows[1][field]:
            raise ValueError(f'msprof routes used different {field}')
    summary = dict(batch=baseline_case['batch'], miss_percent=baseline_case['miss_percent'],
                   fixture_sha256=rows[0]['fixture_sha256'], library_sha256=rows[0]['library_sha256'],
                   profiler_get_match=match_counts(rows[0]['counts'], rows[1]['counts']), **captures)
    if all(capture['attribution'] == 'VERIFIED' for capture in captures.values()):
        base, optimized = (capture['intervals'] for capture in captures.values())
        comparison = [dict(stage=before['name'], baseline_us=before['duration_us'],
                           optimized_us=after['duration_us'],
                           savings_us=before['duration_us'] - after['duration_us'])
                      for before, after in zip(base, optimized, strict=True)]
        with (pair / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
        events = []
        for pid, (label, capture) in enumerate(captures.items(), 1):
            intervals = capture['intervals']
            _overlay(events, intervals, f'{label}: {capture["arm"]}', pid, intervals[0]['begin_us'])
        view = pair / 'comparison_trace.json'
        atomic_json(view, dict(traceEvents=events, displayTimeUnit='us'))
        summary.update(comparison=comparison, comparison_trace=str(view))
        print(f'Open aligned msprof comparison: {view}', flush=True)
    else:
        print('Raw msprof files saved. At least one replay lacks attributable stages.', flush=True)
    atomic_json(pair / 'comparison.json', summary)
    return summary
