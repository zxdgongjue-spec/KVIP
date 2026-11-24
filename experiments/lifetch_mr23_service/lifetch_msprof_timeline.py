# SPDX-License-Identifier: Apache-2.0
"""Verify and label one CANN-exported fused replay."""

from collections import defaultdict
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path


PHASES = (
    ('HOT', 'v9_li_kernel', 'Hot Score'),
    ('THRESHOLD', 'v9_li_kernel_1', 'Resident threshold'),
    ('COLD_SERVICE', 'v9_li_kernel_2', 'Cold Score + rotating Service'),
    ('TERMINAL', 'v9_parallel_terminal', 'Terminal'),
)
PHASES_COLD_P4 = (
    PHASES[0],
    ('THRESHOLD', 'v9_li_kernel_threshold_cold_partitioned', PHASES[1][2]),
    ('COLD_SERVICE', 'v9_li_kernel_cold_partitioned', PHASES[2][2]),
    PHASES[3],
)
PHASES_UNIFIED = (
    ('HOT', 'v9_li_kernel_score_t2', 'Full history Score'),
    PHASES[1],
    ('COLD_SERVICE', 'v9_li_kernel_cold_t2', 'Cold T2 + rotating Service'),
    PHASES[3],
)
ROUTE_PHASES = {'p2': PHASES, 'cold_p4': PHASES_COLD_P4,
                'full_p2': PHASES_UNIFIED, 'full_uniform': PHASES_UNIFIED}
MARKERS = {f'KVIP_LIFETCH_{stage}_{side}' for stage, _, _ in PHASES for side in ('BEGIN', 'END')}
KERNELS = {kernel for phases in ROUTE_PHASES.values() for _, kernel, _ in phases}
COMPUTE_TYPES = {'AI_CORE', 'AI_VECTOR_CORE', 'MIX_AIC', 'MIX_AIV', 'AIV_SQE', 'FUSION'}


def events(document):
    result = document.get('traceEvents') if isinstance(document, dict) else document
    if not isinstance(result, list):
        raise ValueError('Expected a CANN timeline event list')
    return result


def get_interval(event):
    try:
        begin = Decimal(str(event['ts']))
        duration = Decimal(str(event.get('dur', 0)))
    except (KeyError, InvalidOperation):
        return None
    if not begin.is_finite() or not duration.is_finite() or duration < 0:
        return None
    return begin, begin + duration


def require_fused_replay(document, *, cold_route='p2'):
    """Use stable CANN device task fields and fence every kernel by its marks."""
    tasks, markers = [], defaultdict(set)
    for event in events(document):
        if not isinstance(event, dict):
            continue
        name, args = event.get('name'), event.get('args')
        if not isinstance(name, str) or not isinstance(args, dict):
            continue
        if not {'Physic Stream Id', 'Task Id'}.issubset(args):
            continue  # The host also carries the same mark message.
        stream = str(args['Physic Stream Id'])
        interval = get_interval(event)
        if interval is None:
            continue
        if name in MARKERS:
            markers[name].add((stream, str(args['Task Id']), interval[0]))
        if event.get('ph') == 'X' and 'Task Type' in args and interval[1] > interval[0]:
            compact = ''.join(character for character in name.lower() if character.isalnum())
            if 'batchput' in compact:
                raise ValueError(f'BatchPut leaked into the scoped fusion capture ({name})')
            if name in KERNELS or str(args['Task Type']).upper() in COMPUTE_TYPES:
                tasks.append(dict(name=name, stream=stream, begin=interval[0], end=interval[1],
                                  task_id=str(args['Task Id']), batch_id=str(args.get('Batch Id', ''))))

    missing_marks = sorted(MARKERS - markers.keys())
    if missing_marks:
        raise ValueError(f'Fusion replay missing device markers={missing_marks}')
    streams = {stream for copies in markers.values() for stream, _, _ in copies}
    if len(streams) != 1 or any(len(copies) != 1 for copies in markers.values()):
        raise ValueError('Fusion phase markers do not describe exactly one stream replay')
    (owner,) = streams
    result = []
    last_end = Decimal('-Infinity')
    # CANN emits decimal microseconds. Allow 1 ns for timestamp rounding.
    tolerance = Decimal('0.001')
    for stage, kernel, name in ROUTE_PHASES[cold_route]:
        begin_mark = next(iter(markers[f'KVIP_LIFETCH_{stage}_BEGIN']))[2]
        end_mark = next(iter(markers[f'KVIP_LIFETCH_{stage}_END']))[2]
        if begin_mark < last_end - tolerance or end_mark <= begin_mark:
            raise ValueError(f'{kernel} has out-of-order device marks on stream {owner}')
        enclosed = [task for task in tasks if task['stream'] == owner
                    and begin_mark <= task['begin'] + tolerance and end_mark >= task['end'] - tolerance]
        named = [task for task in tasks if task['name'] == kernel]
        if named and any(task not in enclosed for task in named):
            raise ValueError(f'{kernel} is outside its device marks on stream {owner}')
        if any(task['name'] in KERNELS - {kernel} for task in enclosed):
            raise ValueError(f'{kernel} window encloses a different named kernel')
        # CANN may show the task type instead of the op name for direct
        # launches. Only accept a unique compute task within the marked slot.
        if not named and len({(task['task_id'], task['batch_id']) for task in enclosed}) != 1:
            raise ValueError(f'{kernel} has no unique compute task between its device marks')
        selected = named or enclosed
        begin = min(task['begin'] for task in selected)
        end = max(task['end'] for task in selected)
        result.append(dict(name=name, kernel=kernel, stream=owner, begin_us=float(begin), end_us=float(end),
                           duration_us=float(end - begin),
                           marker_begin_us=float(begin_mark), marker_end_us=float(end_mark),
                           task_count=len(selected),
                           task_identity='named_device_task' if named else 'unique_compute_task_between_marks'))
        last_end = end_mark
    return result


def device_business_view(vendor, intervals):
    """Copy the exact msprof device clock and add a readable kernel-stage track."""
    result = dict(vendor) if isinstance(vendor, dict) else {}
    raw = list(events(vendor))
    used_pids = {str(event.get('pid')) for event in raw if isinstance(event, dict)}
    pid = 2000000
    while str(pid) in used_pids:
        pid += 1
    raw.extend((
        dict(name='process_name', ph='M', pid=pid, tid=0, args={'name': 'KVIP fused LIFetch'}),
        dict(name='thread_name', ph='M', pid=pid, tid=1, args={'name': 'Business kernel phases'}),
    ))
    for interval in intervals:
        raw.append(dict(name=interval['name'], cat='KVIP fused kernel', ph='X', pid=pid, tid=1,
                        ts=interval['begin_us'], dur=interval['duration_us'],
                        args=interval))
    result['traceEvents'] = raw
    return result


def publish_business_views(output, vendor, source, intervals, save, *, cold_route='p2'):
    """Expose both clock domains at the root of the same checked capture."""
    source_events = events(source)
    kernels = {event['args'].get('kernel_entry') for event in source_events
               if isinstance(event, dict) and event.get('ph') == 'X'
               and isinstance(event.get('args'), dict) and event['args'].get('marked_lanes')}
    expected_kernels = {kernel for _, kernel, _ in ROUTE_PHASES[cold_route]}
    if kernels != expected_kernels or {interval['kernel'] for interval in intervals} != expected_kernels:
        raise ValueError('Source timeline is missing one or more fusion stages')
    # Keep source cycles and the CANN clock as measured. No cross-clock shifts
    # are necessary to view all of the detailed business stages together.
    source_view = dict(source)
    source_view['metadata'] = {**source.get('metadata', {}),
                               'msprof_verified_device_intervals': intervals}
    business_path = output / 'lifetch_business_trace.json'
    device_path = output / 'lifetch_msprof_trace.json'
    save(business_path, source_view)
    save(device_path, device_business_view(vendor, intervals))
    return business_path, device_path


def verified_vendor_timeline(paths, *, cold_route='p2'):
    accepted, failures = [], {}
    for path in paths:
        try:
            document = json.loads(Path(path).read_text())
            intervals = require_fused_replay(document, cold_route=cold_route)
        except (ValueError, OSError) as error:
            failures[str(path)] = str(error)
        else:
            accepted.append((path, document, intervals))
    if not accepted:
        raise ValueError(f'No msprof timeline contains a complete scoped fusion replay: {failures}')
    if len(accepted) != 1:
        raise ValueError('Multiple complete fusion replays in one msprof capture')
    return accepted[0]
