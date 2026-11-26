# SPDX-License-Identifier: Apache-2.0
"""Verify and label one CANN-exported fused replay."""

import argparse
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
PHASES_DEDICATED = (
    PHASES[0],
    ('THRESHOLD', 'v9_li_kernel_threshold_dedicated_service', PHASES[1][2]),
    ('COLD_SERVICE', 'v9_li_kernel_cold_dedicated_service', 'Cold Score + dedicated Service'),
    PHASES[3],
)
PHASES_STATIC_READY = (
    ('HOT', 'v9_li_kernel_hot_ready_service', PHASES[0][2]),
    PHASES_DEDICATED[1],
    ('COLD_SERVICE', 'v9_li_kernel_cold_static_ready_service', 'Cold Score + fixed Service'),
    PHASES[3],
)
PHASES_DYNAMIC = (
    PHASES_STATIC_READY[0],
    ('COLD_SERVICE', 'v9_li_kernel_dynamic_service', 'Resident Top2048 + dynamic Cold Score and Service'),
    PHASES[3],
)
SMALL_COLD_PHASES = (*PHASES_COLD_P4[:2],
                     ('COLD_SERVICE', 'v9_li_kernel_cold_ready_partitioned', PHASES[2][2]), PHASES[3])
SMALL_COLD_SEPARATE_PHASES = (*PHASES_COLD_P4[:2],
    ('COLD_SERVICE', 'v9_li_kernel_cold_separate_service', 'Cold P4 + separate Service'), PHASES[3])
RESIDENT_HIGH = ('RESIDENT_HIGH', 'v9_li_kernel_resident_topk_high', 'Resident high-byte histogram T2')
SMALL_TOPK_PHASES = (PHASES[0], RESIDENT_HIGH, *PHASES[1:])
READY_SERVICE_ROUTES = ('static_ready', 'dynamic')
ROUTE_PHASES = {'p2': PHASES, 'cold_p4': PHASES_COLD_P4,
                'full_p2': PHASES_UNIFIED, 'full_uniform': PHASES_UNIFIED,
                'dedicated': PHASES_DEDICATED, 'static_ready': PHASES_STATIC_READY,
                'dynamic': PHASES_DYNAMIC, 'small_cold': SMALL_COLD_PHASES, 'small_topk': SMALL_TOPK_PHASES,
                'small_cold_separate': SMALL_COLD_SEPARATE_PHASES}
MARKERS = {f'KVIP_LIFETCH_{stage}_{side}' for stage, _, _ in SMALL_TOPK_PHASES for side in ('BEGIN', 'END')}
KERNELS = {kernel for phases in ROUTE_PHASES.values() for _, kernel, _ in phases}
COMPUTE_TYPES = {'AI_CORE', 'AI_VECTOR_CORE', 'MIX_AIC', 'MIX_AIV', 'AIV_SQE', 'FUSION'}
DEPENDENCIES = (
    'Resident reads the committed Hot Score from GM',
    'Cold consumes resident thresholds. W0 Service consumes the resident TopK state.',
    'Terminal consumes Cold scores and the incremental TopK/Service state. Resolve follows final per-Q TopK.',
)


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

    expected = {f'KVIP_LIFETCH_{stage}_{side}' for stage, _, _ in ROUTE_PHASES[cold_route]
                for side in ('BEGIN', 'END')}
    missing_marks = sorted(expected - markers.keys())
    unexpected_marks = sorted(markers.keys() - expected)
    if missing_marks or unexpected_marks:
        raise ValueError(f'Fusion replay missing device markers={missing_marks}; '
                         f'unexpected device markers={unexpected_marks}')
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


def timed_gap_activity(begin, end, tasks):
    """Split a device gap by positive-duration tasks on its measured stream."""
    cuts = sorted({begin, end, *(edge for task in tasks for edge in (task['begin_us'], task['end_us']))})
    totals = dict.fromkeys(('profile_mark_only', 'other_task_only', 'overlap', 'no_timed_task'), 0.0)
    segments = []
    for start, stop in zip(cuts, cuts[1:]):
        active = [task for task in tasks if task['begin_us'] <= start and stop <= task['end_us']]
        mark = any(task['name'] in MARKERS for task in active)
        other = any(task['name'] not in MARKERS for task in active)
        if mark and other:
            status, label = 'overlap', 'Overlapping mark and device task'
        elif mark:
            status, label = 'profile_mark_only', 'Profiling mark'
        elif other:
            status, label = 'other_task_only', 'Other same-stream device task'
        else:
            status, label = 'no_timed_task', 'No timed device task recorded'
        duration = stop - start
        totals[status] += duration
        segments.append(dict(name=label, status=status, begin_us=start, end_us=stop, duration_us=duration,
                             tasks=[dict(name=task['name'], task_id=task['task_id']) for task in active]))
    return dict(**{status + '_us': value for status, value in totals.items()}, segments=segments)


def device_handoffs(vendor, intervals):
    """Partition same-stream kernel gaps at the existing device markers."""
    device_tasks = []
    for event in events(vendor):
        if not isinstance(event, dict):
            continue
        args, name = event.get('args'), event.get('name')
        if not isinstance(args, dict) or not isinstance(name, str):
            continue
        if (not {'Physic Stream Id', 'Task Id'}.issubset(args)
                or (name not in MARKERS and 'Task Type' not in args)):
            continue
        interval = get_interval(event)
        if interval is not None:
            device_tasks.append((name, args, interval, event.get('ph') == 'X'))

    result = []
    dependencies = DEPENDENCIES
    if len(intervals) == 3:
        dependencies = (DEPENDENCIES[0], DEPENDENCIES[2])
    elif len(intervals) == 5:
        dependencies = (DEPENDENCIES[0], 'Resident merges the two high-byte histograms per query', *DEPENDENCIES[1:])
    for left, right, dependency in zip(intervals, intervals[1:], dependencies):
        stream = left['stream']
        if stream != right['stream']:
            raise ValueError('Cannot attribute an inter-kernel gap across streams')
        begin, end = left['end_us'], max(left['end_us'], right['begin_us'])
        # Tolerate CANN timestamp rounding at adjoining task boundaries.
        end_mark = min(end, max(begin, left['marker_end_us']))
        begin_mark = min(end, max(end_mark, right['marker_begin_us']))
        segments = [
            dict(name=name, begin_us=a, end_us=b, duration_us=b-a)
            for name, a, b in (
                ('Previous kernel end to END mark', begin, end_mark),
                ('Between device marks', end_mark, begin_mark),
                ('BEGIN mark to next kernel start', begin_mark, end),
            )
        ]
        enclosed, timed = {}, {}
        for name, args, (start, stop), complete in device_tasks:
            if str(args['Physic Stream Id']) != stream:
                continue
            start, stop = float(start), float(stop)
            task_id = str(args['Task Id'])
            task = dict(name=name, task_id=task_id, task_type=args.get('Task Type'),
                        begin_us=start, end_us=stop)
            if begin <= start <= stop <= end:
                enclosed[name, task_id, start, stop] = task
            clip_start, clip_stop = max(begin, start), min(end, stop)
            if complete and clip_start < clip_stop:
                timed[name, task_id, clip_start, clip_stop] = dict(task, begin_us=clip_start, end_us=clip_stop)
        result.append(dict(name=f"{left['name']} to {right['name']}", stream=stream,
                           previous_kernel=left['kernel'], following_kernel=right['kernel'],
                           begin_us=begin, end_us=end, duration_us=end-begin,
                           dependency=dependency, segments=segments,
                           timed_task_activity=timed_gap_activity(begin, end, list(timed.values())),
                           observed_stream_tasks=sorted(enclosed.values(), key=lambda row: row['begin_us'])))
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
        dict(name='thread_name', ph='M', pid=pid, tid=2, args={'name': 'Same-stream kernel handoffs'}),
        dict(name='thread_name', ph='M', pid=pid, tid=3, args={'name': 'Handoffs split by device marks'}),
        dict(name='thread_name', ph='M', pid=pid, tid=4, args={'name': 'Handoff device task activity'}),
    ))
    for interval in intervals:
        raw.append(dict(name=interval['name'], cat='KVIP fused kernel', ph='X', pid=pid, tid=1,
                        ts=interval['begin_us'], dur=interval['duration_us'],
                        args=interval))
    handoffs = device_handoffs(vendor, intervals)
    for handoff in handoffs:
        if handoff['duration_us'] > 0:
            raw.append(dict(name=handoff['name'], cat='KVIP kernel handoff', ph='X', pid=pid, tid=2,
                            ts=handoff['begin_us'], dur=handoff['duration_us'], args=handoff))
        for segment in handoff['segments']:
            if segment['duration_us'] > 0:
                raw.append(dict(name=segment['name'], cat='KVIP device marks', ph='X', pid=pid, tid=3,
                                ts=segment['begin_us'], dur=segment['duration_us'],
                                args=dict(segment, handoff=handoff['name'])))
        for segment in handoff['timed_task_activity']['segments']:
            raw.append(dict(name=segment['name'], cat='KVIP device handoff activity', ph='X', pid=pid, tid=4,
                            ts=segment['begin_us'], dur=segment['duration_us'],
                            args=dict(segment, handoff=handoff['name'])))
    result['traceEvents'] = raw
    result['kvip_inter_kernel_handoffs'] = handoffs
    return result


def publish_business_views(output, vendor, source, intervals, save, *, cold_route='p2'):
    """Expose both clock domains at the root of the same checked capture."""
    source_events = events(source)
    kernels = {event['args'].get('kernel_entry') for event in source_events
               if isinstance(event, dict) and event.get('ph') == 'X'
               and isinstance(event.get('args'), dict) and event['args'].get('marked_lanes')}
    expected_kernels = {kernel for _, kernel, _ in ROUTE_PHASES[cold_route]}
    source_kernels = expected_kernels - {RESIDENT_HIGH[1]} if cold_route == 'small_topk' else expected_kernels
    if kernels != source_kernels or {interval['kernel'] for interval in intervals} != expected_kernels:
        raise ValueError('Source timeline is missing one or more fusion stages')
    # Keep source cycles and the CANN clock as measured. No cross-clock shifts
    # are necessary to view all of the detailed business stages together.
    source_view = dict(source)
    source_view['metadata'] = {**source.get('metadata', {}),
                               'msprof_verified_device_intervals': intervals}
    business_path = output / 'lifetch_business_trace.json'
    device_path = output / 'lifetch_msprof_trace.json'
    device_view = device_business_view(vendor, intervals)
    save(business_path, source_view)
    save(device_path, device_view)
    save(output / 'lifetch_handoff_report.json', dict(
        native_device_handoffs=device_view['kvip_inter_kernel_handoffs'],
        source_business_handoffs=source_view['metadata'].get('source_handoffs', []),
        timing_note=('Native intervals use original msprof device timestamps and can contain profiling '
                     'marks and dispatch. Source intervals use GetSystemCycle, with uncovered intervals '
                     'meaning unclassified time in the recorded scopes. Interpret each clock separately.'),
    ))
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


def main(argv=None):
    parser = argparse.ArgumentParser(description='Label a previously exported, marked LIFetch msprof timeline.')
    parser.add_argument('native_trace', type=Path, help='CANN msprof_*.json export')
    parser.add_argument('--route', choices=ROUTE_PHASES, default='p2')
    parser.add_argument('--out', type=Path, help='new JSON file with handoff tracks')
    parser.add_argument('--handoff-report', type=Path, help='compact report in the native device clock')
    args = parser.parse_args(argv)
    if args.out is None and args.handoff_report is None:
        parser.error('Set --out or --handoff-report')
    if args.out and args.handoff_report and args.out.resolve() == args.handoff_report.resolve():
        parser.error('Use different paths for --out and --handoff-report')
    try:
        _, vendor, intervals = verified_vendor_timeline([args.native_trace], cold_route=args.route)
        if args.out is not None:
            labeled = device_business_view(vendor, intervals)
            handoffs = labeled['kvip_inter_kernel_handoffs']
            with args.out.open('x') as output:
                json.dump(labeled, output, ensure_ascii=False)
        else:
            handoffs = device_handoffs(vendor, intervals)
        if args.handoff_report is not None:
            with args.handoff_report.open('x') as output:
                json.dump(dict(clock='CANN device timestamps in microseconds', cold_route=args.route,
                               native_device_handoffs=handoffs), output, ensure_ascii=False, indent=2)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for handoff in handoffs:
        print(f"{handoff['name']}  {handoff['duration_us']:.2f} us")


if __name__ == '__main__':
    main()
