# SPDX-License-Identifier: Apache-2.0
"""Raw fixed-fixture NPU timeline, without retired terminal scratch."""
import json
import math
import re
from collections import Counter
from pathlib import Path
from reporting import atomic_json


def replay_kernel_records(events, scope_name):
    scopes = [e for e in events if e.get('ph') == 'X' and e.get('name') == scope_name]
    if len(scopes) != 1:
        return dict(status='SCOPE_UNAVAILABLE', scope_count=len(scopes), kernels=[])
    scope = scopes[0]
    origin, duration = float(scope['ts']), float(scope['dur'])
    records = []
    for e in events:
        if e.get('ph') != 'X' or 'ts' not in e or 'dur' not in e:
            continue
        cat = str(e.get('cat') or '').lower()
        name = str(e.get('name') or '')
        norm = re.sub('[^a-z0-9]', '', name.lower())
        if any(x in cat for x in ('cpu', 'python', 'runtime', 'launch')):
            continue
        known = norm.startswith(('v9likernel', 'v9parallelterminal', 'v9earlyfetchkernel',
                                 'v9initkernel', 'quantlightningindexer',
                                 'a5lidufusedbatchgetpreparemtp', 'kvafexecutebatchgetaivkernel'))
        if not ('kernel' in cat or (known and (not cat or 'task' in cat or 'npu' in cat))):
            continue
        begin, elapsed = float(e['ts']) - origin, float(e['dur'])
        if not math.isfinite(begin) or not math.isfinite(elapsed) or elapsed <= 0:
            continue
        if begin < 0 or begin + elapsed > duration:
            continue
        records.append(dict(name=name, category=cat, start_us=begin, duration_us=elapsed,
                            end_us=begin+elapsed, pid=e.get('pid'), tid=e.get('tid')))
    records.sort(key=lambda r: (r['start_us'], r['end_us']))
    if not records:
        return dict(status='DEVICE_KERNELS_UNAVAILABLE', scope_us=duration, kernels=[])
    intervals = []
    for r in records:
        a, b = r['start_us'], r['end_us']
        if intervals and a <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], b)
        else:
            intervals.append([a, b])
    busy = sum(b-a for a, b in intervals)
    span = intervals[-1][1] - intervals[0][0]
    return dict(status='RAW_KERNELS_FOUND', scope_us=duration, kernels=records,
                name_counts=dict(Counter(r['name'] for r in records)),
                observed_kernel_union_us=busy, observed_kernel_span_us=span,
                observed_gap_us=span-busy, note='Observed intervals only; coverage not assumed complete.')


def capture_fixed_graph(torch, stream, graph, reset, verify, rounds, folder, path_name):
    import torch_npu
    from torch.profiler import record_function

    folder = Path(folder)
    samples = []
    profile = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False)
    with profile:
        for step in range(rounds):
            reset()
            stream.synchronize()
            start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            name = f'nextgen_fixed_{path_name}_{step}'
            with record_function(name):
                start.record(stream)
                graph.replay()
                end.record(stream)
                end.synchronize()
            assert verify(), f'Profile replay correctness failed: {name}'
            samples.append(dict(name=name, sample=step, event_us=start.elapsed_time(end)*1000))
    trace = folder / (path_name + '_fixed_trace.json')
    raw_samples = folder / (path_name + '_fixed_profile_samples.json')
    atomic_json(raw_samples, samples)
    profile.export_chrome_trace(str(trace.resolve()))
    data = json.loads(trace.read_text())
    events = data['traceEvents'] if isinstance(data, dict) else data
    assert isinstance(events, list) and events, 'Empty profiler trace'
    for row in samples:
        row['timeline'] = replay_kernel_records(events, row['name'])
    attribution = folder / (path_name + '_fixed_timeline.json')
    atomic_json(attribution, samples)
    captured = sum(row['timeline']['status'] == 'RAW_KERNELS_FOUND' for row in samples)
    result = dict(trace=str(trace), samples=str(raw_samples), timeline=str(attribution),
                  profile_rounds=rounds, kernel_scopes_found=captured, attribution_complete=False)
    print('FIXED_PROFILE_CAPTURE', path_name, json.dumps(result), flush=True)
    for row in samples:
        print('FIXED_PROFILE_TIMELINE', path_name, row['sample'], row['event_us'],
              [(x['name'], round(x['duration_us'], 3)) for x in row['timeline']['kernels']], flush=True)
    return result
