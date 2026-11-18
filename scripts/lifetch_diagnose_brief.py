#!/usr/bin/env python3
"""Read one G2 level-3 run; print aggregate counters and three critical cores."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
import re
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests/operators/lifetch/support'))
from diagnostics import diagnostic_table


def stats(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return [statistics.mean(values), statistics.median(values), min(values), max(values), len(values)] if values else [None]*5



def li_kernel_breakdown(capture):
    """Read the same measured graph trace, preserving individual LI launches."""
    source = Path(capture['source'])
    trace_path = source.with_name('trace.json')
    if not trace_path.is_file():
        return {}, f'missing trace: {trace_path}'
    document = json.loads(trace_path.read_text())
    events = document.get('traceEvents', []) if isinstance(document, dict) else document
    scopes = [e for e in events if e.get('ph') == 'X' and e.get('name') == 'lifetch_measured_graph']
    if len(scopes) != 1:
        return {}, f'expected one measured graph scope, found {len(scopes)}'
    scope = scopes[0]
    origin = Decimal(str(scope['ts']))
    duration = float(scope['dur'])
    labels = ('kernel 0: hot/ABC', 'kernel 1: resident TopK',
              'kernel 2: cold/prefetch', 'kernel 3: final TopK')
    calls = capture.get('graph_calls')
    if not isinstance(calls, int) or calls < 1:
        return {}, 'missing measured graph invocation count'
    by_name = defaultdict(list)
    for event in events:
        name = str(event.get('name', '')).lower()
        category = str(event.get('cat') or '').lower()
        if (event.get('ph') != 'X' or (category and 'kernel' not in category)
                or not re.fullmatch(r'v9_li_kernel(?:_\d+)*', name)):
            continue
        start = float(Decimal(str(event['ts'])) - origin)
        end = start + float(event.get('dur', 0))
        if 0 <= start <= end <= duration:
            by_name[name].append((start, end))
    # CANN may add one common numeric dispatch suffix to every function:
    # e.g. kernel_0, kernel_1_0, kernel_2_0, kernel_3_0. A single
    # name such as kernel_1 is ambiguous without the complete family, so
    # accept a convention only when the whole quartet has the expected count
    # and the four launches occur in order for each graph invocation.
    base = 'v9_li_kernel'
    suffixes = {''}
    for name in by_name:
        match = re.fullmatch(r'v9_li_kernel_1(_\d+(?:_\d+)*)', name)
        if match:
            suffixes.add(match.group(1))
    # Some profiler exports number all four kernels directly, yielding
    # kernel_0, kernel_1, kernel_2, kernel_3. This is distinct from CANN's
    # common suffix convention and must be resolved using the entire family.
    families = [[base + f'_{i}' for i in range(4)]]
    families += [[base + suffix] + [base + f'_{i}' + suffix for i in range(1, 4)]
                 for suffix in sorted(suffixes)]
    # Some CANN traces append the launch ordinal separately to each symbol.
    families.append([base + '_0'] + [base + f'_{i}_{i}' for i in range(1, 4)])
    valid = []
    for names in families:
        if set(names) != set(by_name) or any(len(by_name[name]) != calls for name in names):
            continue
        arrays = [sorted(by_name[name]) for name in names]
        if any(not all(arrays[j][i][1] <= arrays[j+1][i][0] for j in range(3))
               for i in range(calls)):
            continue
        if any(arrays[3][i][1] > arrays[0][i+1][0] for i in range(calls-1)):
            continue
        valid.append((names, arrays))
    if len(valid) != 1:
        found = {name: len(rows) for name, rows in sorted(by_name.items())}
        return {}, (f'cannot uniquely identify four LI kernels for {calls} calls; '
                    f'matching families={len(valid)}, observed={found}')
    _, arrays = valid[0]
    return {label: [end-start for start, end in rows]
            for label, rows in zip(labels, arrays)}, ''

def bench_li(root):
    profiles = [p for p in root.rglob('graph_profile.json')
                if 'bench' in p.parts and 'fused' in p.parts]
    for fused in sorted(profiles, key=lambda p: p.stat().st_mtime_ns, reverse=True):
        case = next(d for d in fused.parents if d.name == 'bench')
        config_file = case / 'config.json'
        if not config_file.is_file():
            continue
        config = json.loads(config_file.read_text())
        if (config.get('input'), config.get('group'), config.get('transport')) != ('real', 2, 'urma'):
            continue
        relative = fused.relative_to(case)
        parts = list(relative.parts)
        parts[parts.index('fused')] = 'serial'
        serial = case.joinpath(*parts)
        measured = json.loads(fused.read_text())['timing']
        calls = measured['graph_calls']
        phases, issue = li_kernel_breakdown(dict(source=str(fused), graph_calls=calls))
        if issue:
            raise SystemExit(f'{fused}: {issue}')
        reference = json.loads(serial.read_text())['timing']
        if reference['graph_calls'] != calls:
            raise SystemExit('Serial and fused invocation counts differ')
        total = sum(statistics.mean(values) for values in phases.values())
        reported = measured['stage_mean_us']['li']
        if abs(total - reported) > 0.01:
            raise SystemExit(f'Four-kernel sum {total:.2f} differs from bench LI {reported:.2f}')
        print(f'Fused: {fused}\nSerial: {serial}')
        print(diagnostic_table('Fused LI kernels (us/call)',
                               ['Kernel', 'Mean', 'p50', 'Min', 'Max', 'N'],
                               [[label, *stats(values)] for label, values in phases.items()]))
        baseline = reference['stage_mean_us']['li']
        print(f'Fused sum {total:.2f} us | bench LI {reported:.2f} us | '
              f'serial LI {baseline:.2f} us | fused - serial {total-baseline:+.2f} us')
        return
    raise SystemExit(f'No real G2 URMA bench graph profile under {root}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path('tests/results/lifetch'))
    parser.add_argument("--bench-li", action="store_true", help="summarize the latest real G2 URMA bench LI kernels")
    args = parser.parse_args()
    if args.bench_li:
        return bench_li(args.root)
    matches = []
    for path in args.root.rglob('diagnostic.json'):
        config_path = path.parent / 'config.json'
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        if config.get('input') == 'real' and config.get('group') == 2 and config.get('level') == 3:
            matches.append((path, config))
    if not matches:
        raise SystemExit('No real G2 level3 diagnostic.json found; pass the result root explicitly.')
    path, config = max(matches, key=lambda pair: pair[0].stat().st_mtime_ns)
    result = json.loads(path.read_text())
    lines = [f'SOURCE {path}', f'TRACE {result.get("trace")}',
             'CONFIG '+json.dumps({k:config.get(k) for k in ('transport','device','trace_root','group',
                 'round_range','prefetch_tokens','prefetch_rank_start','intermediate_fetches','warmup','iterations')})]
    def table(title, headers, rows):
        lines.append(diagnostic_table(title, headers, rows))
    for name in ('serial', 'fused'):
        for p in path.parent.rglob('path_result.json'):
            data = json.loads(p.read_text())
            if data.get('path') == name:
                t = data.get('all_rounds', {})
                if t.get('timing_protocol') == 'one_profiled_graph':
                    table('Full graph: event and profiler from same execution (us)',
                          ['Path','Check','Event mean','Profiler mean','Profiler p50','Profiler p99','N'],
                          [[name,data.get('correctness'),*[t.get(k) for k in
                            ('event_mean_us','profiler_mean_us','p50_us','p99_us','samples')]]])
                else:
                    table('Full graph: legacy timing (us)', ['Path','Check','Mean','p50','p99','N'],
                          [[name,data.get('correctness'),*[t.get(k) for k in ('mean_us','p50_us','p99_us','samples')]]])
                if data.get('failures'):
                    lines.append('FAILURES '+json.dumps(data['failures'][:3]))
    for capture in result.get('captures', []):
        label = capture.get('path', 'fused')
        samples = capture['samples']
        values = defaultdict(list)
        for s in samples:
            a = s.get('accounting', {})
            if 'error' in a:
                continue
            for k,v in a.get('stages_us', {}).items():
                values[k].append(v)
            for k,v in a.items():
                if k.endswith('_us') and isinstance(v,(int,float)):
                    values[k].append(v)
        table(f'{label}: PROFILER stages/accounting (us; same-pass, not detailed stamps)',
              ['Metric','Mean','p50','Min','Max','N'], [[k,*stats(v)] for k,v in values.items()])
        if label == 'fused' and capture.get('graph_accounting') and config.get('transport') == 'urma':
            li_phases, issue = li_kernel_breakdown(capture)
            if li_phases:
                table('LI kernel durations: same profiled graph (us)',
                      ['Kernel','Mean','p50','Min','Max','N'],
                      [[k,*stats(v)] for k,v in li_phases.items()])
            if issue:
                lines.append('LI KERNEL TRACE: '+issue+'; check raw event names and source trace.')
        totals = defaultdict(list)
        for s in samples:
            for k,v in s.get('algorithm', {}).get('totals', {}).items():
                if isinstance(v,(int,float)):
                    totals[k].append(v)
            for k in ('prefetch','final'):
                if isinstance(s.get(k),(int,float)):
                    totals[k].append(s[k])
        if totals:
            table(f'{label}: TRAFFIC counts', ['Metric','Mean','p50','Min','Max','N'],
                  [[k,*stats(v)] for k,v in totals.items()])
        bandwidth = defaultdict(list)
        for s in samples:
            for scope in ('prefetch_bandwidth','final_get_bandwidth','combined_observed_bandwidth'):
                for k in ('bytes','span_us','GBps'):
                    v = s.get(scope,{}).get(k)
                    if isinstance(v,(int,float)):
                        bandwidth[f'{scope}.{k}'].append(v)
        if bandwidth:
            table('DETAIL pass: bandwidth', ['Metric','Mean','p50','Min','Max','N'],
                  [[k,*stats(v)] for k,v in bandwidth.items()])
        detailed = [s for s in samples if s.get('timeline')]
        if not detailed:
            continue
        last = Counter((s['timeline']['last_block'], next(c['role'] for c in s['timeline']['cores']
                        if c['block']==s['timeline']['last_block'])) for s in detailed)
        table('Last finisher frequency (top 5)', ['Block','Role','Count','Replays'],
              [[*key,count,len(detailed)] for key,count in last.most_common(5)])
        roles = defaultdict(list)
        for s in detailed:
            grouped = defaultdict(list)
            for c in s['timeline']['cores']:
                grouped[c['role'].split()[0]].append(c['end_us'])
            for role,ends in grouped.items():
                roles[role].append(max(ends))
        table('DETAIL pass: latest end per role per replay (us)', ['Role','Mean','p50','Min','Max','N'],
              [[k,*stats(v)] for k,v in roles.items()])
        ordered = sorted(detailed,key=lambda s:s['timeline']['observed_span_us'])
        sample = ordered[len(ordered)//2]
        timeline = sample['timeline']
        lines.append(f"REPRESENTATIVE detail repeat={sample.get('detail_repeat',sample.get('repeat'))} "
                     f"span={timeline['observed_span_us']:.2f} us; core times relative to fused tail start. "
                     'Separate instrumented pass; nested intervals must not be added.')
        selected = {timeline['last_block']}
        for prefix in ('index_update','fetch'):
            cores=[c for c in timeline['cores'] if c['role'].startswith(prefix)]
            if cores:
                selected.add(max(cores,key=lambda c:c['end_us'])['block'])
        for c in timeline['cores']:
            if c['block'] not in selected:
                continue
            table(f"block={c['block']} {c['role']} end={c['end_us']:.2f} us", ['Phase','Start','End','Duration'],
                  [[t['name'],t['start_us'],t['end_us'],t['duration_us']] for t in c['stages']])
    if result.get('missing'):
        lines.append('TRACE ISSUES '+json.dumps(result['missing'][:8]))
    output = path.parent / 'g2_key_results.log'
    output.write_text('\n\n'.join(lines)+'\n')
    print('\n\n'.join(lines))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
