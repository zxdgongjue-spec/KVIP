#!/usr/bin/env python3
"""Read one G2 level-3 run; print aggregate counters and three critical cores."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests/operators/lifetch/support'))
from diagnostics import diagnostic_table


def stats(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return [statistics.mean(values), statistics.median(values), min(values), max(values), len(values)] if values else [None]*5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path('tests/results/lifetch'))
    args = parser.parse_args()
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
                table('UNPROFILED full graph', ['Path','Check','Mean','p50','p99','N'],
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
