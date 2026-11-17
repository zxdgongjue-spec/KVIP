#!/usr/bin/env python3
"""Extract one Resolve/Fetch timeline from an existing level-3 run (CPU only)."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics


def fmt(value):
    return '—' if value is None else f'{value:.2f}'


def print_critical_requests(cores):
    """Compare per-request stages from a single detailed replay."""
    resolvers = {}
    outputs = {}
    fetch_groups = {1: [], 4: []}
    for core in cores:
        role = core.get('role', '')
        fetch_match = re.fullmatch(r'fetch Q([14]) lane(\d+)', role)
        if fetch_match:
            fetch_groups[int(fetch_match.group(1))].append((int(fetch_match.group(2)), core.get('requests')))
            continue
        match = re.fullmatch(r'(resolver\+finalize|output_pack) b(\d+)', role)
        if not match:
            continue
        target = resolvers if match.group(1) == 'resolver+finalize' else outputs
        target[int(match.group(2))] = core

    print('\nPack end is this request\'s descriptor-packing stage end; '
          'it is not group ready, a doorbell, or CQ completion.')
    for q in (1, 4):
        rows = []
        for request, output in outputs.items():
            resolver = resolvers.get(request)
            if resolver is None:
                continue
            output_stages = {s['name']: s for s in output.get('stages', [])}
            resolver_stages = {s['name']: s for s in resolver.get('stages', [])}
            pack = output_stages.get(f'Q{q}.locations_and_pack')
            if not pack or not isinstance(pack.get('end_us'), (int, float)):
                continue
            rows.append((pack['end_us'], request, resolver, resolver_stages, output_stages, pack))
        rows.sort(key=lambda row: (-row[0], row[1]))

        def print_rows(selected):
            print('Request | Resolve start | Resolve end | Lookup duration | Pack start | Pack end | Resolver end')
            if not selected:
                print('No matching per-request stages in this replay.')
            for _, request, resolver, resolver_stages, output_stages, pack in selected:
                resolve = resolver_stages.get(f'Q{q}.resolve', {})
                lookup = output_stages.get(f'Q{q}.logical_lookup', {})
                print(f'{request} | {fmt(resolve.get("start_us"))} | {fmt(resolve.get("end_us"))} | '
                      f'{fmt(lookup.get("duration_us"))} | {fmt(pack.get("start_us"))} | '
                      f'{fmt(pack.get("end_us"))} | {fmt(resolver.get("end_us"))}')

        groups = sorted(fetch_groups[q], key=lambda group: group[0])
        missing_groups = not groups
        for lane, requests in groups:
            if not isinstance(requests, list) or not requests or not all(type(r) is int for r in requests):
                print(f'\nQ{q} lane{lane}: request group unknown (requests field missing or empty)')
                missing_groups = True
                continue
            print(f'\nQ{q} lane{lane}: up to 2 requests with latest pack end')
            print_rows([row for row in rows if row[1] in requests][:2])
        if missing_groups:
            print(f'\nQ{q}: group unknown; up to 4 requests with latest pack end globally')
            print_rows(rows[:4])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='result directory or diagnostic.json')
    parser.add_argument('--critical-requests', action='store_true',
                        help='print the slowest per-Fetch-lane Q1/Q4 descriptor stages')
    args = parser.parse_args()
    paths = [args.run] if args.run.is_file() else list(args.run.rglob('diagnostic.json'))
    if not paths:
        raise SystemExit('No diagnostic.json found in the supplied path')
    path = max(paths, key=lambda p: p.stat().st_mtime_ns)
    result = json.loads(path.read_text())
    config_path = path.parent / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    samples = [s for cap in result.get('captures', []) if cap.get('path', 'fused') == 'fused'
               for s in cap.get('samples', []) if s.get('timeline')]
    if not samples:
        raise SystemExit(f'{path}: no per-core timeline; run diagnose --level 3')
    samples.sort(key=lambda s: s['timeline']['observed_span_us'])
    sample = samples[len(samples) // 2]
    timeline = sample['timeline']
    cores = timeline['cores']
    mhz = timeline.get('clock_mhz')
    print(f'SOURCE {path}')
    print('CONFIG ' + json.dumps({k: config.get(k) for k in
          ('transport', 'input', 'group', 'round_range', 'batches', 'prefetch_tokens',
           'prefetch_rank_start', 'intermediate_fetches', 'warmup', 'iterations')}))
    print(f'TRACE {result.get("trace")} | detailed replays={len(samples)} | '
          f'median-span replay={sample.get("detail_repeat", sample.get("repeat"))} | '
          f'span={fmt(timeline["observed_span_us"])} us | clock={fmt(mhz)} MHz')
    print('All timestamps below: us from earliest Resolve-kernel core entry; '
          'separate detailed replay, not the benchmark event.')
    if args.critical_requests:
        print_critical_requests(cores)
        return
    frequencies = Counter((c['block'], c['role']) for s in samples
                          for c in s['timeline']['cores']
                          if c['block'] == s['timeline']['last_block'])
    print('LAST FINISHERS ' + ', '.join(f'{b}:{role}={n}/{len(samples)}'
                                         for (b, role), n in frequencies.most_common(4)))
    for key in ('initial_cold', 'prefetch', 'final', 'unused_prefetch'):
        values = [s.get('algorithm', {}).get('totals', {}).get(key) for s in samples]
        values = [v for v in values if isinstance(v, (int, float))]
        if values:
            print(f'TRAFFIC {key} mean={statistics.mean(values):.2f} '
                  f'min={min(values)} max={max(values)} n={len(values)}')
    print('Traffic values describe performance-pass samples, not this detailed replay.')

    print('\nFETCH CORES (Get includes posting and CQ; counters are nested, not additive)')
    print('Core | Requests:reads | last ready | Get start→end | post/CQ us | conflict Get | conflict post/CQ | index wait | end')
    for core in cores:
        if not core['role'].startswith('fetch '):
            continue
        stages = {s['name']: s for s in core.get('stages', [])}
        ready = max((row['end_us'] for name, row in stages.items()
                     if name.startswith('request') and name.endswith('.wait_Q_ready')), default=None)
        safe, conflict = stages.get('safe.KVAF_Get'), stages.get('conflict.KVAF_Get')
        wait = stages.get('finalize.wait_index')
        counters = core.get('KVAF_counters', {})
        def pair(row):
            return f'{fmt(row["start_us"])}→{fmt(row["end_us"])}' if row else '—'
        def counter(key):
            value = counters.get(key)
            return value / mhz if isinstance(value, (int, float)) and mhz else None
        requests = core.get('requests', [])
        counts = core.get('published_misses', [])
        reads = ','.join(f'{r}:{n}' for r, n in zip(requests, counts)) or '—'
        print(f'{core["block"]} {core["role"]} | {reads} | {fmt(ready)} | {pair(safe)} | '
              f'{fmt(counter("safe_post_cycles"))}/{fmt(counter("safe_CQ_cycles"))} | '
              f'{pair(conflict)} | {fmt(counter("conflict_post_cycles"))}/'
              f'{fmt(counter("conflict_CQ_cycles"))} | {pair(wait)} | {fmt(core["end_us"])}')
    print('last ready is the latest request wait end, not the first doorbell; CQ includes polling.')

    resolver = [c for c in cores if c['role'].startswith('resolver+finalize ')]
    if not resolver:
        raise SystemExit('No resolver core in detailed timeline')
    slow = max(resolver, key=lambda c: c['end_us'])
    match = re.search(r'\bb(\d+)$', slow['role'])
    request = match.group(1) if match else None
    related = [c for c in cores if c is slow or (request is not None and
               c['role'] in (f'index_update b{request}', f'output_pack b{request}'))]
    wanted = ('input_and_bitmap', 'initialize', 'prefetch_CQ_join', 'prefetch_completion_join',
              'prefetch_residency', 'residency', 'status', 'metadata', 'apply.prefetch',
              'wait.prefetch_publication', 'input_and_output_masks')
    print(f'\nSLOWEST RESOLVER AND ITS REQUEST (b{request})')
    for core in related:
        print(f'{core["block"]} {core["role"]} end={fmt(core["end_us"])} us')
        for stage in core.get('stages', []):
            name = stage['name']
            if (name in wanted or name.startswith(('Q1.', 'Q2.', 'Q3.', 'Q4.',
                                                    'wait.', 'apply.Q'))):
                print(f'  {name:<27} {fmt(stage["start_us"]):>7}→{fmt(stage["end_us"]):<7} '
                      f'({fmt(stage["duration_us"])} us)')


if __name__ == '__main__':
    main()
