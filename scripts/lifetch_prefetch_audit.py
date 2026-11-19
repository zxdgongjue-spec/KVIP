#!/usr/bin/env python3
"""Audit existing G2 diagnostic counts and original capture prefetch identities."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests/operators/lifetch/support'))
from diagnostics import diagnostic_table
from workloads import load_score_capture, select_series



def validate_transfers(capture):
    """Validate only the records used here, not optional timing/threshold arrays."""
    a, g = capture.arrays, capture.manifest['geometry']
    records = np.asarray(a['post.prefetch_records'])
    if records.ndim != 2 or records.shape[1] != 3 or records.dtype.kind not in 'iu':
        raise ValueError(f"Unsupported prefetch_records: shape={records.shape}, dtype={records.dtype}; expected [request, physical_position, logical_token]")
    if (np.any(records[:, 0] < 0) or np.any(records[:, 0] >= g['batch'])
            or np.any(records[:, 1:] < 0) or np.any(records[:, 1:] >= g['history'])):
        raise ValueError('Prefetch record request/position/token outside capture bounds')
    requests = records[:, 0].astype(np.int64)
    positions = records[:, 1].astype(np.int64)
    if not np.array_equal(a['pre.logical_ids'][requests, positions], records[:, 2]):
        raise ValueError('Prefetch logical tokens differ from captured pre-Index mapping')
    # Repeated tokens are deliberately accepted: this audit counts them.
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path('tests/results/lifetch'))
    args = parser.parse_args()
    matches = []
    for p in args.root.rglob('diagnostic.json'):
        cp = p.parent / 'config.json'
        if cp.exists():
            c = json.loads(cp.read_text())
            if c.get('input') == 'real' and c.get('group') == 2:
                matches.append((p,c))
    if not matches:
        raise SystemExit('No real G2 diagnostic.json/config.json found')
    path, config = max(matches, key=lambda x:x[0].stat().st_mtime_ns)
    data = json.loads(path.read_text())
    lines = [f'RUN {path}', 'A: CURRENT RUN. One count per request/replay; no duplicate collection from totals.']
    fields = ['selected_unique','initial_cold','candidate_hot','prefetch','final','unused_prefetch']
    tag_rows = []
    exact_rows = []
    exact_records = []
    for capture in data.get('captures', []):
        if capture.get('path') == 'serial':
            continue
        samples = [s for s in capture['samples'] if s.get('algorithm',{}).get('requests')]
        for sample in samples:
            for request in sample['algorithm']['requests']:
                for stage in ('prefix_tag_audit', 'final_tag_audit'):
                    if stage in request:
                        audit = request[stage]
                        tag_rows.append([sample.get('repeat', sample.get('round')), request['request'], stage,
                                         sum(q['stable_hot'] for q in audit),
                                         sum(q['missing_hot_tags'] for q in audit)])
                records = request.get('prefetch_transfers', [])
                if not records and request['prefetch']:
                    continue
                if 'prefetch_classes' not in request:
                    continue
                counts = request['prefetch_classes']
                exact_rows.append([sample.get('repeat', sample.get('round')), request['request'], request['prefetch'],
                                   *[counts[k] for k in ('stable_hot', 'reserved_hot', 'abc', 'cold', 'repeated')]])
                exact_records.extend(dict(repeat=sample.get('repeat', sample.get('round')), request=request['request'], **record)
                                     for record in records)
        groups = {}
        for s in samples:
            for r in s['algorithm']['requests']:
                groups.setdefault(r['request'],[]).append(r)
        table = []
        for request, rows in sorted(groups.items()):
            values = [statistics.mean(r[k] for r in rows) if all(isinstance(r.get(k),(int,float)) for r in rows) else None for k in fields]
            fraction = 100*values[2]/values[0] if values[2] is not None and values[0] else None
            table.append([request,*values,fraction,len(rows)])
        lines.append(diagnostic_table('Per-request means', ['Request',*fields,'Candidate hot / demand %','N'],table))
        totals = []
        for s in samples:
            rows = s['algorithm']['requests']
            totals.append({k:sum(r[k] for r in rows) if all(isinstance(r.get(k),(int,float)) for r in rows) else None for k in fields})
        summary = []
        for k in fields:
            values = [r[k] for r in totals if r[k] is not None]
            summary.append([k, statistics.mean(values) if values else None,min(values) if values else None,max(values) if values else None,len(values)])
        lines.append(diagnostic_table('Whole-batch counts per replay',['Metric','Mean','Min','Max','N'],summary))
        ratios=[100*r['candidate_hot']/r['selected_unique'] for r in totals if r['candidate_hot'] is not None and r['selected_unique']]
        if ratios:
            lines.append(f'Candidate hot contains {statistics.mean(ratios):.2f}% of final demand (mean over replays).')
        bounds=[max(0,r['prefetch']-r['unused_prefetch']-r['initial_cold']) for r in totals if all(r[k] is not None for k in ('prefetch','unused_prefetch','initial_cold'))]
        if bounds:
            lines.append(f'Conditional lower bound on initially-local selected prefetch: mean={statistics.mean(bounds):.2f}, min={min(bounds):.2f}, max={max(bounds):.2f}; assumes no repeated token transfers.')
    if tag_rows:
        grouped = {}
        for repeat, request, stage, stable, missing in tag_rows:
            values = grouped.setdefault((repeat, stage), [0, 0])
            values[0] += stable
            values[1] += missing
        lines.append(diagnostic_table('Hot tags: initial prefix vs final incremental TopK',
                     ['Repeat', 'Stage', 'Stable hot positions', 'Missing hot tags'],
                     [[repeat, stage, *values] for (repeat, stage), values in sorted(grouped.items())]))
    if exact_rows:
        by_repeat = {}
        for row in exact_rows:
            by_repeat.setdefault(row[0], []).append(row)
        lines.append(diagnostic_table('Current admissions: exact per-token initial-state audit',
                     ['Repeat', 'Posts', 'Stable hot', 'Reserved hot', 'ABC', 'Cold', 'Repeated tokens'],
                     [[repeat, *[sum(row[i] for row in rows) for i in range(2, 8)]]
                      for repeat, rows in sorted(by_repeat.items())]))
        records_path = path.parent / 'prefetch_token_audit.json'
        records_path.write_text(json.dumps(exact_records, indent=2) + '\n')
        lines.append(f'Every admission: {records_path}; hot_valid/reserved are pre-call snapshots. Counts reconcile with posted ordinals; this is not a hardware WQE capture.')
    else:
        lines.append('This older run has no per-transfer records; exact attribution requires a new diagnose run.')
    output=path.parent/'prefetch_hot_audit.log'
    output.write_text('\n\n'.join(lines)+'\n')
    print('\n\n'.join(lines), flush=True)
    lines = []
    lines.append('B: ORIGINAL CAPTURE (not this run). Exact recorded identities; captured algorithm may differ.')
    root = Path(config['trace_root']).resolve()
    if not root.exists():
        lines.append(f'Source trace unavailable: {root}')
    else:
        refs = select_series(root,config['group'],tuple(config['round_range']))
        rows = []
        for ref in refs:
            try:
                cap = load_score_capture(ref.manifest_path)
                records = validate_transfers(cap)
            except (ValueError, KeyError, OSError) as error:
                lines.append(f'SOURCE AUDIT FAILED {ref.manifest_path}: {error}')
                continue
            a,g = cap.arrays,cap.manifest['geometry']
            for row in range(g['batch']):
                layout = a['pre.logical_ids'][row]
                hot = set(map(int,layout[:g['resident']][a['pre.hot_valid'][row] != 0]))
                lo,hi = map(int,a['pre.tail_bounds'][row])
                tail = set(range(lo,hi)); local=hot|tail
                reserved = hot & set(map(int,layout[a['pre.lru_slots'][row]]))
                demand = set(map(int,layout[a['post.topk'][row]].reshape(-1)))
                tokens = list(map(int,records[records[:,0]==row,2]))
                counts = Counter(tokens)
                # Exclusive classes; ABC has precedence over reserved hot.
                abc = sum(t in tail for t in tokens)
                rh = sum(t in reserved and t not in tail for t in tokens)
                oh = sum(t in hot and t not in reserved and t not in tail for t in tokens)
                cold = sum(t not in local for t in tokens)
                rows.append([ref.source_round,ref.dp_rank,row,len(demand),len(demand-local),len(demand&reserved),
                    len(tokens),len(counts),len(tokens)-len(counts),rh,oh,abc,cold,sum(t not in demand for t in tokens)])
        headers=['Round','DP','Request','Demand','Cold demand','Candidate hot demand','Posts','Unique','Repeated posts','Reserved-hot posts','Other-hot posts','ABC posts','Cold posts','Unused posts']
        lines.append(diagnostic_table('Original captured transfers',headers,rows))
        by_round = {}
        for row in rows:
            by_round.setdefault(row[0],[]).append(row)
        for number, selected in by_round.items():
            sums=[sum(r[i] for r in selected) for i in range(3,len(headers))]
            lines.append(diagnostic_table(f'Original capture round={number}: batch total',headers[3:],[sums]))
    with output.open('a') as stream:
        stream.write('\n\n'.join(lines)+'\n')
    print('\n\n'.join(lines));print(f'Saved: {output}')


if __name__ == '__main__':
    main()
