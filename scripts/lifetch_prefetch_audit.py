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
    for capture in data.get('captures', []):
        if capture.get('path') == 'serial':
            continue
        samples = [s for s in capture['samples'] if s.get('algorithm',{}).get('requests')]
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
    lines.append('Current saved counters do not contain per-transfer token lists. They cannot prove absence of repeated hardware reads or exact hot-refetch attribution.')
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
