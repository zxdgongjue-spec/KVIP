#!/usr/bin/env python3
"""Produce request-level fitting evidence; legacy records are independent snapshots."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from convert_trace import digest


def legacy_rows(arrays, metadata):
    owners = arrays['before.row_owners'].reshape(-1)
    batch = len(owners)
    selected = arrays['after.logical_topk'].reshape(batch, 4, 2048)
    layout = arrays['before.logical_ids']
    valid = arrays.get('before.hot_valid')
    if layout.ndim != 2 or valid is None or valid.ndim != 2:
        raise ValueError('Legacy analysis requires two-dimensional logical_ids/hot_valid')
    history, resident = layout.shape[1], valid.shape[1]
    active = arrays.get('before.query_valid', np.ones((batch, 4))).reshape(batch, 4).astype(bool)
    tails = arrays.get('before.tail_bounds')
    rows = []
    for b, owner_value in enumerate(owners):
        if not active[b].any():
            continue
        owner = int(owner_value)
        if not 0 <= owner < len(layout) or owner >= len(valid):
            raise ValueError('Owner outside captured cache pools')
        mapping = layout[owner]
        hot = set(map(int, mapping[:resident][valid[owner] != 0]))
        local = set(hot)
        if tails is not None:
            lo, hi = map(int, tails.reshape(batch, 2)[b])
            if not 0 <= lo <= hi <= history:
                raise ValueError('Invalid tail bounds')
            local.update(range(lo, hi))
        sets = []
        for q in range(4):
            ids = selected[b, q]
            if active[b, q] and (ids.min() < 0 or ids.max() >= history or len(np.unique(ids)) != 2048):
                raise ValueError('Invalid active logical TopK')
            sets.append(set(map(int, ids)) if active[b, q] else set())
        union = set.union(*sets)
        seen, queries = set(), []
        for q, values in enumerate(sets):
            queries.append(dict(query=q+1, active=bool(active[b,q]), unique=len(values),
                new_unique=len(values-seen), cross_q_duplicates=len(values & seen),
                initial_cold=len(values-local),
                logical_position_hist=np.histogram(list(values), bins=16, range=(0, history))[0].tolist()))
            seen.update(values)
        rows.append(dict(row=b, active_queries=int(active[b].sum()), owner=owner, snapshot_sample=metadata.get('sample'),
            history=history, resident=resident, initial_hot_valid=len(hot),
            selected_unique=len(union), initial_cold=len(union-local), queries=queries,
            q_jaccard=[[len(a & b)/len(a | b) if a | b else None for b in sets] for a in sets],
            temporal=None, prefetch=None, final=None, unused_prefetch=None,
            unavailable=['stable request identity', 'captured scores', 'window transfer records']))
    return rows


def collect(root, support=None, coverage=None):
    root = Path(root)
    index = json.loads((root/'index.json').read_text())
    if index['status'] != 'COMPLETE':
        raise ValueError('Conversion is incomplete')
    rows, previous = [], {}
    if support:
        sys.path.insert(0, str(Path(support)/'tests/operators/lifetch/support'))
    for entry in index['captures']:
        folder = root / entry['bundle']
        m = json.loads((folder/'manifest.json').read_text())
        archive = folder/m['archive']['file']
        if digest(archive) != m['archive']['sha256']:
            raise ValueError('Bundle checksum differs')
        with np.load(archive, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        if m['source_kind'] == 'legacy_operand_v1':
            records = legacy_rows(arrays, m['source_metadata'])
            slots = len(arrays['before.row_owners'].reshape(-1))
        else:
            if not support:
                raise ValueError('Schema-4 statistics need --kvip-root')
            from calibration import capture_rows
            from trace_inputs import ScoreCapture
            source = m['source_metadata']
            metadata = source['metadata']
            records = []
            for i, (record, selected) in enumerate(capture_rows(ScoreCapture(folder, source, arrays))):
                identities = metadata.get('requests', [])
                scheduler = metadata.get('context', {}).get('round')
                record['temporal'] = None
                if i < len(identities) and isinstance(scheduler, int):
                    key = (metadata.get('dp_rank'), metadata.get('group'), identities[i]['request_id'], identities[i]['owner'])
                    old = previous.get((key, scheduler-1))
                    if old is not None:
                        record['temporal'] = dict(retained=len(old & selected),
                            jaccard=len(old & selected)/len(old | selected))
                    previous[(key, scheduler)] = selected
                records.append(record)
            slots = len(records)
        if coverage is not None:
            coverage.append(dict(capture=entry['bundle'], slots=slots, active=len(records),
                                 padding_skipped=slots-len(records)))
        for record in records:
            record['capture'] = entry['bundle']
            record['source_kind'] = m['source_kind']
            rows.append(record)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle_root', type=Path)
    p.add_argument('--kvip-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    coverage = []
    rows = collect(args.bundle_root, args.kvip_root, coverage)
    if not rows:
        raise ValueError("No active request snapshots")
    def stats(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return dict(samples=len(values), quantiles=dict(zip(['p0','p10','p50','p90','p100'],
            np.quantile(values, [0,.1,.5,.9,1]).tolist()))) if values else dict(samples=0)
    summary = dict(kind='lifetch_generator_fit_evidence', fitted_generator=False,
        request_snapshots=len(rows), captured_slots=sum(c["slots"] for c in coverage),
        padding_skipped=sum(c["padding_skipped"] for c in coverage), captures=coverage, metrics={key:stats(key) for key in
        ('selected_unique','initial_cold','initial_hot_valid','prefetch','unused_prefetch','final')},
        temporal_pairs=sum(r.get('temporal') is not None for r in rows))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'request_rows.json').write_text(json.dumps(rows, indent=2))
    (args.output/'fit_targets.json').write_text(json.dumps(summary, indent=2))
    print(f'Analyzed {len(rows)} active request snapshots; captured slots={summary["captured_slots"]}; '
          f'padding skipped={summary["padding_skipped"]}; temporal pairs={summary["temporal_pairs"]}')
    for key, value in summary['metrics'].items():
        print(key, ' '.join(f'{k}={v:.2f}' for k,v in value.get('quantiles', {}).items()) or 'not captured')


if __name__ == '__main__':
    main()
