"""Bundle real per-Q reference IDs for one-/two-row handoff timing and pairwise checking."""
import argparse
import json
from pathlib import Path

import numpy as np


def pack(items):
    row = np.zeros(528, dtype='<u4')
    row[0] = len(items) // 4
    for i, item in enumerate(items):
        row[8+i] = len(item)
        row[16+i*64:16+i*64+len(item)] = item
    return row


def group(group, audit, out):
    raw = np.fromfile(audit / f'g{group}.want', dtype='<u4').reshape(-1, 72)
    meta = json.loads((audit / f'g{group}.json').read_text())
    if len(raw) != len(meta) or len(raw) != 256:
        raise ValueError('Handoff requires full fixtures from fixtures.py --all-model')
    answer = {(m['window'],m['row'],m['q']): row[8:8+row[0]] for row,m in zip(raw,meta)}
    rows, keys = [], []
    for window in range(4):
        candidates = []
        for r in range(16):
            qs = [answer[window,r,q] for q in range(4)]
            candidates.append((sum(map(len,qs)),r,qs))
        _, r, qs = sorted(candidates)[8]
        _, companion, more = candidates[(r+8)%16]
        for records, rows_used in ((qs,[r]),(qs+more,[r,companion])):
            rows.append(pack(records));keys.append(dict(group=group,window=window,rows=rows_used))
    np.stack(rows).tofile(out/f'g{group}.bin')
    (out/f'g{group}.json').write_text(json.dumps(keys,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('all_lists',type=Path);parser.add_argument('out',type=Path)
    cfg = parser.parse_args()
    cfg.out.mkdir(parents=True,exist_ok=True)
    for g in (10,4,17,7,2):
        group(g,cfg.all_lists,cfg.out)
    edges = [pack([np.arange(n,dtype='<u4')+q*64 for q,n in enumerate(ns)])
             for ns in ((0,0,0,0),(64,64,64,64),(0,8,16,0,3,1,0,5),(64,0,1,31,63,8,0,7))]
    np.stack(edges).tofile(cfg.out/'edges.bin')
