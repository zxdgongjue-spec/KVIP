"""Compare two equal-16-AIV schedules on isolated window helper cycles."""
import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean

PASS = re.compile(r'PASS row=(\d+) pass=(\d+) variant=(\d+) count=(\d+) ticks=(\d+)')


def group_results(group, logs, fixtures, writer):
    meta = json.loads((fixtures / f'g{group}.json').read_text())
    measurements = {}
    for match in PASS.finditer((logs/f'model-g{group}.log').read_text()):
        i, run, variant, posted, ticks = map(int, match.groups())
        if not 0 <= i < len(meta) or variant != run or run > 1 or posted != meta[i]['posted'] or ticks <= 0:
            raise ValueError('Expected a production --once run with all outputs checked')
        m = meta[i]
        key = (m['window'],m['row'],m['q'],variant)
        if key in measurements:
            raise ValueError(f'Duplicate record {key}')
        measurements[key] = ticks
    if len(measurements) != 2*4*16*4:
        raise ValueError(f'Incomplete group {group}: {len(measurements)} measurements')
    for m in meta:
        w,b,q = (m[k] for k in ('window','row','q'))
        writer.writerow([group,w,b,q,m['uncovered'],m['posted'],*(measurements[w,b,q,v] for v in range(2))])
    output = []
    for w in range(4):
        for v in range(2):
            times = [[measurements[w,b,q,v] for q in range(4)] for b in range(16)]
            sixteen_rows = max(sum(t) for t in times)
            four_rows = sum(max(max(t) for t in times[start:start+4]) for start in range(0,16,4))
            output.append(dict(window=w, variant=v, one_lane_per_row_ticks=sixteen_rows,
                               four_lanes_per_row_four_batches_ticks=four_rows))
    return dict(group=group, estimated_helper_makespans=output,
                note='Same 16 AIVs and all scores assume ready; excludes scoring, readiness and synchronization.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs',type=Path);parser.add_argument('samples',type=Path)
    parser.add_argument('output_csv',type=Path);parser.add_argument('groups',nargs='*',type=int,default=[10,4,17,7,2])
    cfg=parser.parse_args()
    with cfg.output_csv.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['group','window','row','query','eligible','posted','main_ticks','compact_ticks'])
        for g in cfg.groups:
            print(json.dumps(group_results(g,cfg.logs,cfg.samples,writer)))
