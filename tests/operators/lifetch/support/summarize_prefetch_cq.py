"""Read existing core timeline captures; never launch device work."""
import argparse
import json
from pathlib import Path


def analyze(sample):
    data = sample['streaming_cycles']
    raw = data[0]['core_timeline_raw']
    mhz = raw['clock_mhz']
    if not mhz or mhz <= 0: raise ValueError('missing calibrated clock')
    origin = min(c[0] for c in raw['cores'])
    rows = []
    for b, f in enumerate(raw['fetch']):
        start, end, polls, drain, observed = (int(f[i]) for i in (96, 100, 101, 102, 103))
        if not start or end < start or drain > end-start or observed > end:
            raise ValueError(f'inconsistent prefetch timestamps b{b}')
        rows.append(dict(batch=b, start_us=(start-origin)/mhz, end_us=(end-origin)/mhz,
                         total_us=(end-start)/mhz, drain_us=drain/mhz,
                         surrounding_us=(end-start-drain)/mhz, polls=polls,
                         drain_begin_us=(observed-drain-origin)/mhz,
                         last_observed_us=(observed-origin)/mhz,
                         us_per_probe=(drain/mhz/polls if polls else None)))
    workers, batch = raw['workers_per_query'], raw['batch']
    fetch_rows = []
    for q in range(2):
        for lane in range(workers):
            block = 2*batch+q*workers+lane
            c = raw['cores'][block]
            f = raw['fetch'][lane][q*24:q*24+24]
            fetch_rows.append(dict(block=block, query=q+1, lane=lane,
                prepare_begin_us=(c[2]-origin)/mhz, prepare_end_us=(c[3]-origin)/mhz,
                get_begin_us=(c[5]-origin)/mhz, get_end_us=(c[6]-origin)/mhz,
                get_us=(c[6]-c[5])/mhz, post_us=f[12]/mhz, cq_us=f[13]/mhz,
                core_end_us=(c[1]-origin)/mhz))
    return dict(round=sample['round'], prefetch=rows, final_fetch=fetch_rows,
                prefetch_scope=data[0].get('prefetch_scope'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    a = p.parse_args()
    files = list(a.root.rglob('full_samples.json'))
    if not files: raise SystemExit('FAIL: no full_samples.json; use the existing timeline result root')
    lines = ['OFFLINE: existing capture only; no device execution.',
             'Drain includes CQ probing/owner checks and consumer commit; surrounding includes state loading/status/cache operations.',
             'Probe time is an average of Drain including control overhead, NOT measured single-load latency.',
             'A failed probe does not distinguish network delay from delayed visibility; pending CQE count was not captured.']
    results = []
    for path in files:
        for sample in json.loads(path.read_text()):
            r = analyze(sample)
            r['source'] = str(path)
            results.append(r)
            lines += [f"ROUND {r['round']} source={path}",
                      'b | start_us | end_us | total_us | drain_us | surrounding_us | probes | drain_us/probe']
            for x in r['prefetch']:
                per = '--' if x['us_per_probe'] is None else f"{x['us_per_probe']:.3f}"
                lines.append(f"{x['batch']} | {x['start_us']:.3f} | {x['end_us']:.3f} | {x['total_us']:.3f} | {x['drain_us']:.3f} | {x['surrounding_us']:.3f} | {x['polls']} | {per}")
            lines.append('Fetch block | Q/lane | prepare_start/end | Get_start/end | Get_us | post_us | CQ_us | core_end_us')
            for x in r['final_fetch']:
                lines.append(f"{x['block']} | Q{x['query']}/{x['lane']} | {x['prepare_begin_us']:.3f}/{x['prepare_end_us']:.3f} | {x['get_begin_us']:.3f}/{x['get_end_us']:.3f} | {x['get_us']:.3f} | {x['post_us']:.3f} | {x['cq_us']:.3f} | {x['core_end_us']:.3f}")
            lines.append('prefetch_scope='+json.dumps(r['prefetch_scope']))
    (a.root/'prefetch_cq_audit.json').write_text(json.dumps(results, indent=2))
    text='\n'.join(lines)+'\n'
    (a.root/'prefetch_cq_audit.log').write_text(text)
    print(text, end='')
    print(f"Summary: {a.root/'prefetch_cq_audit.log'}")


if __name__ == '__main__':
    main()
