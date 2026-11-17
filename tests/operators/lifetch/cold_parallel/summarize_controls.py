#!/usr/bin/env python3
"""Summarize measured full-chain results; r0 is always excluded."""
import argparse, csv, hashlib, json, math, statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();rows=[]
manifest=json.loads((a.output/'manifest.json').read_text())
planned=manifest['commands']
if not planned or len({x['name'] for x in planned})!=len(planned):raise SystemExit('Empty or duplicate experiment plan')
receipts={x.name.removesuffix('.receipt.json'):x for x in a.output.glob('*.receipt.json')}
if set(receipts)!={x['name'] for x in planned}:raise SystemExit('Receipt set differs from planned runs: incomplete or extra experiment')
for expected in planned:
    receipt=receipts[expected['name']]
    r=json.loads(receipt.read_text())
    if any(r[k]!=expected[k] for k in ('name','argv','discard')):raise SystemExit('Receipt differs from planned run: '+expected['name'])
    if r['exit_code']:raise SystemExit('Failed run: '+r['name'])
    log=a.output/(r['name']+'.log')
    if not log.is_file() or hashlib.sha256(log.read_bytes()).hexdigest()!=r['log_sha256']:raise SystemExit('Missing or changed run log: '+r['name'])
    reports=[(x,json.loads(x.read_text())) for x in (a.output/r['name']).rglob('report.json')]
    reports=[(x,d) for x,d in reports if 'prototype_with_update' in d.get('timing',{})]
    if len(reports)!=1:raise SystemExit('Expected one B1 report: '+r['name'])
    path,d=reports[0]
    if d.get('correctness')!='PASS':raise SystemExit('Correctness not PASS: '+str(path))
    if r['discard']:continue
    cmd=r['argv'];option=lambda s,default=None:cmd[cmd.index(s)+1] if s in cmd else default
    timings=[d['timing'][k]['p50_us'] for k in ('prototype_with_update','native_chain')]
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in timings):
        raise SystemExit('Invalid measured latency: '+str(path))
    geometry=d.get('score_geometry') or {};resources=d.get('resources') or {}
    rows.append(dict(round=r['name'].split('_')[0],windows=option('--intermediate-fetches'),prefetch=option('--prefetch-tokens'),
        route=option('--score-route'),post=option('--score-post'),prepare=option('--score-prepare'),
        service=option('--score-service','separate'),hot_partitions=option('--score-hot-partitions','0'),
        cold_partitions=option('--score-cold-partitions','0'),fetch_workers=option('--q-workers','2'),
        aic_budget=option('--aic-budget','0'),aiv_budget=option('--aiv-budget','auto'),
        actual_aic=geometry.get('effective_aic',resources.get('effective_aic')),
        actual_aiv=geometry.get('aiv_budget') or (int(option('--aiv-budget')) if option('--aiv-budget') else resources.get('aiv')),
        binary_sha256=d.get('binary_sha256'),
        prototype_p50_us=d['timing']['prototype_with_update']['p50_us'],native_p50_us=d['timing']['native_chain']['p50_us'],
        score_geometry=json.dumps(d.get('score_geometry'),sort_keys=True),report=str(path)))
if not rows:raise SystemExit('No completed measured rounds')
with (a.output/'full_chain.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
groups={}
fields=('windows','prefetch','route','post','prepare','service','hot_partitions','cold_partitions','fetch_workers','aic_budget','aiv_budget')
for r in rows:groups.setdefault(tuple(r[k] for k in fields),[]).append(r)
summary=[]
for key,rr in sorted(groups.items()):
    ratios=[];staged_ratios=[]
    for r in rr:
        pairing=('round','windows','prefetch','fetch_workers','aic_budget','aiv_budget')
        baseline=[b for b in rows if all(b[k]==r[k] for k in pairing) and b['route']=='legacy']
        if len(baseline)!=1:raise SystemExit('Missing paired legacy: '+str(key))
        ratios.append(r['prototype_p50_us']/baseline[0]['prototype_p50_us'])
        previous=[b for b in rows if all(b[k]==r[k] for k in pairing) and
                  (b['route'],b['post'],b['prepare'],b['service'],b['hot_partitions'],b['cold_partitions'])==
                  ('history','separate','wave','separate','0','0')]
        if len(previous)!=1:raise SystemExit('Missing paired staged reference: '+str(key))
        for other in (baseline[0],previous[0]):
            for resource in ('actual_aic','actual_aiv','binary_sha256'):
                if r[resource] is not None and other[resource] is not None and r[resource]!=other[resource]:
                    raise SystemExit('Paired hardware/binary mismatch: '+r['report'])
        staged_ratios.append(r['prototype_p50_us']/previous[0]['prototype_p50_us'])
    item=dict(zip(fields,key))
    item.update(
        rounds=len(rr),prototype_p50_mean_us=statistics.mean(r['prototype_p50_us'] for r in rr),
        native_p50_mean_us=statistics.mean(r['native_p50_us'] for r in rr),candidate_over_legacy_mean=statistics.mean(ratios),
        per_round_candidate_over_legacy=ratios,
        candidate_over_staged_before_mean=statistics.mean(staged_ratios),
        per_round_candidate_over_staged_before=staged_ratios,
        prototype_p50_stdev_us=statistics.stdev(r['prototype_p50_us'] for r in rr) if len(rr)>1 else 0,
        latency_reduction_percent=100*(1-statistics.mean(ratios)),speedup_over_legacy=1/statistics.mean(ratios))
    summary.append(item)
    print(f'W{key[0]} pf{key[1]} F{key[8]} {key[2]}/{key[3]}/{key[4]}/{key[5]} Ph{key[6]}/Pc{key[7]}: {item["prototype_p50_mean_us"]:.2f} us, candidate/legacy={item["candidate_over_legacy_mean"]:.3f}')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
best={}
for item in summary:
    if item['route']=='legacy':continue
    key=tuple(item[k] for k in ('windows','prefetch','fetch_workers','aic_budget','aiv_budget'))
    if key not in best or item['prototype_p50_mean_us']<best[key]['prototype_p50_mean_us']:best[key]=item
(a.output/'best_candidates.json').write_text(json.dumps(list(best.values()),indent=2))
