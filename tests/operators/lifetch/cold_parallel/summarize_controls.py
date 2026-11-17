#!/usr/bin/env python3
"""Summarize measured full-chain results; r0 is always excluded."""
import argparse, csv, hashlib, json, statistics
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
    cmd=r['argv'];option=lambda s:cmd[cmd.index(s)+1]
    rows.append(dict(round=r['name'].split('_')[0],windows=option('--intermediate-fetches'),prefetch=option('--prefetch-tokens'),
        route=option('--score-route'),post=option('--score-post'),prepare=option('--score-prepare'),
        prototype_p50_us=d['timing']['prototype_with_update']['p50_us'],native_p50_us=d['timing']['native_chain']['p50_us'],
        score_geometry=json.dumps(d.get('score_geometry'),sort_keys=True),report=str(path)))
if not rows:raise SystemExit('No completed measured rounds')
with (a.output/'full_chain.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
groups={}
for r in rows:groups.setdefault((r['windows'],r['prefetch'],r['route'],r['post'],r['prepare']),[]).append(r)
summary=[]
for key,rr in sorted(groups.items()):
    ratios=[]
    for r in rr:
        baseline=[b for b in rows if (b['round'],b['windows'],b['prefetch'],b['route'])==(r['round'],r['windows'],r['prefetch'],'legacy')]
        if len(baseline)!=1:raise SystemExit('Missing paired legacy: '+str(key))
        ratios.append(r['prototype_p50_us']/baseline[0]['prototype_p50_us'])
    item=dict(windows=key[0],prefetch=key[1],route=key[2],post=key[3],prepare=key[4],
        rounds=len(rr),prototype_p50_mean_us=statistics.mean(r['prototype_p50_us'] for r in rr),
        native_p50_mean_us=statistics.mean(r['native_p50_us'] for r in rr),candidate_over_legacy_mean=statistics.mean(ratios),
        per_round_candidate_over_legacy=ratios)
    summary.append(item)
    print(f'W{key[0]} pf{key[1]} {key[2]}/{key[3]}/{key[4]}: {item["prototype_p50_mean_us"]:.2f} us, candidate/legacy={item["candidate_over_legacy_mean"]:.3f}')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
