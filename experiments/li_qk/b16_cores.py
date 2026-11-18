#!/usr/bin/env python3
"""B16 score scheduling at 30/32 groups; identical fixture and global TopK."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
from run import geometry,make_fixture,sha
from cases import lease

def configs(fine_grain=128,history=131072):
    if fine_grain not in (128,512):raise ValueError('fine grain must be128 or512')
    rows=[]
    for cores in (30,32):
        for schedule,grain in (('waves',2048),('balanced',2048),('balanced',fine_grain),('balanced_owner',fine_grain)):
            plan=geometry(2,1,history,cores,cores*2,16,schedule,grain)
            work=[sum(r['end']-r['begin'] for r in plan['task_map'][c::cores]) for c in range(cores)]
            rows.append(dict(name=f'C{cores}_T2_{schedule}_G{grain}',batch=16,history_parts=2,query_parts=1,
                aics=cores,aivs=2*cores,schedule=schedule,work_grain=grain,geometry=plan,
                min_score_keys_per_core=min(work),max_score_keys_per_core=max(work)))
    return rows

def summarize(output):
    output=Path(output);m=json.loads((output/'manifest.json').read_text())
    definitions=configs(m['fine_grain'],m['history']);by_name={c['name']:c for c in definitions}
    expected={(r,c['name']) for r in range(m['rounds']+1) for c in definitions}
    actual=[(r['round'],r['config']) for r in m['cases']]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('complete eight cases including r0 required')
    if len({r['name'] for r in m['cases']})!=len(expected):raise ValueError('duplicate execution name')
    if {p.name for p in output.glob('*.receipt.json')}!={r['name']+'.receipt.json' for r in m['cases']}:
        raise ValueError('incomplete/extra receipts')
    rows=[]
    for item in m['cases']:
        c=by_name[item['config']];receipt=json.loads((output/(item['name']+'.receipt.json')).read_text())
        if receipt['exit_code']!=0 or receipt['argv']!=item['argv']:raise ValueError('failed/mismatched case')
        log=output/(item['name']+'.log');folder=output/item['name'];path=folder/'report.json'
        if sha(log)!=receipt['log_sha256'] or sha(path)!=receipt['report_sha256']:raise ValueError('changed log/report')
        r=json.loads(path.read_text());p=c['geometry']
        fields=('status','run_mode','batch','queries','history','heads','k','history_parts','query_parts','schedule','work_grain','phase','model_stage',
                'capacity','score_peak_aics','score_peak_aivs','score_launches','scatter_launches','final_topk_aivs','fetch_launches','warmup','iterations')
        values=('PASS','npu',16,4,m['history'],32,2048,2,1,c['schedule'],c['work_grain'],'both','all',
                c['aics'],c['aics'],c['aivs'],p['score_launches'],p['score_slots'],64,0,m['warmup'],m['iterations'])
        if tuple(r.get(k) for k in fields)!=values:raise ValueError('scope/core/launch mismatch: '+item['name'])
        if not all(r.get(k) for k in ('guards','inputs_unchanged','topk_threshold_oracle')):raise ValueError('oracle rejected')
        if sha(folder/'score-keys.bin')!=m['fixture']['hashes']['oracle']:raise ValueError('canonical score differs')
        for phase in ('score','li'):
            samples=r[phase+'_samples_us'];value=r[phase+'_p50_us']
            if len(samples)!=m['iterations'] or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in samples):raise ValueError('invalid samples')
            if not isinstance(value,(int,float)) or not math.isfinite(value) or not math.isclose(value,statistics.median(samples),rel_tol=1e-8):raise ValueError('invalid p50')
        if item['round']==0:continue
        rows.append(dict(round=item['round'],config=c['name'],score_aics=c['aics'],score_aivs=c['aivs'],schedule=c['schedule'],work_grain=c['work_grain'],
            active_segments=p['active_segments'],score_launches=p['score_launches'],scatter_launches=p['score_slots'],final_topk_aivs=64,
            min_score_keys_per_core=c['min_score_keys_per_core'],max_score_keys_per_core=c['max_score_keys_per_core'],
            score_p50_us=r['score_p50_us'],li_p50_us=r['li_p50_us']))
    lookup={(r['round'],r['config']):r for r in rows}
    for r in rows:
        for phase in ('score','li'):
            for label,ref in (('over_32_waves','C32_T2_waves_G2048'),('over_own_budget_waves',f'C{r["score_aics"]}_T2_waves_G2048'),
                              ('over_matching_32',f'C32_T2_{r["schedule"]}_G{r["work_grain"]}')):
                r[phase+'_'+label]=r[phase+'_p50_us']/lookup[(r['round'],ref)][phase+'_p50_us']
    summary=[]
    for c in definitions:
        selected=[r for r in rows if r['config']==c['name']]
        if len(selected)!=m['rounds']:raise ValueError('missing measured rounds')
        first=selected[0];row={k:first[k] for k in ('config','score_aics','score_aivs','schedule','work_grain','active_segments','score_launches','scatter_launches',
            'final_topk_aivs','min_score_keys_per_core','max_score_keys_per_core')}
        for phase in ('score','li'):
            row[phase+'_p50_mean_us']=statistics.mean(r[phase+'_p50_us'] for r in selected)
            for suffix in ('over_32_waves','over_own_budget_waves','over_matching_32'):
                row[phase+'_'+suffix+'_mean']=statistics.mean(r[phase+'_'+suffix] for r in selected)
                row[phase+'_'+suffix+'_per_round']=[r[phase+'_'+suffix] for r in selected]
        summary.append(row)
    for name,data in (('measurements.csv',rows),('summary.csv',summary)):
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(output/'summary.csv')

def run(a):
    if a.output.exists():raise ValueError('fresh output required')
    if any(x in os.environ.get('LD_PRELOAD','') for x in ('camodel','cmodel')):raise ValueError('physical A5 required')
    build_path=a.build/'build-manifest.json';build=json.loads(build_path.read_text());driver=a.build/'build/li_qk'
    if build['status']!='BUILT' or sha(driver)!=build['binary_sha256']:raise ValueError('changed/unaccepted build')
    for p,h in build['parts_sha256'].items():
        if sha(a.build/p)!=h:raise ValueError('changed device library')
    caps=json.loads(subprocess.check_output([str(driver),'--capabilities'],text=True))
    if caps['run_mode']!='npu' or caps['fetch_launches']!=0:raise ValueError('NPU LI-only binary required')
    with lease('npu',a.device,Path(f'/tmp/kvip-li-qk-device-{a.device}.lease')):
        a.output.mkdir(parents=True)
        definitions=configs(a.fine_grain,a.history);fixture=make_fixture(a.output/'fixture',16,a.history,32)
        rng=random.Random(1417);cases=[]
        for round_no in range(a.rounds+1):
            order=list(definitions);rng.shuffle(order)
            for c in order:
                name=f'r{round_no}_{c["name"]}'
                argv=[str(driver),str(a.output/'fixture'),str(a.output/name),str(a.history),'32',str(c['aics']),str(c['aivs']),'2','1',
                      str(a.warmup),str(a.iterations),str(a.device),'16',c['schedule'],'both',str(c['work_grain'])]
                cases.append(dict(name=name,config=c['name'],round=round_no,argv=argv))
        manifest=dict(scope='B16 numeric LI, mixed 30/32 score budgets; Final64 AIV; no resident/Cold/Service/Fetch/Index',
            fine_grain=a.fine_grain,history=a.history,batch=16,queries=4,heads=32,warmup=a.warmup,iterations=a.iterations,rounds=a.rounds,
            definitions=definitions,cases=cases,fixture=fixture,build_manifest=build,build_manifest_sha256=sha(build_path))
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for item in cases:
            print('Running '+item['name'],flush=True);log=a.output/(item['name']+'.log');started=time.time()
            with log.open('w') as f:
                try:r=subprocess.run(item['argv'],stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout);code=r.returncode
                except subprocess.TimeoutExpired:code=None
            receipt=dict(argv=item['argv'],exit_code=code,start_unix=started,end_unix=time.time(),log_sha256=sha(log))
            report=a.output/item['name']/'report.json'
            if report.exists():receipt['report_sha256']=sha(report)
            (a.output/(item['name']+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
            if code!=0:raise RuntimeError('case failed or timed out: '+str(log))
        if sha(build_path)!=manifest['build_manifest_sha256']:raise ValueError('build receipt changed')
        summarize(a.output)

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    for action in ('plan','run'):
        p=sub.add_parser(action);p.add_argument('--history',type=int,default=131072);p.add_argument('--fine-grain',type=int,choices=(128,512),default=128)
        p.add_argument('--aics',type=int,default=32);p.add_argument('--aivs',type=int,default=64);p.add_argument('--details',action='store_true')
        if action=='run':
            p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=5)
            p.add_argument('--warmup',type=int,default=4);p.add_argument('--iterations',type=int,default=20);p.add_argument('--rounds',type=int,default=3);p.add_argument('--timeout',type=int,default=300)
    p=sub.add_parser('summarize');p.add_argument('output',type=Path)
    a=parser.parse_args()
    if a.action=='summarize':summarize(a.output.resolve());return
    if (a.aics,a.aivs)!=(32,64):parser.error('comparison requires available budget32 AIC/64 AIV; cases request30/60 or32/64')
    definitions=configs(a.fine_grain,a.history)
    if a.action=='plan':
        if not a.details:
            for c in definitions:c['geometry'].pop('task_map')
        print(json.dumps(dict(status='NOT_RUN',batch=16,history=a.history,queries=4,heads=32,fine_grain=a.fine_grain,cases=definitions),indent=2));return
    if a.device<0 or a.rounds<1 or not 1<=a.warmup<=100 or not 1<=a.iterations<=1000 or a.timeout<1:parser.error('valid device and sampling required')
    a.build=a.build.resolve();a.output=a.output.resolve();run(a)

if __name__=='__main__':main()
