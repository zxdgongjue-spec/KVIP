#!/usr/bin/env python3
"""Build or run seven orthogonal B1/Q4 LI controls. No LIFetch/transport calls."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/split_core'))
from score_routes import make_fixture
from prepare_score_source import prepare

# Shared T1/Qparts1 baseline; sweep each axis separately, never a Cartesian grid.
POINTS=((1,1),(2,1),(4,1),(8,1),(16,1),(1,2),(1,4))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(t,qp,history=131072,aics=32,aivs=64):
    if t not in (1,2,4,8,16) or qp not in (1,2,4) or history<2048 or history>210944 or history%256:
        raise ValueError('T=1/2/4/8/16, Qparts=1/2/4, aligned history 2048..210944 required')
    if not 1<=aics<=32 or not 2<=aivs<=64 or history//256<t:
        raise ValueError('valid AIC/AIV budget and nonempty shards required')
    group=4//qp;physical=max(2,group);capacity=min(aics,aivs//2)
    tasks=[dict(query_begin=q,query_count=group,begin=history//256*s//t*256,end=history//256*(s+1)//t*256)
           for q in range(0,4,group) for s in range(t)]
    lengths=[r['end']-r['begin'] for r in tasks]
    waves=[tasks[i:i+capacity] for i in range(0,len(tasks),capacity)]
    logical=4*32*history*128
    return dict(history_parts=t,query_parts=qp,logical_query_group=group,physical_query_tile=physical,
                tasks=len(tasks),capacity=capacity,score_waves=len(waves),row_stride=max(lengths),
                key_tokens_per_task=sorted(set(lengths)),task_map=tasks,
                useful_aivs_peak=max(len(w) for w in waves)*(1 if group==1 else 2),
                allocated_aiv_groups_peak=2*max(len(w) for w in waves),
                logical_qk_macs=logical,cube_macs_with_q_alignment=logical*(2 if group==1 else 1),
                logical_key_read_payload_bytes=qp*history*128,
                score_output_bytes=4*history*2,
                scratch_score_bytes=len(tasks)*physical*max(lengths)*2,
                canonical_alias=t==1 and physical==group and len(waves)==1,
                final_topk_aivs=4,k=2048,fetch_launches=0)


def build(a):
    if a.output.exists():raise ValueError('fresh build output required')
    if not (a.kvaf_include/'kvaf_batch_get_device.h').is_file():raise ValueError('existing compatible KVAF header directory required')
    a.output.mkdir(parents=True)
    before={str(p.relative_to(ROOT)):sha(p) for p in HERE.iterdir() if p.suffix in ('.cpp','.h','.txt','.py')}
    source=prepare(a.output/'source')
    commands=[['cmake','-S',str(HERE),'-B',str(a.output/'build'),'-DRUN_MODE=npu',
               '-DSOC_VERSION='+a.soc,'-DASCEND_CANN_PACKAGE_PATH='+str(a.cann_root),
               '-DSCORE_SOURCE='+str(a.output/'source'),'-DKVAF_INCLUDE='+str(a.kvaf_include)],
              ['cmake','--build',str(a.output/'build'),'-j',str(a.jobs)]]
    receipt=dict(status='BUILDING',commands=commands,wrapper_sha256=before,prepared_source=source)
    (a.output/'build-manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with (a.output/'build.log').open('w') as log:
        for cmd in commands:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                receipt.update(status='BUILD_FAILED',exit_code=result.returncode)
                (a.output/'build-manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
                raise RuntimeError('build failed; see '+str(a.output/'build.log'))
    for path,expected in before.items():
        if sha(ROOT/path)!=expected:raise RuntimeError('wrapper source changed during build: '+path)
    binary=a.output/'build/li_qk'
    receipt.update(status='BUILT',binary_sha256=sha(binary),parts_sha256={str(p.relative_to(a.output)):sha(p)
                   for p in (a.output/'build').rglob('*.a')},note='compiled only; A5 execution not yet accepted')
    (a.output/'build-manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(binary)


def write_summary(output):
    manifest=json.loads((output/'manifest.json').read_text());planned=manifest['cases'];rows=[]
    expected_points={(rnd,t,qp) for rnd in range(manifest['rounds']+1) for t,qp in POINTS}
    actual_points=[(r['round'],r['history_parts'],r['query_parts']) for r in planned]
    if len(actual_points)!=len(expected_points) or set(actual_points)!=expected_points:
        raise ValueError('exactly seven orthogonal points required in every round including r0')
    if len({r['name'] for r in planned})!=len(planned):raise ValueError('duplicate case name')
    receipts={p.stem.removesuffix('.receipt'):p for p in output.glob('*.receipt.json')}
    sdk_cases=manifest.get('sdk_cases',[])
    if sdk_cases and (len(sdk_cases)!=manifest['rounds']+1 or
                      {r['round'] for r in sdk_cases}!=set(range(manifest['rounds']+1))):
        raise ValueError('one SDK reference per round required including r0')
    if set(receipts)!={r['name'] for r in planned+sdk_cases}:raise ValueError('incomplete/extra case receipts')
    baseline={}
    for item in planned:
        receipt=json.loads(receipts[item['name']].read_text())
        if receipt['exit_code'] or receipt['argv']!=item['argv']:raise ValueError('failed or mismatched command')
        if sha(output/(item['name']+'.log'))!=receipt['log_sha256']:raise ValueError('log changed')
        path=output/item['name']/'report.json'
        if sha(path)!=receipt['report_sha256']:raise ValueError('report changed')
        r=json.loads(path.read_text())
        required=('status','run_mode','batch','queries','history','heads','history_parts','query_parts','fetch_launches','warmup','iterations')
        expected=('PASS','npu',1,4,manifest['history'],32,item['history_parts'],item['query_parts'],0,manifest['warmup'],manifest['iterations'])
        if tuple(r[k] for k in required)!=expected or not all(r[k] for k in ('guards','inputs_unchanged','topk_threshold_oracle')):
            raise ValueError('unaccepted scope/correctness: '+str(path))
        for name in ('score','li'):
            v=r[name+'_p50_us'];samples=r[name+'_samples_us']
            if not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 or len(samples)!=manifest['iterations']:
                raise ValueError('invalid device timing')
            if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in samples):raise ValueError('invalid sample')
            if not math.isclose(v,statistics.median(samples),rel_tol=1e-8):raise ValueError('p50 does not match samples')
        if sha(output/item['name']/'score-keys.bin')!=manifest['fixture']['hashes']['oracle']:raise ValueError('canonical oracle SHA mismatch')
        if item['round']==0:continue
        row=dict(round=item['round'],history_parts=item['history_parts'],query_parts=item['query_parts'],
                 score_p50_us=r['score_p50_us'],li_p50_us=r['li_p50_us'],score_waves=r['score_waves'],
                 physical_query_tile=r['physical_query_tile'],report=str(path))
        rows.append(row)
        if (row['history_parts'],row['query_parts'])==(1,1):baseline[row['round']]=row
    if not rows:raise ValueError('no measured rounds')
    for r in rows:
        if r['round'] not in baseline:raise ValueError('missing paired T1/Qparts1 baseline')
        r['score_over_serial']=r['score_p50_us']/baseline[r['round']]['score_p50_us']
        r['li_over_serial']=r['li_p50_us']/baseline[r['round']]['li_p50_us']
    sdk={}
    for item in sdk_cases:
        receipt=json.loads(receipts[item['name']].read_text());path=output/item['name']/'report.json'
        if receipt['exit_code'] or receipt['argv']!=item['argv'] or sha(output/(item['name']+'.log'))!=receipt['log_sha256'] or sha(path)!=receipt['report_sha256']:
            raise ValueError('unaccepted SDK receipt')
        proof=json.loads(path.read_text());samples=proof['sdk_li_samples_us'];v=proof['sdk_li_p50_us']
        if (proof['status'],proof['run_mode'],proof['heads'],proof['history'],proof['fetch_launches'])!=('PASS','npu',32,manifest['history'],0):raise ValueError('SDK scope mismatch')
        if proof['fixture_sha256']!=manifest['fixture']['hashes'] or len(samples)!=manifest['iterations'] or not proof['topk_threshold_oracle']:raise ValueError('SDK fixture/oracle mismatch')
        if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in samples) or not math.isclose(v,statistics.median(samples),rel_tol=1e-8):raise ValueError('invalid SDK timing')
        sdk[item['round']]=v
    summary=[]
    for t,qp in POINTS:
        rr=[r for r in rows if (r['history_parts'],r['query_parts'])==(t,qp)]
        if len(rr)!=manifest['rounds']:raise ValueError('missing measured configuration')
        row=dict(history_parts=t,query_parts=qp,logical_query_group=4//qp,
                 score_p50_mean_us=statistics.mean(r['score_p50_us'] for r in rr),
                 li_p50_mean_us=statistics.mean(r['li_p50_us'] for r in rr),
                 score_over_serial_mean=statistics.mean(r['score_over_serial'] for r in rr),
                 li_over_serial_mean=statistics.mean(r['li_over_serial'] for r in rr),
                 score_over_serial_per_round=[r['score_over_serial'] for r in rr],
                 li_over_serial_per_round=[r['li_over_serial'] for r in rr])
        if sdk_cases:
            if any(r['round'] not in sdk for r in rr):raise ValueError('missing paired SDK reference')
            row.update(sdk_li_p50_mean_us=statistics.mean(sdk[r['round']] for r in rr),
                       li_over_sdk_mean=statistics.mean(r['li_p50_us']/sdk[r['round']] for r in rr))
        summary.append(row)
    for name,data in (('measurements.csv',rows),('summary.csv',summary)):
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(output/'summary.csv')


def run(a):
    if a.output.exists():raise ValueError('fresh experiment output required')
    if 'libruntime_camodel' in os.environ.get('LD_PRELOAD',''):raise ValueError('A5 controls reject Model preload')
    driver=a.build/'build/li_qk';build_path=a.build/'build-manifest.json'
    b=json.loads(build_path.read_text())
    if b['status']!='BUILT' or sha(driver)!=b['binary_sha256']:raise ValueError('unaccepted or changed build')
    for path,expected in b['parts_sha256'].items():
        if sha(a.build/path)!=expected:raise ValueError('changed kernel library')
    caps=json.loads(subprocess.check_output([str(driver),'--capabilities'],text=True))
    if caps['run_mode']!='npu' or caps['fetch_launches']!=0:raise ValueError('physical standalone LI driver required')
    # Lease belongs to this process. No shared-device parallel benchmark launch.
    a.lease.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(a.lease,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        os.write(fd,json.dumps(dict(pid=os.getpid(),device=a.device)).encode());os.close(fd)
        a.output.mkdir(parents=True)
        fixture=make_fixture(a.output/'fixture',1,a.history,32)
        rng=random.Random(1417);cases=[];sdk_cases=[]
        for round_no in range(a.rounds+1):
            if a.sdk_reference:
                name=f'r{round_no}_sdk_li'
                cmd=[sys.executable,str(HERE/'native_li.py'),str(a.output/'fixture'),str(a.output/name),
                     '--device',str(a.device),'--warmup',str(a.warmup),'--iterations',str(a.iterations)]
                sdk_cases.append(dict(name=name,round=round_no,argv=cmd))
            pairs=list(POINTS);rng.shuffle(pairs)
            for t,qp in pairs:
                name=f'r{round_no}_T{t}_Qp{qp}'
                cmd=[str(driver),str(a.output/'fixture'),str(a.output/name),str(a.history),'32',str(a.aics),str(a.aivs),
                     str(t),str(qp),str(a.warmup),str(a.iterations),str(a.device)]
                cases.append(dict(name=name,round=round_no,history_parts=t,query_parts=qp,argv=cmd))
        info=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
        manifest=dict(scope='standalone LI; score and numeric score+canonical+global TopK; no Fetch/Service/Index',
                      saved_lifetch_commit='20adb7e',history=a.history,heads=32,batch=1,queries=4,
                      warmup=a.warmup,iterations=a.iterations,rounds=a.rounds,requested_aics=a.aics,requested_aivs=a.aivs,
                      fixture=fixture,build_manifest_sha256=sha(build_path),build_manifest=b,cases=cases,sdk_cases=sdk_cases,
                      native_reference_source_sha256=sha(HERE/'native_li.py') if a.sdk_reference else None,
                      npu_smi=dict(exit_code=info.returncode,stdout=info.stdout,stderr=info.stderr))
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        ordered=[]
        for round_no in range(a.rounds+1):
            ordered.extend(r for r in sdk_cases if r['round']==round_no)
            ordered.extend(r for r in cases if r['round']==round_no)
        for item in ordered:
            print('Running '+item['name'],flush=True);start=time.time();log=a.output/(item['name']+'.log')
            with log.open('w') as f:
                try:result=subprocess.run(item['argv'],stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    receipt=dict(argv=item['argv'],exit_code=None,status='TIMEOUT',log_sha256=sha(log))
                    (a.output/(item['name']+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
                    raise
            receipt=dict(argv=item['argv'],exit_code=result.returncode,start_unix=start,end_unix=time.time(),log_sha256=sha(log))
            path=a.output/item['name']/'report.json'
            if path.exists():receipt['report_sha256']=sha(path)
            (a.output/(item['name']+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
            if result.returncode:raise RuntimeError('case failed; see '+str(log))
        write_summary(a.output)
    finally:a.lease.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='action',required=True)
    b=subs.add_parser('build');b.add_argument('--output',type=Path,required=True);b.add_argument('--cann-root',type=Path,required=True)
    b.add_argument('--soc',required=True);b.add_argument('--kvaf-include',type=Path,required=True);b.add_argument('--jobs',type=int,default=4)
    for name in ('plan','run'):
        s=subs.add_parser(name);s.add_argument('--history',type=int,default=131072);s.add_argument('--aics',type=int,default=32);s.add_argument('--aivs',type=int,default=64)
        if name=='plan':s.add_argument('--details',action='store_true',help='include every query/history task interval')
        if name=='run':
            s.add_argument('--build',type=Path,required=True);s.add_argument('--output',type=Path,required=True);s.add_argument('--device',type=int,default=5)
            s.add_argument('--warmup',type=int,default=4);s.add_argument('--iterations',type=int,default=20);s.add_argument('--rounds',type=int,default=3)
            s.add_argument('--timeout',type=int,default=300);s.add_argument('--lease',type=Path)
            s.add_argument('--sdk-reference',action='store_true',help='also measure same G32 fixture with SDK standalone LI; no Fetch')
    s=subs.add_parser('summarize');s.add_argument('output',type=Path)
    a=p.parse_args()
    for key in ('output','build','cann_root','kvaf_include'):
        if hasattr(a,key):setattr(a,key,getattr(a,key).resolve())
    if a.action in ('plan','run'):
        rows=[geometry(t,q,a.history,a.aics,a.aivs) for t,q in POINTS]
    if a.action=='plan':
        if not a.details:rows=[{k:v for k,v in r.items() if k!='task_map'} for r in rows]
        print(json.dumps(dict(status='NOT_RUN',batch=1,queries=4,history=a.history,heads=32,cases=rows),indent=2));return
    if a.action=='summarize':write_summary(a.output);return
    if a.action=='build':
        if a.jobs<1:p.error('positive build jobs required')
        build(a);return
    if a.device<0 or a.rounds<1 or not 1<=a.warmup<=100 or not 1<=a.iterations<=1000 or a.timeout<1:p.error('valid sampling and device required')
    a.lease=(a.lease or Path(f'/tmp/kvip-li-qk-device-{a.device}.lease')).resolve()
    run(a)


if __name__=='__main__':main()
