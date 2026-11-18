#!/usr/bin/env python3
"""Build or run orthogonal B1/Q4 and balanced B16 LI controls. No LIFetch/transport calls."""
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
from score_routes import make_fixture as make_small_fixture, score_oracle
from prepare_score_source import prepare

# Shared T1/Qparts1 baseline; sweep each axis separately, never a Cartesian grid.
POINTS=((1,1),(2,1),(4,1),(8,1),(16,1),(1,2),(1,4))

def suite_cases(suite):
    if suite in ('orthogonal7','orthogonal8'):
        return [dict(batch=1,history_parts=t,query_parts=q,schedule='waves')
                for t,q in POINTS+(((32,1),) if suite=='orthogonal8' else ())]
    if suite=='history7':
        return [dict(batch=1,history_parts=t,query_parts=1,schedule='waves')
                for t in (1,2,4,8,16,32,30)]
    if suite=='batch16':
        return [dict(batch=16,history_parts=t,query_parts=1,schedule=s)
                for t,s in ((1,'waves'),(2,'waves'),(2,'balanced'))]
    raise ValueError('unknown suite')



def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def make_fixture(output,batch,history,heads):
    if batch==1:return make_small_fixture(output,batch,history,heads)
    # B16/H128K: avoid an all-owner float32 decoded key and dot temporary.
    # Owner-wise oracle generation is outside all device timing intervals.
    import numpy as np
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(20261115);query=np.zeros((batch,4,heads,128),np.uint8)
    for b in range(batch):
        for q in range(4):
            for g in range(heads):query[b,q,g,(g*7+q*13+b)%128]=0x38 if (b+q+g)%3 else 0xB8
    key=np.memmap(output/'key.bin',mode='w+',dtype=np.uint8,shape=(batch,history,128))
    codes=np.array([0x00,0x20,0x30,0x38,0x40,0xB0,0xB8],np.uint8)
    for b in range(batch):key[b]=rng.choice(codes,(history,128))
    key.flush()
    wf=rng.choice(np.array([0.25,0.5,1,-0.5],np.float32),(batch,4,heads))
    weight=(wf.view(np.uint32)>>16).astype(np.uint16)
    qs=rng.choice(np.array([0.5,1,2],np.float32),weight.shape)
    ks=rng.choice(np.array([0.5,1,2],np.float32),(batch,history))
    oracle=np.memmap(output/'oracle.bin',mode='w+',dtype=np.uint16,shape=(batch,4,history))
    for b in range(batch):oracle[b]=score_oracle(query[b:b+1],key[b:b+1],weight[b:b+1],qs[b:b+1],ks[b:b+1])[0]
    oracle.flush()
    for name,value in dict(query=query,weight=weight,qscale=qs,kscale=ks).items():value.tofile(output/(name+'.bin'))
    meta=dict(batch=batch,history=history,heads=heads,seed=20261115,
              numerical_domain='finite sparse dyadic FP8, BF16 weights, signed per-owner data',
              hashes={name:sha(output/(name+'.bin')) for name in ('query','key','weight','qscale','kscale','oracle')})
    (output/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n');return meta


def geometry(t,qp,history=131072,aics=32,aivs=64,batch=1,schedule="waves",work_grain=2048,score_only=False):
    minimum=256 if score_only and schedule in ('balanced','balanced_owner') else 2048
    if t not in (1,2,4,8,16,32,30) or qp not in (1,2,4) or history<minimum or history>210944 or history%256:
        raise ValueError('T=1/2/4/8/16/32/30, Qparts=1/2/4, aligned history 2048..210944 required')
    if not 1<=aics<=32 or not 2<=aivs<=64 or (schedule=='waves' and history//256<t):
        raise ValueError('valid AIC/AIV budget and nonempty shards required')
    if batch not in (1,16) or schedule not in ('waves','balanced','balanced_owner') or (batch==16 and (t>2 or qp!=1)):
        raise ValueError('B1 axes or B16 T1/T2 Q4 required')
    group=4//qp;physical=max(2,group);capacity=min(aics,aivs//2)
    if schedule=='waves' and work_grain!=2048:raise ValueError('work grain applies to balanced score only')
    quantum=128 if t==30 else 256
    tasks=[dict(query_begin=q,query_count=group,begin=history//quantum*s//t*quantum,end=history//quantum*(s+1)//t*quantum)
           for b in range(batch) for q in range(0,4,group) for s in range(t)]
    if batch>1:
        for i,task in enumerate(tasks):task["owner"]=i//(qp*t)
    logical_tasks=len(tasks)
    core_blocks=[]
    if schedule in ('balanced','balanced_owner'):
        if ((batch,t,qp)!=(16,2,1) or capacity not in (30,32) or not 128<=work_grain<=65536 or
                work_grain&(work_grain-1) or history%(2*work_grain)):
            raise ValueError("balanced requires B16/T2/Q4,30 or32 groups,power-of-two grain128..65536,H multiple of2*grain")
        domains=1 if schedule=='balanced_owner' else 2
        total=batch*history//work_grain;domain=history//(domains*work_grain);cores=[]
        for c in range(capacity):
            lo=total*c//capacity;hi=total*(c+1)//capacity;core_blocks.append(hi-lo);pieces=[]
            while lo<hi:
                logical=lo//domain;end=min(hi,(logical+1)*domain);begin=(logical%domains*domain+lo%domain)*work_grain
                pieces.append(dict(query_begin=0,query_count=4,begin=begin,end=begin+(end-lo)*work_grain,owner=logical//domains));lo=end
            cores.append(pieces)
        tasks=[cores[c][slot] if slot<len(cores[c]) else dict(query_begin=0,query_count=0,begin=0,end=0,owner=0)
               for slot in range(max(map(len,cores))) for c in range(capacity)]
    lengths=[r['end']-r['begin'] for r in tasks]
    waves=[tasks[i:i+capacity] for i in range(0,len(tasks),capacity)]
    logical=batch*4*32*history*128
    return dict(batch=batch,schedule=schedule,work_grain=work_grain,logical_tasks=logical_tasks,active_segments=sum(x>0 for x in lengths),core_blocks=core_blocks,history_parts=t,query_parts=qp,logical_query_group=group,physical_query_tile=physical,
                tasks=len(tasks),capacity=capacity,score_waves=1 if schedule!='waves' else len(waves),score_slots=len(waves),score_launches=1 if schedule!='waves' else len(waves),row_stride=max(lengths),
                key_tokens_per_task=sorted(set(lengths)),task_map=tasks,
                useful_aivs_peak=max(len(w) for w in waves)*(1 if group==1 else 2),
                allocated_aiv_groups_peak=2*max(len(w) for w in waves),
                logical_qk_macs=logical,cube_macs_with_q_alignment=logical*(2 if group==1 else 1),
                logical_key_read_payload_bytes=batch*qp*history*128,
                score_output_bytes=batch*4*history*2,
                scratch_score_bytes=len(tasks)*physical*max(lengths)*2,
                canonical_alias=t==1 and physical==group and len(waves)==1,
                history_partition_quantum=work_grain if schedule!='waves' else quantum,
                score_peak_aics=capacity if schedule!='waves' else min(capacity,len(tasks)),
                score_peak_aivs=2*(capacity if schedule!='waves' else min(capacity,len(tasks))),
                final_topk_aivs=batch*4,k=2048,fetch_launches=0)


def build(a):
    if a.output.exists():raise ValueError('fresh build output required')
    if not (a.kvaf_include/'kvaf_batch_get_device.h').is_file():raise ValueError('existing compatible KVAF header directory required')
    a.output.mkdir(parents=True)
    before={str(p.relative_to(ROOT)):sha(p) for p in HERE.iterdir() if p.suffix in ('.cpp','.h','.txt','.py')}
    source=prepare(a.output/'source')
    commands=[['cmake','-S',str(HERE),'-B',str(a.output/'build'),'-DRUN_MODE='+a.run_mode,
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
    configs=suite_cases(manifest.get('suite','orthogonal7'))
    key=lambda r:(r['history_parts'],r['query_parts'],r.get('schedule','waves'))
    expected_points={(rnd,*key(c)) for rnd in range(manifest['rounds']+1) for c in configs}
    actual_points=[(r['round'],*key(r)) for r in planned]
    if len(actual_points)!=len(expected_points) or set(actual_points)!=expected_points:
        raise ValueError('exactly the selected suite points required in every round including r0 (seven orthogonal by default)')
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
        expected=('PASS','npu',manifest.get('batch',1),4,manifest['history'],32,item['history_parts'],item['query_parts'],0,manifest['warmup'],manifest['iterations'])
        if tuple(r[k] for k in required)!=expected or not all(r[k] for k in ('guards','inputs_unchanged','topk_threshold_oracle')):
            raise ValueError('unaccepted scope/correctness: '+str(path))
        if r.get('schedule','waves')!=item.get('schedule','waves') or r.get('phase','both')!='both' or r.get('model_stage','all')!='all':
            raise ValueError('unaccepted schedule/isolated profiling report')
        if r.get('work_grain',2048)!=(manifest.get('work_grain',2048) if item.get('schedule')=='balanced' else 2048):
            raise ValueError('unaccepted work grain')
        if manifest.get('suite')=='history7':
            expected_plan=geometry(item['history_parts'],1,manifest['history'],32,64)
            for field in ('score_peak_aics','score_peak_aivs','history_partition_quantum','final_topk_aivs'):
                if r.get(field)!=expected_plan[field]:raise ValueError('history7 core allocation mismatch: '+field)
            if r['score_waves']!=1 or r.get('score_launches')!=1 or r.get('tasks')!=item['history_parts']:
                raise ValueError('history7 requires one score launch covering every shard')
        for name in ('score','li'):
            v=r[name+'_p50_us'];samples=r[name+'_samples_us']
            if not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 or len(samples)!=manifest['iterations']:
                raise ValueError('invalid device timing')
            if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in samples):raise ValueError('invalid sample')
            if not math.isclose(v,statistics.median(samples),rel_tol=1e-8):raise ValueError('p50 does not match samples')
        if sha(output/item['name']/'score-keys.bin')!=manifest['fixture']['hashes']['oracle']:raise ValueError('canonical oracle SHA mismatch')
        if item['round']==0:continue
        row=dict(round=item['round'],batch=manifest.get('batch',1),schedule=item.get('schedule','waves'),work_grain=r.get('work_grain',2048),history_parts=item['history_parts'],query_parts=item['query_parts'],
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
    for config in configs:
        t,qp=config['history_parts'],config['query_parts']
        rr=[r for r in rows if key(r)==key(config)]
        if len(rr)!=manifest['rounds']:raise ValueError('missing measured configuration')
        row=dict(batch=manifest.get('batch',1),schedule=config['schedule'],work_grain=rr[0]['work_grain'],history_parts=t,query_parts=qp,logical_query_group=4//qp,
                 reference='B16 T1: one group per owner' if manifest.get('batch',1)==16 else 'B1 T1: one group total',
                 score_p50_mean_us=statistics.mean(r['score_p50_us'] for r in rr),
                 li_p50_mean_us=statistics.mean(r['li_p50_us'] for r in rr),
                 score_over_serial_mean=statistics.mean(r['score_over_serial'] for r in rr),
                 li_over_serial_mean=statistics.mean(r['li_over_serial'] for r in rr),
                 score_over_serial_per_round=[r['score_over_serial'] for r in rr],
                 li_over_serial_per_round=[r['li_over_serial'] for r in rr])
        if manifest.get('suite')=='history7':
            planned=geometry(t,1,manifest['history'],32,64)
            row.update(score_peak_aics=planned['score_peak_aics'],score_peak_aivs=planned['score_peak_aivs'],
                       history_partition_quantum=planned['history_partition_quantum'],
                       min_history_per_core=min(planned['key_tokens_per_task']),max_history_per_core=max(planned['key_tokens_per_task']),
                       score_launches=1,final_topk_aivs=4)
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
    if any(x in os.environ.get('LD_PRELOAD','') for x in ('libruntime_camodel','libruntime_cmodel')):raise ValueError('A5 controls reject Model preload')
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
        configs=suite_cases(a.suite);batch=configs[0]['batch']
        fixture=make_fixture(a.output/'fixture',batch,a.history,32)
        rng=random.Random(1417);cases=[];sdk_cases=[]
        for round_no in range(a.rounds+1):
            if a.sdk_reference:
                name=f'r{round_no}_sdk_li'
                cmd=[sys.executable,str(HERE/'native_li.py'),str(a.output/'fixture'),str(a.output/name),
                     '--device',str(a.device),'--warmup',str(a.warmup),'--iterations',str(a.iterations)]
                sdk_cases.append(dict(name=name,round=round_no,argv=cmd))
            pairs=list(configs);rng.shuffle(pairs)
            for config in pairs:
                t,qp=config['history_parts'],config['query_parts']
                name=f'r{round_no}_T{t}_Qp{qp}'+('_balanced' if config['schedule']=='balanced' else '')
                cmd=[str(driver),str(a.output/'fixture'),str(a.output/name),str(a.history),'32',str(a.aics),str(a.aivs),
                     str(t),str(qp),str(a.warmup),str(a.iterations),str(a.device),str(batch),config['schedule'],'both',
                     str(a.work_grain if config['schedule']=='balanced' else 2048)]
                cases.append(dict(name=name,round=round_no,history_parts=t,query_parts=qp,schedule=config['schedule'],argv=cmd))
        info=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
        manifest=dict(scope='standalone LI; score and numeric score+canonical+global TopK; no Fetch/Service/Index',
                      saved_lifetch_commit='20adb7e',suite=a.suite,work_grain=a.work_grain,history=a.history,heads=32,batch=batch,queries=4,
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
    b.add_argument('--soc',required=True);b.add_argument('--kvaf-include',type=Path,required=True);b.add_argument('--jobs',type=int,default=4);b.add_argument('--run-mode',choices=('npu','sim'),default='npu')
    for name in ('plan','run'):
        s=subs.add_parser(name);s.add_argument('--history',type=int,default=131072);s.add_argument('--aics',type=int,default=32);s.add_argument('--aivs',type=int,default=64)
        s.add_argument('--suite',choices=('orthogonal7','orthogonal8','history7','batch16'),default='orthogonal7')
        s.add_argument('--work-grain',type=int,default=2048,help='balanced work allocation quantum; hardware score tile remains128')
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
        configs=suite_cases(a.suite)
        if a.suite!='batch16' and a.work_grain!=2048:p.error('work grain applies to batch16 balanced only')
        rows=[geometry(c['history_parts'],c['query_parts'],a.history,a.aics,a.aivs,c['batch'],c['schedule'],a.work_grain if c['schedule']=='balanced' else 2048) for c in configs]
        if a.suite in ('orthogonal8','history7') and min(a.aics,a.aivs//2)<32:p.error('K32 suite requires32 AIC/64 AIV; use orthogonal7 for smaller budgets')
        if a.suite=='batch16' and min(a.aics,a.aivs//2)!=30:p.error('batch16 suite requires --aics30 --aivs60 (or64)')
    if a.action=='plan':
        if not a.details:rows=[{k:v for k,v in r.items() if k!='task_map'} for r in rows]
        print(json.dumps(dict(status='NOT_RUN',suite=a.suite,batch=configs[0]['batch'],queries=4,history=a.history,heads=32,cases=rows),indent=2));return
    if a.action=='summarize':write_summary(a.output);return
    if a.action=='build':
        if a.jobs<1:p.error('positive build jobs required')
        build(a);return
    if a.device<0 or a.rounds<1 or not 1<=a.warmup<=100 or not 1<=a.iterations<=1000 or a.timeout<1:p.error('valid sampling and device required')
    if a.sdk_reference and a.suite=='batch16':p.error('SDK fixture adapter supports B1 only')
    a.lease=(a.lease or Path(f'/tmp/kvip-li-qk-device-{a.device}.lease')).resolve()
    run(a)


if __name__=='__main__':main()
