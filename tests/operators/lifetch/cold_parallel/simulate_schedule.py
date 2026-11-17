#!/usr/bin/env python3
"""Execute before/after host launch code and count scheduled work, not A5 time/ISA.

The source reference is immutable Git data; the candidate is this checkout.
Device launch functions record arguments. Device math, DMA concurrency, CQ and
bandwidth are not simulated. Run verify_cpu.py for actual pack/publication bodies.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import tempfile

from verify_cpu import SOURCE, RuntimeShape, Plan, ScoreWave, compile_host

FILES=('csrc/operators/common/lifetch/staged_schedule.h',
       'csrc/operators/lightning_indexer_with_fetch/include/runtime_shape.h',
       'csrc/operators/lightning_indexer_with_fetch/op_host/launch.cpp',
       'python/kvip/operators/lifetch/schedule.py')

def snapshot(ref, root):
    for path in FILES:
        content=subprocess.check_output(['git','show',f'{ref}:{path}'],cwd=SOURCE)
        target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    return subprocess.check_output(['git','rev-parse',ref],cwd=SOURCE,text=True).strip()

def execute(source, lib, tags, windows, post, prepare, service, hot_p, cold_p, aics, aivs):
    split=runpy.run_path(str(source/'python/kvip/operators/lifetch/schedule.py'))['split_score_tasks']
    h,r,b=131072,16384,1
    shape=RuntimeShape(batch=b,history=h,resident=r,indexHeads=32,aicCount=aics,rounds=windows,tokenBytes=656)
    assert lib.kvip_prepare_staged_shape(C.byref(shape),aivs,2)==0
    waves=[];tasks_all=[];ranges=[]
    def phase(begin,end,partitions):
        opts=dict(partitions=partitions) if partitions else {}
        qg,_,tasks=split(b,begin,end,aics,aivs,'history',**opts)
        tasks_all.extend(tasks);ranges.extend((t[2],t[3]) for t in tasks)
        assert tasks[0][2]==begin and tasks[-1][3]==end
        assert all(t[3]==z[2] for t,z in zip(tasks,tasks[1:]))
        result=[];cap=min(aics,aivs//2)
        for first in range(0,len(tasks),cap):
            chunk=tasks[first:first+cap];stride=max(t[3]-t[2] for t in chunk)
            result.append(ScoreWave(100,101,102,len(chunk),stride,qg,end if first+len(chunk)==len(tasks) else 0))
        waves.extend(result);return result
    hot=phase(0,r,hot_p);cold=[];begin=r
    for epoch in range(1,windows):
        end=r+((h-r)//128*epoch//(windows-1))*128
        cold.extend(phase(begin,end,cold_p));begin=end
    hot_array=(ScoreWave*len(hot))(*hot);cold_array=(ScoreWave*len(cold))(*cold)
    plan=Plan(300,hot_array,cold_array,len(hot),len(cold),aivs,2,400,len(waves),post,prepare,service)
    lib.reset(0)
    status=lib.kvip_lifetch_staged_forward(1,2,3,C.byref(shape),C.byref(plan))
    assert status==0,status
    names={v:k.removeprefix('aclrtlaunch_v9_') for k,v in tags.items()};names[0]='init'
    sequence=[names[lib.event(i,0)] for i in range(lib.count())]
    assert sequence[-1]=='streaming_owner_wave'
    if service:assert 'staged_progress' not in sequence
    if prepare==2:
        indexes=[i for i,x in enumerate(sequence) if x=='staged_prepare_all']
        first_service=sequence.index('staged_service_ready' if service else 'staged_service')
        assert indexes[0]<first_service<indexes[1]
    # Prepared Q / weight / qscale payload, each direction. Pages are unique H
    # ranges; padding, cache sectors and other operator traffic are excluded.
    prepared_rows=sum(w.rows for w in waves)
    input_bytes=prepared_rows*4*32*(128+2+4)
    cold_keys=4*(h-r);bitmap_bytes=cold_keys//4
    score_payload=cold_keys*2*({0:4,1:3,2:1}[post])
    cold_tasks=len(tasks_all)-sum(w.rows for w in hot)
    return dict(windows=windows,post=post,prepare=prepare,service=service,
        requested_hot_partitions=hot_p,requested_cold_partitions=cold_p,
        dispatch_count=len(sequence),dispatch_sequence=sequence,prepared_task_rows=prepared_rows,
        prepared_q_weight_qscale_bytes_each_direction=input_bytes,
        cold_score_gm_payload_bytes=score_payload,cold_bitmap_payload_bytes=bitmap_bytes,
        cold_predicate_comparisons=cold_keys,
        direct_theta_padded_dma_bytes=cold_tasks*4*256 if post==2 else 0,
        hot_key_lengths=sorted(set(t[3]-t[2] for t in tasks_all[:sum(w.rows for w in hot)])),
        cold_key_lengths=sorted(set(t[3]-t[2] for t in tasks_all[sum(w.rows for w in hot):])),
        resident_topk_h_partitions=1,final_topk_h_partitions=1)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='04ddfaa')
    parser.add_argument('--aics',type=int,default=28);parser.add_argument('--aivs',type=int,default=56)
    parser.add_argument('--output',type=Path,default=SOURCE/'tests/results/score_tuning_simulation.json')
    args=parser.parse_args();rows=[]
    with tempfile.TemporaryDirectory(prefix='kvip-score-schedule-') as folder:
        folder=Path(folder);before=folder/'before';revision=snapshot(args.base,before)
        for label,source in (('before',before),('after',SOURCE)):
            build=folder/label/'build';build.mkdir(parents=True)
            lib,tags=compile_host(source,build)
            cases=[('separate_wave_auto',0,0,0,0,0),('direct_invocation_auto',2,1,0,0,0)]
            if label=='after':cases.extend([
                ('direct_invocation_inline_auto',2,1,1,0,0),
                ('direct_phase_inline_auto',2,2,1,0,0),
                ('direct_phase_inline_Ph4_Pc8',2,2,1,4,8),
                ('direct_invocation_inline_Ph4_Pc8',2,1,1,4,8)])
            for windows in (2,8):
                for name,post,prepare,service,hot,cold in cases:
                    row=execute(source,lib,tags,windows,post,prepare,service,hot,cold,args.aics,args.aivs)
                    rows.append(dict(source=label,case=name,**row))
        for row in rows:
            base=next(x for x in rows if x['source']=='before' and x['windows']==row['windows'] and x['case']=='separate_wave_auto')
            row['dispatch_reduction_vs_before_percent']=100*(1-row['dispatch_count']/base['dispatch_count'])
            row['prepare_payload_reduction_vs_before_percent']=100*(1-row['prepared_q_weight_qscale_bytes_each_direction']/base['prepared_q_weight_qscale_bytes_each_direction'])
            print(f"{row['source']} W{row['windows']} {row['case']}: {row['dispatch_count']} dispatches, {row['prepared_q_weight_qscale_bytes_each_direction']} B prepared input/direction")
    result=dict(status='PASS',measurement_kind='actual host source dispatch + scheduled payload counts',
        excludes=['A5 ISA','device latency','MAC/reduction performance','Service/Fetch/CQ timing','full GM traffic'],
        base_revision=revision,candidate_base_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        candidate_source_state='working tree; exact source bytes recorded below',
        candidate_source_sha256={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in FILES},
        aics=args.aics,aivs=args.aivs,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(args.output)

if __name__=='__main__':main()
