#!/usr/bin/env python3
"""Exhaustive source scheduling + explicitly conditional cost sensitivity.

Counts execute actual host dispatch and allocator. Conditional microseconds
reconstruct the user's approximate W8 kernel table; they are not predictions,
ISA simulation or A5 timings. Full Model results must be reported separately.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import runpy
import tempfile

from simulate_schedule import SOURCE,execute
from verify_cpu import compile_host

REPORT=dict(score=100.1,progress=362.4,service=72.1,prepare=52.3,
            scatter=51.3,topk=41.0,owner=74.2,init_clear=11.2)

def conditional_cost(row,baseline,score_fixed_fraction,prepare_fixed_fraction,predicate_loop_us,inline_epoch_us):
    # One point cannot identify startup vs per-tile compute: vary that split.
    score_per_wave=REPORT['score']/8
    score_fixed=score_per_wave*score_fixed_fraction
    score_tile=score_per_wave*(1-score_fixed_fraction)/5
    prep_launch=(row['hot_score_waves']+row['cold_score_waves']) if row['prepare']==0 else row['prepare']
    prep=REPORT['prepare']*(prepare_fixed_fraction*prep_launch/8+
        (1-prepare_fixed_fraction)*row['prepared_q_weight_qscale_bytes_each_direction']/baseline['prepared_q_weight_qscale_bytes_each_direction'])
    hot_scatter=REPORT['scatter']/8*row['hot_score_waves']
    cold_scatter=REPORT['scatter']/8*row['cold_score_waves'] if row['post']==0 else 0
    ready=6*row['windows'] if row['service']==0 else inline_epoch_us*row['windows']
    # Split publication startup from unchanged total separate predicate work.
    separate_predicate=REPORT['progress']-8*6 if row['post']==0 else 0
    direct_predicate=predicate_loop_us*row['cold_critical_128key_iterations'] if row['post']==2 else 0
    score=score_fixed*(row['hot_score_waves']+row['cold_score_waves'])+score_tile*row['score_critical_128key_iterations']
    return (score+prep+hot_scatter+cold_scatter+ready+separate_predicate+direct_predicate+
        REPORT['service']/8*row['windows']+REPORT['topk']+REPORT['owner']+REPORT['init_clear'])

def frontier(rows,fields):
    # Remove duplicate vectors, then keep non-dominated structural costs.
    unique={tuple(r[f] for f in fields):r for r in rows}
    result=[]
    for vector,row in sorted(unique.items()):
        if any(all(x[f]<=row[f] for f in fields) for x in result):continue
        result=[x for x in result if not all(row[f]<=x[f] for f in fields)]
        result.append(row)
    return result

def run(output):
    output.mkdir(parents=True,exist_ok=False)
    splitter=runpy.run_path(str(SOURCE/'python/kvip/operators/lifetch/schedule.py'))['split_score_tasks']
    rows=[];secondary=[]
    with tempfile.TemporaryDirectory(prefix='kvip-explore-host-') as folder:
        lib,tags=compile_host(SOURCE,Path(folder))
        def count(w,post,prepare,service,ph,pc,c=28,v=56,**kw):
            return execute(SOURCE,lib,tags,w,post,prepare,service,ph,pc,c,v,splitter=splitter,compact=True,**kw)
        for windows in (2,8):
            for ph,pc,prepare,service in itertools.product(range(1,29),range(1,29),range(3),range(2)):
                rows.append(count(windows,2,prepare,service,ph,pc))
            for post,prepare,service in itertools.product(range(3),range(3),range(2)):
                if service and not post:continue
                rows.append(count(windows,post,prepare,service,0,0))
        for c,v in ((2,16),(4,16),(8,32),(16,32),(28,56),(32,64)):
            values=sorted({p for p in (1,2,4,8,16,c) if p<=min(c,v//2)})
            for b,w,f,ph,pc in itertools.product((1,2,4,16,32),(2,8),(1,2,4),values,values):
                if v<b or v<=4*f or (v-4*f)//3<1:continue
                secondary.append(count(w,2,2,1,ph,pc,c,v,batch=b,workers=f))
            for b,w,route in itertools.product((1,2,4,16,32),(2,8),('bsq','auto')):
                if v<b:continue
                secondary.append(count(w,2,2,1,0,0,c,v,batch=b,route=route))
    def write(name,data):
        fields=list(data[0])
        with (output/name).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
            for row in data:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    write('primary_grid.csv',rows);write('secondary_grid.csv',secondary)
    structural=[];sensitivities=[]
    fields=('dispatch_count','prepared_q_weight_qscale_bytes_each_direction',
            'score_critical_128key_iterations','prepare_bytes_before_hot_service')
    base=next(r for r in rows if r['windows']==8 and (r['post'],r['prepare'],r['service'])==(0,0,0))
    reference_total=sum(REPORT.values())
    for windows in (2,8):
        direct=[r for r in rows if r['windows']==windows and r['post']==2 and r['service']==1]
        structural.extend(frontier(direct,fields))
        # Retain the first Hot service ordering for this ranking. Invocation is
        # still in the full structural grid, since it may move Cold copy ahead.
        candidates=[r for r in direct if r['prepare']==2]
        current=next(r for r in candidates if (r['requested_hot_partitions'],r['requested_cold_partitions'])==(4,8))
        for sf,pf,pred,inline in itertools.product((0.,.25,.5,.75,.9),(0.,.5,.9),(0.,.1,.5,1.),(0.,.5,2.)):
            parameters=(sf,pf,pred,inline)
            best=min(candidates,key=lambda r:conditional_cost(r,base,*parameters))
            best_cost=conditional_cost(best,base,*parameters);current_cost=conditional_cost(current,base,*parameters)
            assert abs(conditional_cost(base,base,*parameters)-reference_total)<1e-6
            sensitivities.append(dict(windows=windows,score_fixed_fraction=sf,prepare_fixed_fraction=pf,
                added_direct_predicate_us_per_critical_tile=pred,inline_ready_extra_us_per_epoch=inline,
                current_Ph4_Pc8_conditional_us=current_cost,current_over_approx_staged=current_cost/reference_total,
                best_hot=best['requested_hot_partitions'],best_cold=best['requested_cold_partitions'],
                best_conditional_us=best_cost,best_over_approx_staged=best_cost/reference_total,
                current_over_conditional_best=current_cost/best_cost))
    write('conditional_sensitivity.csv',sensitivities)
    selected=[]
    for windows,ph,pc in itertools.product((2,8),(4,8,16,26,28),(8,16,28)):
        row=next(r for r in rows if (r['windows'],r['requested_hot_partitions'],r['requested_cold_partitions'],r['prepare'],r['service'])==(windows,ph,pc,2,1))
        selected.append(row)
    result=dict(status='SOURCE_COUNTS_PASSED',primary_cases=len(rows),secondary_cases=len(secondary),
        source_scope='actual host dispatch, independent task allocator, exact coverage, calculated payload; no ISA or bandwidth simulation',
        source_sha256={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [SOURCE/'csrc/operators/lightning_indexer_with_fetch/op_host/launch.cpp',SOURCE/'python/kvip/operators/lifetch/schedule.py',Path(__file__).resolve()]},
        structural_frontier_count=len(structural),structural_frontier_fields=fields,
        structural_frontier=structural,selected=selected,conditional_scenarios=len(sensitivities),
        conditional_basis='user-supplied approximate W8 B1 stage table, not raw-trace verified; all coefficient splits are assumptions',
        approximate_report_stage_us=REPORT,approximate_report_total_us=reference_total,
        excludes=['A5 speedup','ISA count','URMA/CQ latency','actual first Fetch','nonzero-prefetch race validation'])
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','primary_cases','secondary_cases','structural_frontier_count','conditional_scenarios')},indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
