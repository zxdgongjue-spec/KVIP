#!/usr/bin/env python3
"""Count dynamic Model instruction issue logs, never turn counts into A5 time."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import re

LINE=re.compile(r'\[(\d+)\]\s+\(PC: (0x[0-9a-fA-F]+)\)\s+(\w+)\s*:\s*\(Binary:.*?\)\s*\(ID:\s*(\d+)\)\s+(\w+)')

def classify(pipe,op):
    if 'FLAG' in op or 'BAR' in op or 'INTRA_BLOCK' in op or op.startswith(('WAIT','SYNC')):return 'sync_control'
    if pipe in ('CUBE','MATRIX') or op.startswith(('MMAD','MAD_')):return 'cube_math'
    if pipe.startswith('MTE') or op.startswith(('COPY','LOAD_2D','LOAD_3D')):return 'data_movement'
    if pipe in ('FIXP','FIXPIPE','FIX'):return 'fixpipe'
    if pipe=='PUSHQ':return 'vector_dispatch_mask_control'
    if pipe in ('VECTOR','VEC') or op.startswith(('VF','VADD','VMUL','VMAX','VREDUCE','VCAST')):return 'vector_math'
    if pipe in ('RVECLD','RVECST'):return 'vector_ub_load_store'
    if pipe=='RVECEX':return 'vector_math'
    if pipe in ('RVECSU','RVECLP'):return 'vector_loop_scalar_control'
    if pipe=='FLOWCTRL':return 'scalar_flow_control'
    if pipe=='SCALAR':return 'scalar_control_address'
    return 'unknown'

def analyze(folder,pc_map=None):
    folder=Path(folder);files=sorted(folder.glob('*.instr_log.dump'))+sorted(folder.glob('*.instr_log.dump.gz'))
    if not files:raise ValueError('no dynamic instruction issue logs')
    rows=[];ops=[];coverage=[]
    for path in files:
        counters=Counter();opcount=Counter();pc=Counter();kernel_totals=Counter();first=None;last=None;unparsed=0;total=0;resets=0;previous=None;active_kernel=None
        open_file=gzip.open if path.suffix=='.gz' else open
        with open_file(path,'rt',errors='replace') as f:
            for line in f:
                if not line.strip():continue
                m=LINE.search(line)
                if not m:unparsed+=1;continue
                tick,addr,pipe,i,op=m.groups();tick=int(tick);i=int(i)
                matches=[r for r in (pc_map or []) if int(r['begin'],16)<=int(addr,16)<int(r['end'],16)]
                if len(matches)>1:raise ValueError('overlapping PC regions')
                kernel=matches[0]['kernel'] if matches else 'unmapped'
                if matches and matches[0].get('shared_owners'):
                    if active_kernel in matches[0]['shared_owners']:kernel=active_kernel
                elif matches:active_kernel=kernel
                if previous is not None and i<previous:resets+=1
                previous=i;total+=1;first=tick if first is None else min(first,tick);last=tick if last is None else max(last,tick)
                kernel_totals[kernel]+=1;counters[(kernel,pipe,classify(pipe,op))]+=1;opcount[(kernel,pipe,op)]+=1;pc[addr]+=1
        core=path.name.split('.instr_log')[0]
        coverage.append(dict(file=path.name,core=core,instructions=total,unparsed_lines=unparsed,id_order_decreases=resets,
                             first_issue_tick=first,last_issue_tick=last,pc_count=len(pc),kernel_counts=dict(kernel_totals)))
        for (kernel,pipe,category),count in counters.items():rows.append(dict(core=core,kernel=kernel,pipeline=pipe,category=category,count=count,share_of_kernel_core_instructions=count/kernel_totals[kernel],share_of_core_instructions=count/total))
        for (kernel,pipe,op),count in opcount.items():ops.append(dict(core=core,kernel=kernel,pipeline=pipe,opcode=op,count=count,share_of_kernel_core_instructions=count/kernel_totals[kernel],share_of_core_instructions=count/total))
    if not sum(x['instructions'] for x in coverage):raise ValueError('empty/unrecognized instruction logs')
    return dict(scope='dynamic issued instruction counts in logged cores only; not pipeline cycles or algorithm time',
                denominator='all parsed issued instructions per core; AIC/AIV kept separate',coverage=coverage,
                id_note='IDs can issue out of order across pipelines; decreases do not establish kernel boundaries',
                isa_note='AIV denominator includes scalar/PUSHQ issue events and expanded RVEC instruction issue events',
                categories=rows,opcodes=ops,kernel_pc_map=pc_map,
                algorithm_pc_mapping='kernel ranges only; algorithm subfunctions NOT_AVAILABLE')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pc-map',type=Path,help='verified loaded kernel PC ranges JSON; omit to keep unmapped');a=p.parse_args()
    result=analyze(a.folder,json.loads(a.pc_map.read_text()) if a.pc_map else None);a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'instructions.json').write_text(json.dumps(result,indent=2)+'\n')
    for name in ('categories','opcodes','coverage'):
        with (a.output/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(result[name][0]));w.writeheader();w.writerows(result[name])
    print(a.output)

if __name__=='__main__':main()
