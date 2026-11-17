#!/usr/bin/env python3
"""Attribute Model issues to verified launch phases; counts are not cycle shares."""
import argparse
import bisect
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import re
from instructions import LINE,classify
from map_model_pcs import ENTRY

def classify_tail(pipe,op):
    # RVECLD/RVECST also issue SIMT GM loads/stores. Pipeline names alone
    # do not establish a UB access, and RVECEX includes integer addressing.
    if op.startswith(('SIMT_LDG','SIMT_STG')):return 'global_memory_load_store'
    if op.startswith(('SIMT_LDS','SIMT_STS')):return 'ub_load_store'
    if op.startswith(('SIMT_IADD','SIMT_IMAD','SIMT_IMUL','SIMT_LEA')):return 'integer_arithmetic_addressing'
    if op.startswith(('SIMT_ISETP','SIMT_PLOP3','SIMT_LOP3','SIMT_SHF','SIMT_SEL')):return 'predicate_integer_logic'
    if op.startswith('SIMT_MOV'):return 'vector_register_move'
    return classify(pipe,op)

def analyze_phases(folder):
    manifest=json.loads((folder.parent/'manifest.json').read_text())
    row=next(r for r in manifest['rows'] if r['name']==folder.name)
    if row['status']!='MODEL_PASSED' or row['exit_code']!=0 or int(manifest['logged_core_mask'],16)!=0xffffffff:
        raise ValueError('complete all-core capture required')
    proof=json.loads((folder/'verified-kernel-instructions/pc-map-proof.json').read_text());base=int(proof['base'],16)
    functions={m[2]:(int(m[0],16),int(m[1],16)) for m in re.findall(r'^([0-9a-f]+)\s+\w+\s+F\s+\.text\s+([0-9a-f]+)\s+(\S+)',proof['elf']['symbols'],re.M)}
    starts={base+o:n.rsplit('_',1)[0] for n,(o,s) in functions.items() if n.startswith('li_tail_')}
    categories=Counter();opcodes=Counter();counts=Counter();bounds={};percore=[];entries_proof={}
    for path in sorted(folder.glob('*.instr_log.dump.gz')):
        match=re.fullmatch(r'core(\d+)\.veccore([01])\.instr_log.dump.gz',path.name)
        if not match:continue
        physical=int(match[1])*2+int(match[2]);entries=[]
        with gzip.open(path,'rt',errors='replace') as stream:
            for line in stream:
                e=ENTRY.search(line);m=LINE.search(line)
                if e and m and int(e[1],16) in starts:entries.append((int(m[1]),starts[int(e[1],16)]))
        if not entries:continue
        entries.sort();expected=[x for x in row['report']['launches'] if physical<x['cores']]
        if [k for t,k in entries]!=[x['kernel'] for x in expected]:raise ValueError('launch/core sequence mismatch '+path.name)
        times=[t for t,k in entries];entries_proof[path.name]=[dict(tick=t,**x) for (t,k),x in zip(entries,expected)]
        local=Counter();limits={}
        with gzip.open(path,'rt',errors='replace') as stream:
            for line in stream:
                m=LINE.search(line)
                if not m:continue
                tick,pc,pipe,ident,opcode=m.groups();tick=int(tick);index=bisect.bisect_right(times,tick)-1
                if index<0:raise ValueError('issued instruction before verified entry')
                phase=expected[index]['phase'];counts[phase]+=1;local[phase]+=1
                categories[(phase,pipe,classify_tail(pipe,opcode))]+=1;opcodes[(phase,pipe,opcode)]+=1
                for target in (bounds,limits):
                    span=target.setdefault(phase,[tick,tick]);span[0]=min(span[0],tick);span[1]=max(span[1],tick)
        for phase,n in local.items():percore.append(dict(core=path.name.split('.instr_log')[0],phase=phase,count=n,first=limits[phase][0],last=limits[phase][1],issue_span=limits[phase][1]-limits[phase][0]))
    totals=[dict(phase=k,count=n,share_of_all_issued=n/sum(counts.values()),all_core_issue_envelope=bounds[k][1]-bounds[k][0],max_core_issue_span=max(x['issue_span'] for x in percore if x['phase']==k)) for k,n in counts.items()]
    cats=[dict(phase=p,pipeline=pipe,category=c,count=n,share_of_phase_issued=n/counts[p]) for (p,pipe,c),n in sorted(categories.items())]
    ops=[dict(phase=p,pipeline=pipe,opcode=c,count=n,share_of_phase_issued=n/counts[p]) for (p,pipe,c),n in sorted(opcodes.items())]
    original=json.loads((folder/'verified-kernel-instructions/instructions.json').read_text())
    if sum(counts.values())!=sum(x['instructions'] for x in original['coverage']):raise ValueError('phase denominator differs from complete issue count')
    return dict(scope='all-core dynamic issue counts by verified launch phase; not time percentages',totals=totals,categories=cats,opcodes=ops,cores=percore,entry_proof=entries_proof)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze_phases(a.folder);a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'phases.json').write_text(json.dumps(result,indent=2)+'\n')
    for key in ('totals','categories','opcodes','cores'):
        with (a.output/(key+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[key][0]));w.writeheader();w.writerows(result[key])
    print(a.output)

if __name__=='__main__':main()
