#!/usr/bin/env python3
"""Post-score controls: exact TopK, history predicates, static mapping classification."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import random
import re
import shutil
import shlex
import statistics
import subprocess
import time
from run import HERE,ROOT,make_fixture,prepare,sha
from cases import lease
from instructions import analyze
from map_model_pcs import elf,ENTRY

CONTROLS={'C0':(0,0),'C1':(1,0),'C2':(0,1),'C3':(1,1)}

def plan(a):
    if not a.controls or len(set(a.controls))!=len(a.controls):
        raise ValueError('unique controls required')
    if a.batch not in (1,16) or a.history%256 or not 4096<=a.history<=131072 or not 2048<=a.resident<=16384 or a.resident%256 or a.resident>=a.history:
        raise ValueError('B1/B16, H4096..131072, R2048..16384 aligned256 required')
    if a.parts not in (1,2,4,8,16) or a.history%a.parts or a.history//a.parts<2048 or a.history//a.parts%256:
        raise ValueError('aligned H part >=2048 required')
    if a.resident_parts not in (1,2,4) or a.resident%a.resident_parts or a.resident//a.resident_parts<2048 or a.resident//a.resident_parts%256:
        raise ValueError('aligned resident part >=2048 required')
    result=[]
    for name in a.controls:
        cp,fs=CONTROLS[name]
        result.append(dict(name=name,batch=a.batch,history=a.history,resident=a.resident,parts=a.parts,resident_parts=a.resident_parts,cold_parallel=cp,final_split=fs,
            cold_aiv_jobs=a.batch*a.parts*2 if cp else a.batch,final_local_aiv_jobs=a.batch*4*(a.parts if fs else 1),final_candidates_per_query=a.parts*2048 if fs else 0,
            aiv_budget=64,scope='post-score controls; no QK/Fetch/Service/Index; immutable mapping'))
    return result

def build(a):
    a.output.mkdir(parents=True,exist_ok=False)
    before={str(p.relative_to(ROOT)):sha(p) for p in HERE.iterdir() if p.suffix in ('.cpp','.h','.txt','.py')}
    source=prepare(a.output/'source')
    commands=[['cmake','-S',str(HERE),'-B',str(a.output/'build'),'-DRUN_MODE='+a.run_mode,'-DLI_TAIL_CONTROLS=ON',
        '-DSOC_VERSION='+a.soc,'-DASCEND_CANN_PACKAGE_PATH='+str(a.cann_root),'-DSCORE_SOURCE='+str(a.output/'source'),'-DKVAF_INCLUDE='+str(a.kvaf_include)],
        ['cmake','--build',str(a.output/'build'),'--target','li_tail','-j',str(a.jobs)]]
    receipt=dict(status='BUILDING',commands=commands,wrapper_sha256=before,prepared_source=source)
    record=a.output/'build-manifest.json';record.write_text(json.dumps(receipt,indent=2)+'\n')
    with (a.output/'build.log').open('w') as log:
        for cmd in commands:
            r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:
                receipt.update(status='FAILED',exit_code=r.returncode);record.write_text(json.dumps(receipt,indent=2)+'\n');raise RuntimeError('build failed: '+str(a.output/'build.log'))
    for p,h in before.items():
        if sha(ROOT/p)!=h:raise RuntimeError('source changed during build: '+p)
    receipt.update(status='BUILT',binary_sha256=sha(a.output/'build/li_tail'),parts_sha256={str(p.relative_to(a.output)):sha(p) for p in (a.output/'build').rglob('*.a')})
    record.write_text(json.dumps(receipt,indent=2)+'\n');print(a.output/'build/li_tail')

def accepted(a,env):
    m=json.loads((a.build/'build-manifest.json').read_text());driver=a.build/'build/li_tail'
    if m['status']!='BUILT' or sha(driver)!=m['binary_sha256']:raise ValueError('exact captured binary required')
    for p,h in m['parts_sha256'].items():
        if sha(a.build/p)!=h:raise ValueError('changed device library')
    caps=json.loads(subprocess.check_output([str(driver),'--capabilities'],env=env,text=True).splitlines()[-1])
    return m,driver,caps

def argv(a,driver,fixture,dest,c,warm,iters):
    return [str(driver),str(fixture),str(dest),str(a.batch),str(a.history),str(a.resident),str(a.parts),str(a.resident_parts),
        str(c['cold_parallel']),str(c['final_split']),a.stage,str(warm),str(iters),str(a.device)]

def map_pcs(folder,build,objdump,receipt,row):
    functions,size,evidence=elf(build/'build/li_tail_kernel_merge_obj_dir/device_aiv.o',objdump)
    logs=sorted(folder.glob('*.instr_log.dump.gz'))+sorted(folder.glob('*.instr_log.dump'))
    entries={}
    for path in logs:
        values=[];opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',errors='replace') as f:
            for line in f:
                m=ENTRY.search(line)
                if m:values.append(int(m[1],16))
        entries[path.name]=values
    proof=row['report'];first=proof['launches'][0]['kernel']
    name=next(n for n in functions if n.startswith(first+'_'))
    candidates=[v[0] for k,v in entries.items() if 'veccore' in k and v]
    if not candidates:raise ValueError('AIV entries missing')
    base=candidates[0]-functions[name][0]
    kernels={n:v for n,v in functions.items() if n.startswith('li_tail_')}
    # These larger prologues read PARA_BASE twice (entry and +0x44).
    # Validate both against the ELF function, but count only its start as entry.
    if any(not any(base+off<=pc<base+off+size for off,size in kernels.values()) for values in entries.values() for pc in values):raise ValueError('unverified runtime prologue PC')
    regions=[]
    for name,(off,n) in functions.items():
        if name.startswith('li_tail_'):
            label=next(k for k in ('li_tail_select','li_tail_gather','li_tail_map','li_tail_resident_finish','li_tail_cold','li_tail_classify','li_tail_union') if name.startswith(k+'_'))
        else:
            owners={'ClassifyTopkHalf':'li_tail_classify','MergeClassifyBitmap':'li_tail_classify','ColdBits':'li_tail_cold','GatherCandidates':'li_tail_gather','MapCandidates':'li_tail_map','ResidentTheta':'li_tail_resident_finish','HotBits':'li_tail_resident_finish','UnionBits':'li_tail_union','GatherVictimScores':'li_tail_select','ClassifyResidentScores':'li_tail_select'}
            label=next((v for k,v in owners.items() if k in name),None)
            if not label:raise ValueError('unassigned named device function '+name)
        regions.append(dict(kernel=label,begin=hex(base+off),end=hex(base+off+n)))
    last=max(off+n for off,n in functions.values())
    if last<size:regions.append(dict(kernel='shared_tail_vf',begin=hex(base+last),end=hex(base+size),shared_owners=[r['kernel'] for r in regions]))
    stats=analyze(folder,regions);stats.update(capture_complete=True,capture_status='MODEL_PASSED',binary_sha256=receipt['binary_sha256'])
    out=folder/'verified-kernel-instructions';out.mkdir()
    (out/'pc-map-proof.json').write_text(json.dumps(dict(prologue_para_reads=entries,base=hex(base),elf=evidence,note='PARA_BASE reads within an ELF kernel may repeat; only function starts are entries'),indent=2)+'\n')
    (out/'pc-map.json').write_text(json.dumps(regions,indent=2)+'\n');(out/'instructions.json').write_text(json.dumps(stats,indent=2)+'\n')
    return stats

def model(a):
    configs=plan(a);env=dict(os.environ,LD_PRELOAD=str(a.model_lib/'libruntime_camodel.so'),LD_LIBRARY_PATH=str(a.model_lib)+':'+os.environ.get('LD_LIBRARY_PATH',''))
    build,driver,caps=accepted(a,env)
    if caps['run_mode']!='sim':raise ValueError('Model needs sim binary')
    a.output.mkdir(parents=True,exist_ok=False);fixture=a.fixture or a.output/'fixture'
    if a.fixture:
        metadata=json.loads((fixture/'fixture.json').read_text())
        if (metadata['batch'],metadata['history'],metadata['heads'])!=(a.batch,a.history,32):raise ValueError('fixture shape changed')
        for n,h in metadata['hashes'].items():
            if sha(fixture/(n+'.bin'))!=h:raise ValueError('fixture changed')
    else:metadata=make_fixture(fixture,a.batch,a.history,32)
    record=dict(status='RUNNING',scope='950 Model post-score; not A5 time',build_manifest=build,fixture=metadata,logged_core_mask=a.core_mask,rows=[])
    path=a.output/'manifest.json';paths=[a.model_lib/'config.json',a.model_lib/'config_stars.json'];original={p:p.read_text() for p in paths}
    with lease('model',0):
        try:
            config=json.loads(original[paths[0]]);stars=json.loads(original[paths[1]])
            config['LOG']['core_enable_mask']=[a.core_mask];config['LOG']['rotating_file_size']=512*1024*1024;config['LOG']['rotating_file_number']=2
            stars['model_top'].update(num_aic=32,num_aiv=64);stars.setdefault('pem',{}).update(parsim=1,parsim_thd_limit=4)
            for p,v in zip(paths,(config,stars)):p.write_text(json.dumps(v,indent=2)+'\n')
            record['loaded_configs']={str(p):json.loads(p.read_text()) for p in paths}
            for c in configs:
                folder=a.output/(c['name']+'_'+a.stage);folder.mkdir();row=dict(c,name=folder.name,stage=a.stage,status='RUNNING')
                command=argv(a,driver,fixture,folder/'oracle',c,0,1);row['argv']=command;record['rows'].append(row);path.write_text(json.dumps(record,indent=2)+'\n');print('Model '+folder.name,flush=True)
                with (folder/'run.log').open('w') as log:
                    r=subprocess.run(['timeout','--signal=TERM','--kill-after=10s',str(a.timeout),*command],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
                row['exit_code']=r.returncode;log=(folder/'run.log').read_text(errors='replace');ticks=re.findall(r'Total tick:\s*(\d+)',log)
                proof=folder/'oracle/report.json';row['report']=json.loads(proof.read_text()) if proof.exists() else {}
                ok=r.returncode==0 and len(ticks)==1 and 'Model stopped successfully.' in log and row['report'].get('status')=='PASS' and row['report'].get('tail_p50_us') is None
                row.update(status='MODEL_PASSED' if ok else 'FAILED',whole_model_ticks=int(ticks[0]) if len(ticks)==1 else None,log_sha256=sha(folder/'run.log'))
                for raw in folder.glob('*.dump'):
                    with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb',compresslevel=1) as dst:shutil.copyfileobj(src,dst)
                    raw.unlink()
                path.write_text(json.dumps(record,indent=2)+'\n')
                if not ok:raise RuntimeError('Model failed: '+str(folder/'run.log'))
                stats=map_pcs(folder,a.build,a.objdump,build,row)
                active=[c for c in stats['coverage'] if c['instructions']]
                row['all_core_issue_envelope']=max(c['last_issue_tick'] for c in active)-min(c['first_issue_tick'] for c in active)
                row['issued_instructions']=sum(c['instructions'] for c in active);row['max_aiv_issue_span']=max(c['last_issue_tick']-c['first_issue_tick'] for c in active)
                row['logged_aiv_cores']=len(active);print(folder.name+' PASS ticks='+str(row['whole_model_ticks']),flush=True)
            record['status']='MODEL_PASSED'
        finally:
            for p,v in original.items():p.write_text(v)
            if record['status']=='RUNNING':record['status']='INCOMPLETE'
            path.write_text(json.dumps(record,indent=2)+'\n')
    rows=[];base=next((r['whole_model_ticks'] for r in record['rows'] if r['name'].startswith('C0_')),None)
    for r in record['rows']:rows.append({k:r[k] for k in ['name','batch','history','resident','parts','resident_parts','whole_model_ticks','all_core_issue_envelope','issued_instructions','max_aiv_issue_span','logged_aiv_cores']}|{'ticks_over_C0':r['whole_model_ticks']/base if base else None})
    with (a.output/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(a.output/'summary.csv')

def bench(a):
    if any(x in os.environ.get('LD_PRELOAD','') for x in ('camodel','cmodel')):raise ValueError('physical A5 benchmark required')
    build,driver,caps=accepted(a,os.environ)
    if caps['run_mode']!='npu':raise ValueError('native binary required')
    a.output.mkdir(parents=True,exist_ok=False);fixture=a.output/'fixture';metadata=make_fixture(fixture,a.batch,a.history,32)
    configs=plan(a);record=dict(status='RUNNING',scope='physical post-score timing; no QK/Fetch',build_manifest=build,fixture=metadata,rows=[])
    with lease('npu',a.device,Path(f'/tmp/kvip-li-qk-device-{a.device}.lease')):
        for rnd in range(a.rounds+1):
            order=list(configs);random.Random(1400+rnd).shuffle(order)
            for c in order:
                folder=a.output/f"r{rnd}_{c['name']}";command=argv(a,driver,fixture,folder,c,a.warmup,a.iterations)
                with (a.output/(folder.name+'.log')).open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if r.returncode:raise RuntimeError('bench failed '+folder.name)
                proof=json.loads((folder/'report.json').read_text());row=dict(c,round=rnd,report=proof,argv=command);record['rows'].append(row)
                (a.output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        record['status']='PASS';(a.output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    rows=[]
    for c in configs:
        measured=[r for r in record['rows'] if r['name']==c['name'] and r['round']>0]
        values=[r['report']['tail_p50_us'] for r in measured]
        ratio=[]
        for r in measured:
            base=next(x for x in record['rows'] if x['name']=='C0' and x['round']==r['round']);ratio.append(r['report']['tail_p50_us']/base['report']['tail_p50_us'])
        rows.append(dict(control=c['name'],batch=a.batch,history=a.history,resident=a.resident,parts=a.parts,resident_parts=a.resident_parts,tail_p50_mean_us=statistics.mean(values),over_C0_mean=statistics.mean(ratio)))
    with (a.output/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(a.output/'summary.csv')

def profile(a):
    if any(x in os.environ.get('LD_PRELOAD','') for x in ('camodel','cmodel')):raise ValueError('physical A5 profile required')
    build,driver,caps=accepted(a,os.environ)
    if caps['run_mode']!='npu' or not shutil.which('msprof'):raise ValueError('NPU binary and msprof required')
    configs=plan(a);a.output.mkdir(parents=True,exist_ok=False);fixture=a.output/'fixture';metadata=make_fixture(fixture,a.batch,a.history,32)
    record=dict(scope='physical msprof post-score; excluded from paired bench',fixture=metadata,build_manifest=build,rows=[])
    with lease('npu',a.device,Path(f'/tmp/kvip-li-qk-device-{a.device}.lease')):
        for c in configs:
            folder=a.output/c['name'];folder.mkdir();command=argv(a,driver,fixture,folder/'oracle',c,4,5)
            app=folder/'application.sh';app.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(command)+'\n');app.chmod(0o700)
            collect=['msprof','--application='+shlex.quote(str(app)),'--output='+str(folder/'trace'),'--task-time=on','--aic-metrics='+a.metrics]
            export=['msprof','--export=on','--output='+str(folder/'trace')]
            row=dict(c,argv=collect,export_argv=export);record['rows'].append(row)
            for name,cmd in [('collect',collect),('export',export)]:
                with (folder/(name+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
                if r.returncode:raise RuntimeError('msprof '+name+' failed '+c['name'])
            proof=json.loads((folder/'oracle/report.json').read_text())
            traces=list((folder/'trace').rglob('*.json'))+list((folder/'trace').rglob('*.csv'))
            if proof['status']!='PASS' or not traces:raise ValueError('profile correctness/export rejected')
            row.update(status='PASS',report=proof,trace_files=[str(x.relative_to(a.output)) for x in traces])
            (a.output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(a.output)

def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='action',required=True)
    s=subs.add_parser('build');s.add_argument('--output',type=Path,required=True);s.add_argument('--cann-root',type=Path,required=True);s.add_argument('--kvaf-include',type=Path,required=True)
    s.add_argument('--soc',default='Ascend950PR_9599');s.add_argument('--run-mode',choices=('sim','npu'),default='npu');s.add_argument('--jobs',type=int,default=4)
    for name in ['plan','model','bench','profile']:
        s=subs.add_parser(name);s.add_argument('--batch',type=int,default=1);s.add_argument('--history',type=int,default=131072);s.add_argument('--resident',type=int,default=16384)
        s.add_argument('--parts',type=int,default=4);s.add_argument('--resident-parts',type=int,default=1);s.add_argument('--controls',choices=list(CONTROLS),nargs='+',default=list(CONTROLS));s.add_argument('--stage',choices=('all','cold','final','resident'),default='all');s.add_argument('--device',type=int,default=0 if name=='model' else 5)
        if name!='plan':s.add_argument('--build',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
        if name=='model':s.add_argument('--model-lib',type=Path,required=True);s.add_argument('--objdump',required=True);s.add_argument('--timeout',type=int,default=600);s.add_argument('--core-mask',default='0xffffffff');s.add_argument('--fixture',type=Path)
        if name=='bench':s.add_argument('--warmup',type=int,default=4);s.add_argument('--iterations',type=int,default=20);s.add_argument('--rounds',type=int,default=3)
        if name=='profile':s.add_argument('--metrics',choices=('PipeUtilization','Memory','ResourceConflictRatio'),default='PipeUtilization');s.add_argument('--timeout',type=int,default=900)
    a=p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    if a.action=='plan':print(json.dumps(plan(a),indent=2));return
    if a.action=='bench' and (a.stage!='all' or 'C0' not in a.controls or a.rounds<1 or a.warmup<1):p.error('paired physical bench needs all/C0/warmup/rounds')
    if a.action=='profile' and a.stage!='all':p.error('physical profile uses all stage')
    {'build':build,'model':model,'bench':bench,'profile':profile}[a.action](a)

if __name__=='__main__':main()
