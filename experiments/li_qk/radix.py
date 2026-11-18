#!/usr/bin/env python3
"""Exact sharded radix vs existing per-Q radix; immutable GM score, no QK/Fetch."""
import argparse,bisect,csv,gzip,json,math,os,random,re,shlex,shutil,statistics,subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from run import HERE,ROOT,make_fixture,prepare,sha
from cases import lease
from instructions import LINE
from map_model_pcs import ENTRY,elf
from tail_instructions import classify_tail

def definitions(history=131072):
    rows=[]
    for b,parts in ((1,(1,4,16,32)),(16,(1,2,4))):
        for route,p in [('original',1)]+[('sharded',p) for p in parts]:
            tasks=b*4*p;workers=min(64,tasks)
            rows.append(dict(name=f'B{b}_{route}_P{p}',batch=b,history=history,parts=p,route=route,
                aiv_budget=64,scan_workers=b*4 if route=='original' else workers,
                tasks_per_scan=tasks if route=='sharded' else b*4,
                max_segments_per_aiv=(tasks+workers-1)//workers,
                launches=1 if route=='original' else 5))
    return rows

def build(a):
    target=getattr(a,'target','li_radix')
    if target not in ('li_radix','li_qk_radix'):raise ValueError('invalid build target')
    a.output.mkdir(parents=True,exist_ok=False)
    before={str(p.relative_to(ROOT)):sha(p) for p in HERE.iterdir() if p.suffix in ('.cpp','.h','.txt','.py')}
    source=prepare(a.output/'source');commands=[['cmake','-S',str(HERE),'-B',str(a.output/'build'),'-DRUN_MODE='+a.run_mode,
        '-DLI_RADIX_CONTROLS=ON','-DSOC_VERSION='+a.soc,'-DASCEND_CANN_PACKAGE_PATH='+str(a.cann_root),
        '-DSCORE_SOURCE='+str(a.output/'source'),'-DKVAF_INCLUDE='+str(a.kvaf_include)],
        ['cmake','--build',str(a.output/'build'),'--target',target,'-j',str(a.jobs)]]
    record=a.output/'build-manifest.json';receipt=dict(status='BUILDING',commands=commands,wrapper_sha256=before,prepared_source=source)
    record.write_text(json.dumps(receipt,indent=2)+'\n')
    with (a.output/'build.log').open('w') as log:
        for cmd in commands:
            r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:
                receipt.update(status='FAILED',exit_code=r.returncode);record.write_text(json.dumps(receipt,indent=2)+'\n');raise RuntimeError('build failed: '+str(a.output/'build.log'))
    for p,h in before.items():
        if sha(ROOT/p)!=h:raise ValueError('source changed during build: '+p)
    receipt.update(status='BUILT',target=target,binary_sha256=sha(a.output/'build'/target),parts_sha256={str(p.relative_to(a.output)):sha(p) for p in (a.output/'build').rglob('*.a')})
    record.write_text(json.dumps(receipt,indent=2)+'\n');print(a.output/'build'/target)

def fixture(path,b,h,distribution):
    if distribution=='score':return make_fixture(path,b,h,32)
    path.mkdir();rng=np.random.default_rng(1418)
    keys=rng.integers(0,65536,(b*4,h),dtype=np.uint16)
    if distribution=='ties':keys[:]=np.uint16(0x8100)
    if distribution=='zeros':keys[:]=0
    if distribution=='skew':keys[:]=np.uint16(0x8100);keys[:,::61]=np.uint16(0xffff)
    keys.tofile(path/'oracle.bin');m=dict(batch=b,history=h,heads=32,distribution=distribution,hashes=dict(oracle=sha(path/'oracle.bin')))
    (path/'fixture.json').write_text(json.dumps(m,indent=2)+'\n');return m

def verify_build(a,env):
    record=a.build/'build-manifest.json';m=json.loads(record.read_text());driver=a.build/'build/li_radix'
    if m['status']!='BUILT' or sha(driver)!=m['binary_sha256']:raise ValueError('exact accepted binary required')
    for p,h in m['parts_sha256'].items():
        if sha(a.build/p)!=h:raise ValueError('changed device library')
    caps=json.loads(subprocess.check_output([str(driver),'--capabilities'],text=True,env=env).splitlines()[-1]);return m,driver,caps

def arguments(a,driver,c,fixture_path,out,warm,iters):
    return [str(driver),str(fixture_path),str(out),str(c['batch']),str(a.history),str(c['parts']),c['route'],str(warm),str(iters),str(a.device)]

def instructions(folder,build,objdump,row):
    target='li_post' if row['route']=='original' else 'li_radix_kernel'
    functions,size,evidence=elf(build/f'build/{target}_merge_obj_dir/device_aiv.o',objdump)
    first=row['report']['launches'][0]['kernel'];symbol=next(n for n in functions if re.fullmatch(re.escape(first)+r'_\d+',n))
    paths=sorted(folder.glob('*.instr_log.dump.gz'));active=[];base=None
    for path in paths:
        if 'veccore' not in path.name:continue
        with gzip.open(path,'rt',errors='replace') as f:
            for line in f:
                match=ENTRY.search(line)
                if match:base=int(match[1],16)-functions[symbol][0];break
        if base is not None:break
    if base is None:raise ValueError('missing actual kernel entry')
    starts={base+off:n.rsplit('_',1)[0] for n,(off,sz) in functions.items() if re.fullmatch(r'li_(?:radix_\w+|qk_topk)_\d+',n)}
    counts=Counter();ops=Counter();cats=Counter();bounds={};cores=[];entry_proof={}
    for path in paths:
        match=re.fullmatch(r'core(\d+)\.veccore([01])\.instr_log.dump.gz',path.name)
        if not match:continue
        physical=int(match[1])*2+int(match[2]);expected=[x for x in row['report']['launches'] if physical<x['cores']];entries=[]
        with gzip.open(path,'rt',errors='replace') as f:
            for line in f:
                e=ENTRY.search(line);m=LINE.search(line)
                if e and m and int(e[1],16) in starts:entries.append((int(m[1]),starts[int(e[1],16)]))
        entries.sort()
        if [k for t,k in entries]!=[x['kernel'] for x in expected]:raise ValueError('launch/ELF sequence mismatch '+path.name)
        if not entries:continue
        times=[t for t,k in entries];entry_proof[path.name]=entries;local=Counter();localbounds={}
        with gzip.open(path,'rt',errors='replace') as f:
            for line in f:
                m=LINE.search(line)
                if not m:continue
                tick,pc,pipe,ident,opcode=m.groups();tick=int(tick);i=bisect.bisect_right(times,tick)-1
                if i<0:raise ValueError('instruction before verified entry')
                label=expected[i]['kernel'];counts[label]+=1;local[label]+=1;ops[(label,pipe,opcode)]+=1;cats[(label,classify_tail(pipe,opcode))]+=1
                for dst in (bounds,localbounds):
                    v=dst.setdefault(label,[tick,tick]);v[0]=min(v[0],tick);v[1]=max(v[1],tick)
        cores.append(dict(core=path.name,instructions=sum(local.values()),kernels=dict(local),bounds=localbounds))
    if len(cores)!=max(x['cores'] for x in row['report']['launches']):raise ValueError('all active AIV logs required')
    total=sum(counts.values());phases=[dict(kernel=k,count=n,share_of_issued=n/total,issue_envelope=bounds[k][1]-bounds[k][0]) for k,n in counts.items()]
    result=dict(scope='all active AIV issued instructions and issue windows; not A5 cycles',capture_complete=True,
        issued_instructions=total,phases=phases,cores=cores,
        opcodes=[dict(kernel=k,pipeline=p,opcode=o,count=n,share_of_kernel=n/counts[k]) for (k,p,o),n in ops.items()],
        categories=[dict(kernel=k,category=c,count=n,share_of_kernel=n/counts[k]) for (k,c),n in cats.items()],
        proof=dict(base=hex(base),elf=evidence,entries=entry_proof))
    dest=folder/'verified-instructions';dest.mkdir();(dest/'instructions.json').write_text(json.dumps(result,indent=2)+'\n')
    for key in ('phases','opcodes','categories'):
        with (dest/(key+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[key][0]));w.writeheader();w.writerows(result[key])
    return result

def summarize(output):
    m=json.loads((output/'manifest.json').read_text());rows=m['rows'];mode=m['mode']
    if m['status'] not in ('PASS','MODEL_PASSED'):raise ValueError('completed run required')
    for b,fixture_meta in m['fixtures'].items():
        for key,digest in fixture_meta['hashes'].items():
            if sha(output/f'fixture_b{b}'/(key+'.bin'))!=digest:raise ValueError('changed input fixture')
    expected={(r,c['name']) for r in range(m['rounds']+1) for c in m['definitions']} if mode=='bench' else {(0,c['name']) for c in m['definitions']}
    if len(rows)!=len(expected) or {(r['round'],r['config']) for r in rows}!=expected:raise ValueError('complete planned cases including r0 required')
    lookup={}
    for r in rows:
        proof=r['report'];folder=output/r['name'];report=folder/'oracle/report.json' if mode=='model' or mode=='profile' else folder/'report.json'
        log=folder/'run.log' if mode=='model' else output/(r['name']+'.log')
        if r['exit_code']!=0 or proof.get('status')!='PASS' or not all(proof.get(k) for k in ('oracle','guards','inputs_unchanged')):raise ValueError('failed oracle')
        if sha(report)!=r['report_sha256'] or sha(log)!=r['log_sha256']:raise ValueError('changed evidence')
        if tuple(proof[k] for k in ('batch','history','parts','route','aiv_budget','k','score_launches','fetch_launches'))!=(r['batch'],m['history'],r['parts'],r['route'],64,2048,0,0):raise ValueError('scope mismatch')
        queries=r['batch']*4;workers=min(64,queries*r['parts'])
        expected_launches=[dict(kernel='li_qk_topk',cores=queries)] if r['route']=='original' else [dict(kernel=k,cores=cores) for k,cores in (
            ('li_radix_high',workers),('li_radix_high_reduce',queries),('li_radix_low',workers),('li_radix_low_reduce',queries),('li_radix_emit',workers))]
        if proof['launches']!=expected_launches:raise ValueError('unexpected kernels/core budget')
        if mode=='model':
            if r['status']!='MODEL_PASSED' or proof['radix_p50_us'] is not None or not r['whole_model_ticks'] or proof['warmup']!=0 or proof['iterations']!=1:raise ValueError('incomplete Model')
            stats=folder/'verified-instructions/instructions.json'
            if sha(stats)!=r['instruction_sha256']:raise ValueError('changed instruction evidence')
        elif mode=='bench':
            v=proof['samples_us']
            if proof['warmup']!=m['warmup'] or proof['iterations']!=m['iterations'] or len(v)!=m['iterations'] or any(not math.isfinite(x) or x<=0 for x in v) or not math.isclose(proof['radix_p50_us'],statistics.median(v),rel_tol=1e-8):raise ValueError('invalid p50')
        lookup[(r['round'],r['config'])]=r
    summary=[]
    for c in m['definitions']:
        chosen=[r for r in rows if r['config']==c['name'] and (mode!='bench' or r['round']>0)]
        metric='whole_model_ticks' if mode=='model' else 'radix_p50_us';values=[r[metric] if mode=='model' else r['report'][metric] for r in chosen]
        row={k:c[k] for k in ('name','batch','history','route','parts','scan_workers','max_segments_per_aiv','launches')};row[metric+'_mean']=statistics.mean(values)
        for suffix,reference in (('over_original',f'B{c["batch"]}_original_P1'),('over_sharded_P1',f'B{c["batch"]}_sharded_P1')):
            ratios=[]
            for r,value in zip(chosen,values):
                base=lookup.get((r['round'],reference))
                if base:ratios.append(value/(base[metric] if mode=='model' else base['report'][metric]))
            row[suffix+'_mean']=statistics.mean(ratios) if ratios else None;row[suffix+'_per_round']=ratios
        if mode=='model':row['issued_instructions']=chosen[0]['issued_instructions']
        summary.append(row)
    with (output/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(output/'summary.csv')

def execute(a):
    configs=definitions(a.history)
    if a.cases:
        names=a.cases.split(',');configs=[c for c in configs if c['name'] in names]
        if len(configs)!=len(names):raise ValueError('unknown/duplicate case')
    if a.action=='plan':print(json.dumps(configs,indent=2));return
    env=dict(os.environ)
    if a.action=='model':env.update(LD_PRELOAD=str(a.model_lib/'libruntime_camodel.so'),LD_LIBRARY_PATH=str(a.model_lib)+':'+env.get('LD_LIBRARY_PATH',''))
    elif any(x in env.get('LD_PRELOAD','') for x in ('camodel','cmodel')):raise ValueError('physical NPU required')
    build,driver,caps=verify_build(a,env)
    if caps['run_mode']!=('sim' if a.action=='model' else 'npu'):raise ValueError('run mode mismatch')
    if a.action=='profile' and not shutil.which('msprof'):raise ValueError('msprof required')
    a.output.mkdir(parents=True,exist_ok=False);fixtures={}
    for b in sorted({c['batch'] for c in configs}):fixtures[b]=fixture(a.output/f'fixture_b{b}',b,a.history,a.distribution)
    m=dict(status='RUNNING',mode=a.action,scope='numeric radix only; immutable score; no QK/resident/Cold/Fetch',history=a.history,distribution=a.distribution,
        build_manifest=build,fixtures=fixtures,definitions=configs,rounds=a.rounds if a.action=='bench' else 0,warmup=a.warmup,iterations=a.iterations,rows=[])
    record=a.output/'manifest.json';paths=[];original={}
    with lease('model' if a.action=='model' else 'npu',a.device, None if a.action=='model' else Path(f'/tmp/kvip-li-qk-device-{a.device}.lease')):
        try:
            if a.action=='model':
                paths=[a.model_lib/'config.json',a.model_lib/'config_stars.json'];original={p:p.read_text() for p in paths}
                cfg=json.loads(original[paths[0]]);stars=json.loads(original[paths[1]]);cfg['LOG'].update(core_enable_mask=['0xffffffff'],rotating_file_size=512*1024*1024,rotating_file_number=2)
                stars['model_top'].update(num_aic=32,num_aiv=64);stars.setdefault('pem',{}).update(parsim=1,parsim_thd_limit=4)
                for p,v in zip(paths,(cfg,stars)):p.write_text(json.dumps(v,indent=2)+'\n')
                m['loaded_configs']={str(p):json.loads(p.read_text()) for p in paths}
            for rnd in range(m['rounds']+1):
                order=list(configs)
                if a.action=='bench':random.Random(1418+rnd).shuffle(order)
                for c in order:
                    name=f'r{rnd}_{c["name"]}';folder=a.output/name
                    if a.action in ('model','profile'):folder.mkdir()
                    dest=folder/'oracle' if a.action in ('model','profile') else folder
                    command=arguments(a,driver,c,a.output/f'fixture_b{c["batch"]}',dest,0 if a.action=='model' else a.warmup,1 if a.action=='model' else a.iterations)
                    row=dict(c,name=name,config=c['name'],round=rnd,argv=command,status='RUNNING');m['rows'].append(row);record.write_text(json.dumps(m,indent=2)+'\n');print('Running '+name,flush=True)
                    log=folder/'run.log' if a.action=='model' else a.output/(name+'.log');run_cmd=command
                    if a.action=='profile':
                        app=folder/'application.sh';app.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(command)+'\n');app.chmod(0o700)
                        run_cmd=['msprof','--application='+shlex.quote(str(app)),'--output='+str(folder/'trace'),'--task-time=on','--aic-metrics=PipeUtilization']
                    if a.action=='model':run_cmd=['timeout','--signal=TERM','--kill-after=10s',str(a.timeout),*command]
                    with log.open('w') as f:r=subprocess.run(run_cmd,cwd=folder if a.action=='model' else None,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout+30)
                    row['exit_code']=r.returncode;report=dest/'report.json';row['report']=json.loads(report.read_text()) if report.exists() else {};row['log_sha256']=sha(log)
                    if report.exists():row['report_sha256']=sha(report)
                    if a.action=='model':
                        text=log.read_text(errors='replace');ticks=re.findall(r'Total tick:\s*(\d+)',text)
                        ok=r.returncode==0 and len(ticks)==1 and 'Model stopped successfully.' in text and row['report'].get('status')=='PASS'
                        row.update(status='MODEL_PASSED' if ok else 'FAILED',whole_model_ticks=int(ticks[0]) if len(ticks)==1 else None)
                        for raw in folder.glob('*.dump'):
                            with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb',compresslevel=1) as dst:shutil.copyfileobj(src,dst)
                            raw.unlink()
                        record.write_text(json.dumps(m,indent=2)+'\n')
                        if not ok:raise RuntimeError('Model failed '+str(log))
                        stats=instructions(folder,a.build,a.objdump,row);row['issued_instructions']=stats['issued_instructions']
                        row['instruction_sha256']=sha(folder/'verified-instructions/instructions.json')
                        print(name+' PASS ticks='+str(row['whole_model_ticks']),flush=True)
                    elif r.returncode or row['report'].get('status')!='PASS':raise RuntimeError('device failed '+str(log))
                    else:row['status']='PASS'
                    if a.action=='profile':
                        subprocess.run(['msprof','--export=on','--output='+str(folder/'trace')],check=True,stdout=subprocess.DEVNULL)
                        if not list((folder/'trace').rglob('*.csv')) and not list((folder/'trace').rglob('*.json')):raise ValueError('empty exported trace')
                    record.write_text(json.dumps(m,indent=2)+'\n')
            m['status']='MODEL_PASSED' if a.action=='model' else 'PASS'
        finally:
            for p,v in original.items():p.write_text(v)
            if m['status']=='RUNNING':m['status']='INCOMPLETE'
            record.write_text(json.dumps(m,indent=2)+'\n')
    if a.action!='profile':summarize(a.output)

def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='action',required=True)
    s=subs.add_parser('build');s.add_argument('--output',type=Path,required=True);s.add_argument('--cann-root',type=Path,required=True);s.add_argument('--kvaf-include',type=Path,required=True)
    s.add_argument('--soc',default='Ascend950PR_9599');s.add_argument('--run-mode',choices=('sim','npu'),default='npu');s.add_argument('--jobs',type=int,default=2)
    for action in ('plan','model','bench','profile'):
        s=subs.add_parser(action);s.add_argument('--history',type=int,default=131072);s.add_argument('--cases');s.add_argument('--distribution',choices=('score','uniform','ties','zeros','skew'),default='score')
        if action=='plan':continue
        s.add_argument('--build',type=Path,required=True);s.add_argument('--output',type=Path,required=True);s.add_argument('--device',type=int,default=0 if action=='model' else 5);s.add_argument('--timeout',type=int,default=1200)
        s.add_argument('--warmup',type=int,default=4);s.add_argument('--iterations',type=int,default=20 if action=='bench' else 5);s.add_argument('--rounds',type=int,default=3)
        if action=='model':s.add_argument('--model-lib',type=Path,required=True);s.add_argument('--objdump',required=True)
    s=subs.add_parser('summarize');s.add_argument('output',type=Path)
    a=p.parse_args()
    if a.action=='summarize':summarize(a.output.resolve());return
    if a.action=='build':a.output=a.output.resolve();build(a);return
    if a.history<4096 or a.history>131072 or a.history%256:p.error('aligned H4096..131072 required')
    if a.action!='plan':
        if a.rounds<1 or not 1<=a.warmup<=100 or not 1<=a.iterations<=1000 or a.timeout<1:p.error('valid sampling required')
        a.build=a.build.resolve();a.output=a.output.resolve()
    execute(a)

if __name__=='__main__':main()
