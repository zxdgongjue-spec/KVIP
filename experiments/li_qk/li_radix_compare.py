#!/usr/bin/env python3
"""B1 identical-input full LI: serial, equal score P16, equal P16 + sharded radix."""
import argparse,csv,gzip,json,math,os,random,shlex,shutil,statistics,subprocess
from pathlib import Path
import radix
from run import HERE,make_fixture,sha
from cases import lease

CASES=(('serial_li',1,'original'),('uniform_p16',16,'original'),('uniform_p16_radix',16,'sharded'))

def definitions():
    return [dict(name=name,score_parts=p,topk_route=route,score_aics=p,score_aivs=p*2,
        radix_parts=16 if route=='sharded' else 1,tail_peak_aivs=64 if route=='sharded' else 4,
        scatter_launches=0 if p==1 else 1,tail_kernel_launches=5 if route=='sharded' else 1,
        full_li_kernel_launches=2 if p==1 else (7 if route=='sharded' else 3)) for name,p,route in CASES]

def summarize(output):
    m=json.loads((output/'manifest.json').read_text());mode=m['mode']
    if m['status']!='PASS':raise ValueError('completed comparison required')
    for key,digest in m['fixture']['hashes'].items():
        if sha(output/'fixture'/(key+'.bin'))!=digest:raise ValueError('input fixture changed')
    rounds=range(m['rounds']+1) if mode=='bench' else (0,)
    expected={(r,c['name']) for r in rounds for c in definitions()};rows=m['rows']
    if len(rows)!=len(expected) or {(r['round'],r['case']['name']) for r in rows}!=expected:raise ValueError('all three paired cases and r0 required')
    lookup={}
    for row in rows:
        c=row['case'];folder=output/row['name'];p=folder/'report.json';log=output/(row['name']+'.log');r=json.loads(p.read_text())
        if row['exit_code'] or sha(p)!=row['report_sha256'] or sha(log)!=row['log_sha256']:raise ValueError('changed/failed evidence')
        fields=('status','run_mode','batch','queries','history','heads','history_parts','query_parts','score_peak_aics','score_peak_aivs','score_launches','scatter_launches','topk_route','radix_parts','tail_kernel_launches','tail_peak_aivs','full_li_kernel_launches','fetch_launches','full_valid_history')
        wanted=('PASS','sim' if mode=='model' else 'npu',1,4,m['history'],32,c['score_parts'],1,c['score_aics'],c['score_aivs'],1,c['scatter_launches'],c['topk_route'],c['radix_parts'],c['tail_kernel_launches'],c['tail_peak_aivs'],c['full_li_kernel_launches'],0,True)
        if tuple(r[k] for k in fields)!=wanted or not all(r[k] for k in ('guards','inputs_unchanged','topk_threshold_oracle')):raise ValueError('scope/oracle mismatch')
        if sha(folder/'score-keys.bin')!=m['fixture']['hashes']['oracle']:raise ValueError('computed canonical keys differ')
        if mode=='model':
            if r['warmup']!=0 or r['iterations']!=1 or any(r[k+'_p50_us'] is not None for k in ('score','topk','li')) or not row['whole_model_ticks']:raise ValueError('invalid Model receipt')
        else:
            if r['warmup']!=m['warmup'] or r['iterations']!=m['iterations']:raise ValueError('sampling mismatch')
            for metric in ('score','topk','li'):
                v=r[metric+'_samples_us'];median=r[metric+'_p50_us']
                if len(v)!=m['iterations'] or any(not math.isfinite(x) or x<=0 for x in v) or not math.isclose(statistics.median(v),median,rel_tol=1e-8):raise ValueError('invalid device samples')
        lookup[row['round'],c['name']]=(row,r)
    result=[]
    for c in definitions():
        selected=[lookup[r,c['name']] for r in rounds if mode!='bench' or r>0]
        out=dict(c)
        for metric in (('whole_model_ticks',) if mode=='model' else ('score','topk','li')):
            values=[row[metric] if mode=='model' else proof[metric+'_p50_us'] for row,proof in selected]
            out['whole_model_ticks_mean' if mode=='model' else metric+'_p50_mean_us']=statistics.mean(values)
            for suffix,base in (('over_serial','serial_li'),('over_uniform_p16','uniform_p16')):
                ratios=[]
                for (row,proof),value in zip(selected,values):
                    br,bp=lookup[row['round'],base];ratios.append(value/(br[metric] if mode=='model' else bp[metric+'_p50_us']))
                out[metric+'_'+suffix]=statistics.mean(ratios)
        result.append(out)
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]),lineterminator='\n');w.writeheader();w.writerows(result)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(output/'summary.csv')

def execute(a):
    env=dict(os.environ)
    env['LI_QK_MODEL_STAGE']='all'
    if a.action=='model':env.update(LD_PRELOAD=str(a.model_lib/'libruntime_camodel.so'),LD_LIBRARY_PATH=str(a.model_lib)+':'+env.get('LD_LIBRARY_PATH',''))
    elif 'camodel' in env.get('LD_PRELOAD','') or 'cmodel' in env.get('LD_PRELOAD',''):raise ValueError('physical NPU required')
    build=json.loads((a.build/'build-manifest.json').read_text());driver=a.build/'build/li_qk_radix'
    if build['status']!='BUILT' or build.get('target')!='li_qk_radix' or sha(driver)!=build['binary_sha256']:raise ValueError('accepted li_qk_radix build required')
    for key,digest in build['parts_sha256'].items():
        if sha(a.build/key)!=digest:raise ValueError('device library changed')
    caps=json.loads(subprocess.check_output([str(driver),'--capabilities'],env=env,text=True).splitlines()[-1])
    if caps['run_mode']!=('sim' if a.action=='model' else 'npu'):raise ValueError('build run mode mismatch')
    a.output.mkdir(parents=True,exist_ok=False);fixture=make_fixture(a.output/'fixture',1,a.history,32)
    m=dict(status='RUNNING',mode=a.action,history=a.history,batch=1,queries=4,heads=32,k=2048,fixture=fixture,build_manifest=build,
        scope='full standalone numeric LI; no SDK/Fetch/resident/Service/Index',rounds=a.rounds if a.action=='bench' else 0,warmup=a.warmup,iterations=a.iterations,definitions=definitions(),rows=[])
    record=a.output/'manifest.json';original={}
    with lease('model' if a.action=='model' else 'npu',a.device,None if a.action=='model' else Path(f'/tmp/kvip-li-qk-device-{a.device}.lease')):
        try:
            if a.action=='model':
                paths=[a.model_lib/'config.json',a.model_lib/'config_stars.json'];original={p:p.read_text() for p in paths}
                cfg=json.loads(original[paths[0]]);stars=json.loads(original[paths[1]])
                cfg['LOG'].update(core_enable_mask=['0xffffffff'],rotating_file_size=512*1024**2,rotating_file_number=2)
                stars['model_top'].update(num_aic=32,num_aiv=64);stars.setdefault('pem',{}).update(parsim=1,parsim_thd_limit=4)
                for p,v in zip(paths,(cfg,stars)):p.write_text(json.dumps(v,indent=2)+'\n')
                m['loaded_model_configs']={str(p):json.loads(p.read_text()) for p in paths}
            for rnd in range(m['rounds']+1):
                order=definitions()
                if a.action=='bench':random.Random(1418+rnd).shuffle(order)
                for c in order:
                    name=f'r{rnd}_{c["name"]}';folder=a.output/name;log=a.output/(name+'.log')
                    cmd=[str(driver),str(a.output/'fixture'),str(folder),str(a.history),'32','32','64',str(c['score_parts']),'1',str(0 if a.action=='model' else a.warmup),str(1 if a.action=='model' else a.iterations),str(a.device),'1','waves','li' if a.action=='model' else 'both','2048',c['topk_route']]
                    row=dict(name=name,round=rnd,case=c,argv=cmd);m['rows'].append(row);record.write_text(json.dumps(m,indent=2)+'\n');print('Running '+name,flush=True)
                    run_cmd=cmd;work=a.output/(name+'_trace')
                    if a.action=='model':work.mkdir();run_cmd=['timeout','--signal=TERM','--kill-after=10s',str(a.timeout),*cmd]
                    if a.action=='profile':
                        work.mkdir();app=work/'application.sh';app.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(cmd)+'\n');app.chmod(0o700)
                        run_cmd=['msprof','--application='+shlex.quote(str(app)),'--output='+str(work/'msprof'),'--task-time=on','--aic-metrics=PipeUtilization']
                    with log.open('w') as f:r=subprocess.run(run_cmd,cwd=work if a.action=='model' else None,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout+30)
                    row.update(exit_code=r.returncode,log_sha256=sha(log))
                    if (folder/'report.json').exists():row['report_sha256']=sha(folder/'report.json')
                    if a.action=='model':
                        import re
                        ticks=re.findall(r'Total tick:\s*(\d+)',log.read_text(errors='replace'));row['whole_model_ticks']=int(ticks[0]) if len(ticks)==1 else None
                        for raw in work.glob('*.dump'):
                            with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb',compresslevel=1) as dst:shutil.copyfileobj(src,dst)
                            raw.unlink()
                        if not row['whole_model_ticks'] or 'Model stopped successfully.' not in log.read_text(errors='replace'):raise RuntimeError('incomplete Model: '+str(log))
                    if r.returncode or not (folder/'report.json').exists():raise RuntimeError('execution failed: '+str(log))
                    if a.action=='profile':
                        subprocess.run(['msprof','--export=on','--output='+str(work/'msprof')],check=True)
                        if not list((work/'msprof').rglob('*.csv')) and not list((work/'msprof').rglob('*.json')):raise ValueError('empty msprof export')
                    record.write_text(json.dumps(m,indent=2)+'\n')
            m['status']='PASS'
        finally:
            for p,v in original.items():p.write_text(v)
            if m['status']=='RUNNING':m['status']='INCOMPLETE'
            record.write_text(json.dumps(m,indent=2)+'\n')
    summarize(a.output)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('build');s.add_argument('--output',type=Path,required=True);s.add_argument('--cann-root',type=Path,required=True);s.add_argument('--kvaf-include',type=Path,required=True)
    s.add_argument('--soc',default='Ascend950PR_9599');s.add_argument('--run-mode',choices=('npu','sim'),default='npu');s.add_argument('--jobs',type=int,default=2)
    sub.add_parser('plan')
    s=sub.add_parser('summarize');s.add_argument('output',type=Path)
    for mode in ('bench','profile','model'):
        s=sub.add_parser(mode);s.add_argument('--build',type=Path,required=True);s.add_argument('--output',type=Path,required=True);s.add_argument('--device',type=int,default=0 if mode=='model' else 5)
        s.add_argument('--history',type=int,default=131072);s.add_argument('--warmup',type=int,default=4);s.add_argument('--iterations',type=int,default=20 if mode=='bench' else 5);s.add_argument('--rounds',type=int,default=3);s.add_argument('--timeout',type=int,default=1200)
        if mode=='model':s.add_argument('--model-lib',type=Path,required=True)
    a=p.parse_args()
    if a.action=='plan':print(json.dumps(definitions(),indent=2));return
    if a.action=='summarize':summarize(a.output.resolve());return
    a.output=a.output.resolve()
    if a.action=='build':a.target='li_qk_radix';radix.build(a);return
    if not 8192<=a.history<=131072 or a.history%4096 or not 1<=a.warmup<=100 or not 1<=a.iterations<=1000 or a.rounds<1 or a.timeout<1:p.error('valid H8192..128K aligned4096 and sampling required')
    a.build=a.build.resolve();execute(a)

if __name__=='__main__':main()
