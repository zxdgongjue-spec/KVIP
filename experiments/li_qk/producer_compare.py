#!/usr/bin/env python3
"""Fixed B1/Q4/H128K/P16 producer-fusion controls; no Fetch/Service/Index."""
import argparse,csv,json,os,random,shlex,statistics,subprocess,gzip,shutil,re
from pathlib import Path
from run import HERE,ROOT,prepare,make_fixture,sha
from cases import lease
CASES=('split_scatter','direct','high_fused','local_topk')
def build(a):
 a.output.mkdir(parents=True,exist_ok=False);source=prepare(a.output/'source')
 before={str(p.relative_to(ROOT)):sha(p) for p in HERE.iterdir() if p.suffix in ('.cpp','.h','.txt','.py')}
 cmds=[['cmake','-S',str(HERE),'-B',str(a.output/'build'),'-DRUN_MODE='+a.run_mode,'-DLI_PRODUCER_CONTROLS=ON','-DSOC_VERSION='+a.soc,'-DASCEND_CANN_PACKAGE_PATH='+str(a.cann_root),'-DSCORE_SOURCE='+str(a.output/'source'),'-DKVAF_INCLUDE='+str(a.kvaf_include)],['cmake','--build',str(a.output/'build'),'--target','li_producer','-j','2']]
 m=dict(status='BUILDING',commands=cmds,wrapper_sha256=before,source=source)
 record=a.output/'build-manifest.json';record.write_text(json.dumps(m,indent=2)+'\n')
 with (a.output/'build.log').open('w') as f:
  for cmd in cmds:
   r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
   if r.returncode:raise RuntimeError('build failed: '+str(a.output/'build.log'))
 for path,digest in before.items():
  if sha(ROOT/path)!=digest:raise ValueError('source changed during build: '+path)
 m.update(status='BUILT',binary_sha256=sha(a.output/'build/li_producer'),parts_sha256={str(p.relative_to(a.output)):sha(p) for p in (a.output/'build/lib').glob('*.a')});record.write_text(json.dumps(m,indent=2)+'\n');print(a.output/'build/li_producer')
def summarize(output):
 m=json.loads((output/'manifest.json').read_text());assert m['status']=='PASS'
 for k,v in m['fixture']['hashes'].items():assert sha(output/'fixture'/(k+'.bin'))==v
 selected=m['cases'];expected={(r,c) for r in range(m['rounds']+1) for c in selected}
 assert {(r['round'],r['case']) for r in m['rows']}==expected and len(m['rows'])==len(expected)
 rows=[];lookup={}
 for r in m['rows']:
  d=output/r['name'];p=json.loads((d/'report.json').read_text());assert r['exit_code']==0 and p['status']=='PASS' and all(p[k] for k in ('oracle','guards','inputs_unchanged'))
  assert sha(d/'report.json')==r['report_sha256'] and sha(d/'score-keys.bin')==m['fixture']['hashes']['oracle']
  assert p['route']==CASES.index(r['case']) and p['history']==131072 and p['parts']==16 and p['warmup']==m['warmup'] and p['iterations']==m['iterations']
  assert sha(output/(r['name']+'.log'))==r['log_sha256']
  assert p['run_mode']==('sim' if m['mode']=='model' else 'npu') and p['fetch_launches']==0 and p['full_li_launches']==(3 if p['route']==0 else 2)
  if m['mode']!='model':
   import math
   for metric in ('li','score','tail'):
    v=p[metric+'_samples_us'];assert len(v)==m['iterations'] and all(math.isfinite(x) and x>0 for x in v) and math.isclose(statistics.median(v),p[metric+'_p50_us'],rel_tol=1e-8)
  lookup[r['round'],r['case']]=(r,p)
 for case in selected:
  vals=[lookup[r,case] for r in range(m['rounds']+1) if m['mode']!='bench' or r>0];out=dict(case=case)
  for metric in (('whole_model_ticks',) if m['mode']=='model' else ('li','score','tail')):
   values=[r[metric] if m['mode']=='model' else p[metric+'_p50_us'] for r,p in vals];assert all(v>0 for v in values)
   out[metric+'_mean']=statistics.mean(values)
   base='split_scatter'
   if base in selected:out[metric+'_over_baseline']=statistics.mean(v/(lookup[r['round'],base][0][metric] if m['mode']=='model' else lookup[r['round'],base][1][metric+'_p50_us']) for (r,p),v in zip(vals,values))
  rows.append(out)
 with (output/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(output/'summary.csv')
def execute(a):
 env=dict(os.environ);env.pop('LI_QK_MODEL_STAGE',None);driver=a.build/'build/li_producer';bm=json.loads((a.build/'build-manifest.json').read_text());assert bm['status']=='BUILT' and sha(driver)==bm['binary_sha256']
 if a.action!='model' and ('camodel' in env.get('LD_PRELOAD','') or 'cmodel' in env.get('LD_PRELOAD','')):raise ValueError('physical A5 required')
 for p,h in bm.get('parts_sha256',{}).items():
  if sha(a.build/p)!=h:raise ValueError('changed device library: '+p)
 if a.action=='model':env.update(LD_PRELOAD=str(a.model_lib/'libruntime_camodel.so'),LD_LIBRARY_PATH=str(a.model_lib)+':'+env.get('LD_LIBRARY_PATH',''))
 caps=json.loads(subprocess.check_output([str(driver),'--capabilities'],env=env,text=True).splitlines()[-1]);assert caps['run_mode']==('sim' if a.action=='model' else 'npu')
 a.output.mkdir(parents=True,exist_ok=False);fixture=make_fixture(a.output/'fixture',1,131072,32)
 warm,it=(0,1) if a.action=='model' else (4,20 if a.action=='bench' else 5)
 m=dict(status='RUNNING',source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),source_dirty=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),scope='full numerical LI; includes canonical score output for all routes; no resident/Service/Fetch/Index',mode=a.action,cases=a.cases,rounds=3 if a.action=='bench' else 0,warmup=warm,iterations=it,fixture=fixture,build=bm,rows=[]);record=a.output/'manifest.json';original={}
 with lease('model' if a.action=='model' else 'npu',a.device,None if a.action=='model' else Path(f'/tmp/kvip-li-qk-device-{a.device}.lease')):
  try:
   if a.action=='model':
    for file in ('config.json','config_stars.json'):
     p=a.model_lib/file;original[p]=p.read_text();cfg=json.loads(original[p])
     if file=='config.json':cfg['LOG'].update(core_enable_mask=['0x1'],rotating_file_size=512*1024**2,rotating_file_number=2)
     else:cfg['model_top'].update(num_aic=32,num_aiv=64);cfg.setdefault('pem',{}).update(parsim=1,parsim_thd_limit=4)
     p.write_text(json.dumps(cfg,indent=2)+'\n')
    m['loaded_model_configs']={str(p):json.loads(p.read_text()) for p in original}
   for rnd in range(m['rounds']+1):
    order=list(a.cases)
    if a.action=='bench':random.Random(1820+rnd).shuffle(order)
    for case in order:
     name=f'r{rnd}_{case}';folder=a.output/name;log=a.output/(name+'.log');cmd=[str(driver),str(a.output/'fixture'),str(folder),str(CASES.index(case)),str(warm),str(it),str(a.device)];work=a.output/(name+'_trace');work.mkdir()
     r=dict(name=name,case=case,round=rnd,argv=cmd);m['rows'].append(r);record.write_text(json.dumps(m,indent=2)+'\n');print('Running '+name,flush=True)
     run=cmd
     if a.action=='profile':
      app=work/'application.sh';app.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(cmd)+'\n');app.chmod(0o700);run=['msprof','--application='+shlex.quote(str(app)),'--output='+str(work/'msprof'),'--task-time=on','--aic-metrics=PipeUtilization']
     if a.action=='model':run=['timeout','--signal=TERM','--kill-after=10s',str(a.timeout),*cmd]
     with log.open('w') as f:ret=subprocess.run(run,env=env,cwd=work,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout+30)
     r.update(exit_code=ret.returncode,log_sha256=sha(log))
     if ret.returncode:raise RuntimeError('execution failed: '+str(log))
     r['report_sha256']=sha(folder/'report.json')
     if a.action=='model':
      ticks=re.findall(r'Total tick:\s*(\d+)',log.read_text(errors='replace'));assert len(ticks)==1 and 'Model stopped successfully.' in log.read_text(errors='replace');r['whole_model_ticks']=int(ticks[0])
      for raw in work.glob('*.dump'):
       with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb',compresslevel=1) as dst:shutil.copyfileobj(src,dst)
       raw.unlink()
     if a.action=='profile':subprocess.run(['msprof','--export=on','--output='+str(work/'msprof')],check=True)
   m['status']='PASS'
  finally:
   if m['status']=='RUNNING':m['status']='INCOMPLETE'
   for p,v in original.items():p.write_text(v)
   record.write_text(json.dumps(m,indent=2)+'\n')
 summarize(a.output)
def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 s=sub.add_parser('build');s.add_argument('--output',type=Path,required=True);s.add_argument('--cann-root',type=Path,required=True);s.add_argument('--kvaf-include',type=Path,required=True);s.add_argument('--soc',default='Ascend950PR_9599');s.add_argument('--run-mode',choices=('sim','npu'),default='npu')
 s=sub.add_parser('summarize');s.add_argument('output',type=Path)
 for action in ('model','bench','profile'):
  s=sub.add_parser(action);s.add_argument('--build',type=Path,required=True);s.add_argument('--output',type=Path,required=True);s.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));s.add_argument('--device',type=int,default=0 if action=='model' else 5);s.add_argument('--timeout',type=int,default=1500)
  if action=='model':s.add_argument('--model-lib',type=Path,required=True)
 a=p.parse_args();a.output=a.output.resolve()
 if a.action=='build':build(a)
 elif a.action=='summarize':summarize(a.output)
 else:a.build=a.build.resolve();execute(a)
if __name__=='__main__':main()
