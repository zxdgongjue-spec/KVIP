#!/usr/bin/env python3
"""CANN950 Model stage captures. Run in an isolated CANN simulator container."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from run import geometry, make_fixture, sha, suite_cases
from cases import lease
from instructions import analyze

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('plan','run'));p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--history',type=int,default=131072)
    p.add_argument('--model-lib',type=Path,required=True,help='dedicated Model library directory containing both config files')
    p.add_argument('--case',action='append',choices=('T1_waves','T2_waves','T2_balanced'))
    p.add_argument('--stage',action='append',choices=('all','score','scatter','topk'))
    p.add_argument('--core-mask',default='0x1',help='logged physical core groups; default core0; 0xffffffff logs all, needs substantial disk')
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--fixture',type=Path,help='reuse an existing exact B16/H/G32 fixture')
    p.add_argument('--parsim',type=int,default=4,help='Model host threads; no change to simulated 32/64 topology')
    a=p.parse_args();a.build=a.build.resolve();a.output=a.output.resolve();a.model_lib=a.model_lib.resolve()
    configs=suite_cases('batch16');stages=a.stage or ['score','scatter','topk']
    rows=[]
    for c in configs:
        name=f"T{c['history_parts']}_{c['schedule']}"
        if a.case and name not in a.case:continue
        for stage in stages:
            rows.append(dict(c,name=name+'_'+stage,stage=stage,geometry=geometry(c['history_parts'],1,a.history,30,60,16,c['schedule'])))
    if a.action=='plan':print(json.dumps(rows,indent=2));return
    if a.timeout<1 or a.parsim<1 or not 0<=int(a.core_mask,16)<=0xffffffff:p.error('valid timeout, threads and 32-bit core mask required')
    build=json.loads((a.build/'build-manifest.json').read_text());driver=a.build/'build/li_qk'
    if build['status']!='BUILT' or sha(driver)!=build['binary_sha256']:p.error('unaccepted build')
    a.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,LD_PRELOAD=str(a.model_lib/'libruntime_camodel.so'),
             LD_LIBRARY_PATH=str(a.model_lib)+':'+os.environ.get('LD_LIBRARY_PATH',''))
    caps=json.loads(subprocess.check_output([str(driver),'--capabilities'],env=env,text=True).splitlines()[-1])
    if caps['run_mode']!='sim':p.error('Model requires a RUN_MODE=sim binary')
    paths=[a.model_lib/'config.json',a.model_lib/'config_stars.json'];original={x:x.read_text() for x in paths}
    manifest=dict(status='RUNNING',scope='CANN950 Model; not A5 timing',history=a.history,batch=16,rows=rows,
                  build_manifest=build,logged_core_mask=a.core_mask,source_configs={str(k):json.loads(v) for k,v in original.items()},
                  dynamic_count_scope='only logged cores; no extrapolation to all cores',stage_input='scatter/topk receive oracle GM scores; all executes actual dependency chain')
    receipt=a.output/'manifest.json'
    with lease('model',0):
        try:
            config=json.loads(original[paths[0]]);stars=json.loads(original[paths[1]])
            config['LOG']['core_enable_mask']=[a.core_mask]
            config['LOG']['rotating_file_size']=512*1024*1024;config['LOG']['rotating_file_number']=2
            stars['model_top'].update(num_aic=32,num_aiv=64)
            stars.setdefault('pem',{}).update(parsim=1,parsim_thd_limit=a.parsim)
            for path,data in ((paths[0],config),(paths[1],stars)):path.write_text(json.dumps(data,indent=2)+'\n')
            manifest['loaded_configs']={str(x):json.loads(x.read_text()) for x in paths}
            fixture=a.fixture.resolve() if a.fixture else a.output/'fixture'
            if a.fixture:
                metadata=json.loads((fixture/'fixture.json').read_text())
                if (metadata['batch'],metadata['history'],metadata['heads'])!=(16,a.history,32):raise ValueError('fixture geometry mismatch')
                for name,h in metadata['hashes'].items():
                    if sha(fixture/(name+'.bin'))!=h:raise ValueError('fixture changed')
                manifest['fixture']=metadata
            else:manifest['fixture']=make_fixture(fixture,16,a.history,32)
            manifest['fixture_path']=str(fixture)
            receipt.write_text(json.dumps(manifest,indent=2)+'\n')
            for row in rows:
                work=a.output/row['name'];work.mkdir();dest=work/'oracle'
                argv=[str(driver),str(fixture),str(dest),str(a.history),'32','30','60',str(row['history_parts']),'1','0','1','0','16',row['schedule'],'li']
                row.update(argv=argv,status='RUNNING');receipt.write_text(json.dumps(manifest,indent=2)+'\n')
                print('Model '+row['name'],flush=True);started=time.monotonic()
                with (work/'run.log').open('w') as f:
                    try:
                        r=subprocess.run(['timeout','--signal=TERM','--kill-after=10s',str(a.timeout),*argv],cwd=work,
                                         env=dict(env,LI_QK_MODEL_STAGE=row['stage']),stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout+20)
                        row['exit_code']=r.returncode
                    except subprocess.TimeoutExpired:row['exit_code']=124
                row['host_wall_seconds']=time.monotonic()-started;row['log_sha256']=sha(work/'run.log')
                log=(work/'run.log').read_text(errors='replace');ticks=re.findall(r'Total tick:\s*(\d+)',log)
                proof=json.loads((dest/'report.json').read_text()) if (dest/'report.json').exists() else {}
                valid=(row['exit_code']==0 and proof.get('status')=='PASS' and proof.get('run_mode')=='sim' and
                       proof.get('model_stage')==row['stage'] and len(ticks)==1 and 'Model stopped successfully.' in log and
                       proof.get('score_p50_us') is None and proof.get('li_p50_us') is None and
                       proof.get('guards') and proof.get('inputs_unchanged') and
                       (row['stage'] not in ('all','topk') or proof.get('topk_threshold_oracle')) and
                       sha(dest/'score-keys.bin')==manifest['fixture']['hashes']['oracle'])
                row.update(status='MODEL_PASSED' if valid else ('TIMEOUT' if row['exit_code']==124 else 'FAILED'),
                           whole_model_ticks=int(ticks[0]) if len(ticks)==1 else None,proof=proof)
                if any(work.glob('*.instr_log.dump')):
                    try:
                        stats=analyze(work);(work/'instructions.json').write_text(json.dumps(stats,indent=2)+'\n');row['dynamic_instructions']='CAPTURED'
                    except ValueError as e:row['dynamic_instructions']=str(e)
                # Preserve own raw artifacts compressed. Warn on rotations, do
                # not claim exact instruction totals from truncated logs.
                row['rotated_logs']=[x.name for x in work.glob('*.dump.*')]
                for raw in work.glob('*.dump'):
                    with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb',compresslevel=1) as dst:shutil.copyfileobj(src,dst)
                    raw.unlink()
                receipt.write_text(json.dumps(manifest,indent=2)+'\n');print(row['name']+' '+row['status'],flush=True)
                if not valid:raise RuntimeError('Model execution failed; see '+str(work/'run.log'))
            manifest['status']='MODEL_PASSED'
        finally:
            for path,data in original.items():path.write_text(data)
            if manifest['status']=='RUNNING':manifest['status']='INCOMPLETE'
            receipt.write_text(json.dumps(manifest,indent=2)+'\n')
    print(a.output)

if __name__=='__main__':main()
