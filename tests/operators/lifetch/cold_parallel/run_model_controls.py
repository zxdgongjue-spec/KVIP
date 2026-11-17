#!/usr/bin/env python3
"""Run actual CANN Model controls in an existing isolated container.

The container must bind this checkout at /repo and --work at /work, plus KVAF
headers. The driver uses actual production score and full algorithmic tail,
prefetch0 and synchronous GM final Fetch. Never treat Model cycles as A5 time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'experiments/split_core'))
from cases import lease

def configs(matrix):
    h,r,c,v=(8192,6400,4,16) if matrix=='smoke' else (131072,16384,28,56)
    base=dict(batch=1,history=h,resident=r,aics=c,aivs=v,miss=205,workers=2)
    rows=[dict(base,name='legacy_W2',route='legacy',windows=2,post='separate',prepare='wave',service='separate',hot=0,cold=0)]
    for windows in (2,8):
        if matrix=='smoke' and windows==8:continue
        for name,post,prepare,service,hot,cold in (
            ('staged_before','separate','wave','separate',0,0),
            ('scatter','scatter','wave','separate',0,0),
            ('direct','direct','wave','separate',0,0),
            ('direct_phase','direct','phase','separate',0,0),
            ('inline_auto','direct','phase','inline',0,0),
            ('inline_Ph4_Pc4' if matrix=='smoke' else 'inline_Ph4_Pc8','direct','phase','inline',4,4 if matrix=='smoke' else 8),
            ('inline_Ph4_Pc4' if matrix=='smoke' else 'inline_Ph8_Pc16','direct','phase','inline',4 if matrix=='smoke' else 8,4 if matrix=='smoke' else 16),
            ('inline_Ph2_Pc4' if matrix=='smoke' else ('inline_Ph26_Pc28' if windows==2 else 'inline_Ph26_Pc26'),
             'direct','phase','inline',2 if matrix=='smoke' else 26,4 if matrix=='smoke' else (28 if windows==2 else 26)),
            ('invocation_Ph4_Pc4' if matrix=='smoke' else 'invocation_Ph4_Pc8','direct','invocation','inline',4,4 if matrix=='smoke' else 8)):
            row=dict(base,name=f'{name}_W{windows}',route='history',windows=windows,post=post,
                     prepare=prepare,service=service,hot=hot,cold=cold)
            if any(all(row[k]==x[k] for k in row if k!='name') for x in rows):continue
            rows.append(row)
    if matrix=='smoke':
        rows.append(dict(base,name='bsq_direct',route='bsq',windows=2,post='direct',prepare='phase',service='inline',hot=0,cold=0))
        rows.append(dict(base,name='multi_owner',batch=3,route='history',windows=2,post='direct',prepare='phase',service='inline',hot=2,cold=2))
    return rows

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_result(row,proof,log,exit_code):
    """Accept only a complete oracle with an independently bounded clock."""
    ticks=re.findall(r'Total tick:\s*(\d+)',log)
    whole=int(ticks[0]) if len(ticks)==1 else None
    start,end=proof.get('model_clock_start',0),proof.get('model_clock_end',0)
    expected_bitmap=0 if row['route']=='legacy' else row['batch']*(row['history']-row['resident'])
    valid=(exit_code==0 and proof.get('status')=='PASSED' and whole is not None
           and 'Model stopped successfully.' in log and 0<start<end<=whole
           and proof.get('forward_model_cycles')==end-start
           and all(proof.get(k)==row[k] for k in ('batch','history','resident'))
           and proof.get('score_route')==row['route'] and proof.get('prefetch_budget')==0
           and proof.get('cold_bitmap_bytes_checked')==expected_bitmap
           and all(proof.get(k) is True for k in ('all_index_fields','all_hot_payload','guards','completion')))
    return valid,whole

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('plan','run'));p.add_argument('--matrix',choices=('smoke','full'),default='smoke')
    p.add_argument('--container',required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',action='append')
    p.add_argument('--timeout',type=int,default=3600);p.add_argument('--driver',default='build/staged_routes')
    p.add_argument('--source-manifest',default='source-v2/staged-source.json')
    p.add_argument('--host-relink-receipt',default='host-relink.json')
    p.add_argument('--aic-budget',type=int);p.add_argument('--aiv-budget',type=int)
    a=p.parse_args();a.work=a.work.resolve();a.output=a.output.resolve()
    a.output.relative_to(a.work)
    rows=configs(a.matrix);names={x['name'] for x in rows}
    if (a.aic_budget is not None and not 1<=a.aic_budget<=32) or (a.aiv_budget is not None and not 2<=a.aiv_budget<=64):
        p.error('invalid independent AIC/AIV budget')
    for row in rows:
        row['status']='NOT_RUN'
        if a.aic_budget is not None:row['aics']=a.aic_budget
        if a.aiv_budget is not None:row['aivs']=a.aiv_budget
    if a.case and not set(a.case)<=names:p.error('unknown case')
    if a.timeout<1:p.error('positive timeout required')
    if a.action=='plan':print(json.dumps(rows,indent=2));return
    if a.output.exists():p.error('fresh output required')
    a.output.mkdir(parents=True)
    binary=a.work/a.driver;binary_sha=digest(binary)
    source_manifest=a.work/a.source_manifest
    if not source_manifest.is_file():p.error('prepared source manifest required')
    fixture_name='fixture-smoke' if a.matrix=='smoke' else 'fixture-full'
    fixture=a.work/fixture_name
    expected=json.loads((fixture/'expected.json').read_text())
    require_h=rows[0]['history'];require_r=rows[0]['resident']
    if (expected['history'],expected['resident'],expected['miss_per_query'])!=(require_h,require_r,205):
        raise RuntimeError('fixture does not match matrix')
    fixture_sha={x.name:digest(x) for x in fixture.iterdir() if x.is_file()}
    metadata=dict(scope='CANN9.1/950 Model; production score; prefetch0; synchronous GM final Fetch; two common clock probes',
                  binary_sha256=binary_sha,fixture_sha256=fixture_sha,rows=rows,
                  source_manifest_sha256=digest(source_manifest),source_manifest_path=str(source_manifest))
    simulator='/usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_3510/lib/'
    metadata['model_config']={name:json.loads(subprocess.check_output(
        ['docker','exec',a.container,'cat',simulator+name],text=True))
        for name in ('config.json','config_stars.json')}
    metadata['binary_parts_sha256']={str(x.relative_to(a.work)):digest(x) for x in
        [binary.parent/'lib/libproducer.a',binary.parent/'lib/libtail.a'] if x.exists()}
    receipt=a.work/a.host_relink_receipt
    metadata['host_relink_receipt']=json.loads(receipt.read_text()) if receipt.exists() else None
    if metadata['host_relink_receipt'] and metadata['host_relink_receipt'].get('binary_sha256')!=binary_sha:
        raise RuntimeError('host relink receipt does not match the driver binary')
    model_shape=metadata['model_config']['config_stars.json']['model_top']
    if any(row['aics']>model_shape['num_aic'] or row['aivs']>model_shape['num_aiv']
           for row in rows if not a.case or row['name'] in a.case):
        raise RuntimeError('application budgets exceed the configured Model resources')
    metadata['runner_sha256']=digest(Path(__file__).resolve())
    (a.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with lease('model',0):
        for row in rows:
            if a.case and row['name'] not in a.case:continue
            run_folder=a.output/(row['name']+'-runtime');run_folder.mkdir()
            dest=a.output/row['name'];mapped=lambda x:'/work/'+str(x.relative_to(a.work))
            cmd=['docker','exec','-w',mapped(run_folder),
                 '-e','LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_3510/lib:/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64:/usr/local/Ascend/cann-9.1.0/aarch64-linux/devlib',
                 '-e','LD_PRELOAD=/usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_3510/lib/libruntime_camodel.so',
                 a.container,'timeout','--signal=TERM','--kill-after=10s',str(a.timeout),
                 '/work/'+a.driver,mapped(fixture),mapped(dest),str(row['batch']),str(row['history']),str(row['resident']),
                 str(row['aics']),str(row['aivs']),str(row['miss']),row['route'],'0',str(row['windows']),row['post'],
                 row['prepare'],row['service'],str(row['hot']),str(row['cold']),str(row['workers'])]
            row.update(status='RUNNING',argv=cmd);start=time.monotonic()
            (a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            log=a.output/(row['name']+'.log');print('Running '+row['name'],flush=True)
            with log.open('w') as out:
                try:
                    result=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=a.timeout+60)
                    exit_code=result.returncode
                except subprocess.TimeoutExpired:
                    # docker exec can outlive its client. Stop only this driver,
                    # then preserve TIMEOUT instead of leaving an obsolete RUNNING row.
                    subprocess.run(['docker','exec',a.container,'pkill','-KILL','-f','^/work/'+re.escape(a.driver)+' '],check=False)
                    exit_code=124
            row.update(exit_code=exit_code,host_wall_seconds=time.monotonic()-start,log_sha256=digest(log))
            proof_path=dest/'correctness.json';proof=json.loads(proof_path.read_text()) if proof_path.exists() else {}
            valid,whole=validate_result(row,proof,log.read_text(errors='replace'),exit_code)
            row.update(status='MODEL_PASSED' if valid else ('TIMEOUT' if exit_code==124 else 'FAILED'),
                       proof=proof,whole_model_ticks=whole)
            (a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(row['name']+' '+row['status'],flush=True)
            if not valid:raise RuntimeError('Model failed; see '+str(log))
    print(a.output/'results.json')

if __name__=='__main__':main()
