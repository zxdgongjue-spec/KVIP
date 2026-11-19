#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Same-fixture real AscendC incremental TopK control; no URMA/Final/score."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent


def parse_result(log):
    if re.search(r'\[ERROR\]|already has same set_flag|double free|corruption|Aborted',log):
        raise RuntimeError('Model/runtime error; numerical PASS alone is not valid timing evidence')
    lines=[line for line in log.splitlines() if line.startswith('QUERY_TOPK_PASS ')]
    if len(lines)!=1:raise RuntimeError('Missing/duplicate successful TopK contract result')
    line=lines[0]
    result={key:value for key,value in re.findall(r'(\w+)=([^ ]+)',line)}
    result['batch']=int(re.search(r'\bB(\d+) ',line).group(1))
    for key in ('topk_keys','candidate_count','candidate_hash','work_span_cycles','max_lane_cycles'):
        result[key]=int(result[key])
    if result['work_span_cycles']<=0 or result['max_lane_cycles']<=0:
        raise RuntimeError('Invalid work cycle measurement')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kvaf-root',type=Path,required=True)
    p.add_argument('--cann',type=Path,default=Path(os.environ.get('ASCEND_HOME_PATH','/usr/local/Ascend/cann')))
    p.add_argument('--soc',default='Ascend950PR_9599')
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--mode',choices=('sim','npu'),default='sim')
    p.add_argument('--batches',default='2,4')
    p.add_argument('--timeout-seconds',type=int,default=900)
    p.add_argument('--output',type=Path)
    a=p.parse_args();batches=[int(x) for x in a.batches.split(',')]
    if not batches or len(set(batches))!=len(batches) or any(b not in (2,4) for b in batches):p.error('Use distinct B2/B4')
    include=a.kvaf_root.resolve()/'access_operator/demo/aiv_batch_get/include'
    if not (include/'kvaf_batch_get_device.h').is_file():p.error('Use patched pinned KVAF device headers')
    out=a.output.resolve() if a.output else Path(tempfile.mkdtemp(prefix='query_topk_'))
    if a.output:out.mkdir(parents=True,exist_ok=False)
    print('Results directory: '+str(out),flush=True)
    build=Path(tempfile.mkdtemp(prefix='build-',dir=out))
    cann=a.cann.resolve();manifest=dict(status='BUILDING',scope='incremental Top2048 + ranked export; no score, theta, post, URMA or Final',
        fixture=dict(history=8192,resident=2048,query=4,rounds=4,epoch=1,covered='empty',score_type='sortable BF16 uint16'),
        compiled_soc=a.soc,mode=a.mode,batches=batches,build=str(build),cann=str(cann))
    source_files=[HERE/'query_topk_smoke.cpp',HERE/'query_model/CMakeLists.txt']
    op=HERE.parents[1]/'csrc/operators/lightning_indexer_with_fetch'
    source_files+=[op/'op_kernel/score_window_service.cpp',*sorted((op/'op_kernel/indexer').glob('*.h')),
                   *sorted((op/'include').glob('*.h'))]
    manifest['source_sha256']={str(path.relative_to(HERE.parents[1])):hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in source_files}
    manifest['samples']=[]
    simulator=cann/'tools/simulator'/a.soc
    manifest['simulator_configs']={name:json.loads((simulator/'lib'/name).read_text())
        for name in ('config.json','config_stars.json') if (simulator/'lib'/name).is_file()}
    config=simulator/'camodel/camodel_v100.json'
    if config.is_file():manifest['camodel_config']=json.loads(config.read_text())
    try:
        for command,name in [(['cmake','-S',str(HERE/'query_model'),'-B',str(build),'-DRUN_MODE='+a.mode,
            '-DSOC_VERSION='+a.soc,'-DASCEND_CANN_PACKAGE_PATH='+str(cann),'-DKVAF_INCLUDE='+str(include)],'configure.log'),
            (['cmake','--build',str(build),'--target','query_topk_smoke','--parallel','1'],'build.log')]:
            with (out/name).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        binary=build/'query_topk_smoke';manifest['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
        env=dict(os.environ)
        paths=[str(cann/'lib64')]
        if a.mode=='sim':
            paths=[str(simulator/'camodel'),str(simulator/'lib')]+paths
            env['LD_PRELOAD']=str(simulator/'lib/libruntime_camodel.so')+(':'+env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
        env['LD_LIBRARY_PATH']=':'.join(paths+[env.get('LD_LIBRARY_PATH','')])
        rows=[]
        for b in batches:
            for route in ('serial','query'):
                name=f'b{b}-{route}';print('Running '+name,flush=True)
                model=out/name;model.mkdir();env['CAMODEL_LOG_PATH']=str(model)
                path=out/(name+'.log')
                attempt=dict(batch=b,route=route,log=str(path),status='RUNNING')
                manifest['samples'].append(attempt)
                try:
                    with path.open('w') as log:
                        run=subprocess.run([str(binary),str(a.device),str(b),route],env=env,
                            stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout_seconds)
                    attempt['exit_code']=run.returncode
                    if run.returncode:raise RuntimeError(f'{name} exit {run.returncode}; inspect {path}')
                finally:
                    if path.exists():attempt['log_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                contents=path.read_text(errors='replace');result=parse_result(contents)
                result.update(log=str(path),log_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                versions=set(re.findall(r'soc_version:(\w+)',contents))
                result['reported_soc']=','.join(sorted(versions)) or None
                result['acl_cleanup_pass']='QUERY_TOPK_CLEANUP_PASS ' in contents
                result['skip_duplicate_model_atexit']='QUERY_TOPK_MODEL_EXIT skip_duplicate_atexit=true' in contents
                if not result['acl_cleanup_pass']:raise RuntimeError('Missing successful explicit ACL cleanup')
                ticks=re.findall(r'Total tick:\s*(\d+)',contents)
                result['whole_model_ticks']=int(ticks[-1]) if ticks else None
                rows.append(result)
                attempt.update(status='PASS',result=result)
                print(next(line for line in contents.splitlines() if line.startswith('QUERY_TOPK_PASS ')),flush=True)
        for row in rows:
            ref=next(r for r in rows if r['batch']==row['batch'] and r['route']=='serial')
            if any(row[k]!=ref[k] for k in ('candidate_count','candidate_hash','topk_keys')):
                raise RuntimeError('Serial/query ranked candidates differ')
            row['work_span_over_serial']=row['work_span_cycles']/ref['work_span_cycles']
        with (out/'summary.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,list(rows[0]));writer.writeheader();writer.writerows(rows)
        manifest.update(status='PASS',results=rows,note='Cycle span is timed work after InitBuffers, before pipe.Destroy; not full kernel or A5 chain latency')
    except BaseException as error:
        for attempt in manifest['samples']:
            if attempt['status']=='RUNNING':attempt.update(status='FAILED',error=str(error))
        manifest.update(status='FAILED',error=str(error));raise
    finally:(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Completed. Summary: '+str(out/'summary.csv'),flush=True)


if __name__=='__main__':main()
