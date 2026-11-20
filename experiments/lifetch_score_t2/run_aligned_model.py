#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate real aligned score/TopK/classify kernels; no URMA or full-chain timing."""
import argparse
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
        raise RuntimeError('Model/runtime error; disregard timings')
    lines=[s for s in log.splitlines() if s.startswith('ALIGNED_PASS ')]
    if len(lines)!=1 or 'ALIGNED_CLEANUP_PASS' not in log:
        raise RuntimeError('Missing unique numerical/state/ACL success')
    result=dict(re.findall(r'(\w+)=([^ ]+)',lines[0]))
    result['batch']=int(re.search(r'\bB(\d+) ',lines[0]).group(1))
    for key in ('H','R','selected','hash','work_span_cycles'):result[key]=int(result[key])
    if 'parts' in result:result['parts']=int(result['parts'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kvaf-root',type=Path,required=True)
    p.add_argument('--cann',type=Path,default=Path('/usr/local/Ascend/cann'))
    p.add_argument('--soc',default='Ascend950PR_9599')
    p.add_argument('--mode',choices=('sim','npu'),default='sim')
    p.add_argument('--batches',default='2,4')
    p.add_argument('--routes',default='numeric,classify')
    p.add_argument('--score-parts',type=int,default=0,help='Model-only score P; 0 uses the A5 budget map')
    p.add_argument('--output',type=Path)
    p.add_argument('--timeout-seconds',type=int,default=900)
    a=p.parse_args();batches=[int(x) for x in a.batches.split(',')];routes=a.routes.split(',')
    if any(b not in (1,2,4,8,16) for b in batches) or len(set(batches))!=len(batches) or not batches:
        p.error('Use distinct B1/2/4/8/16')
    if any(r not in ('numeric','classify','score') for r in routes) or len(set(routes))!=len(routes) or not routes:
        p.error('Use distinct numeric/classify/score routes')
    if not 0<=a.score_parts<=32 or (a.score_parts and any(b*a.score_parts>32 for b in batches)):
        p.error('score-parts must fit B*P<=32')
    include=a.kvaf_root.resolve()/'access_operator/demo/aiv_batch_get/include'
    if not (include/'kvaf_batch_get_device.h').is_file():p.error('Patched pinned KVAF headers required')
    out=a.output.resolve() if a.output else Path(tempfile.mkdtemp(prefix='aligned_model_'))
    if a.output:out.mkdir(parents=True,exist_ok=False)
    build=out/'build';rows=[]
    manifest=dict(status='RUNNING',scope='full-H radix Top2048 and original classification; no Fetch, no A5 speedup claim',
                  compiled_soc=a.soc,mode=a.mode,batches=batches,routes=routes,results=rows)
    op=HERE.parents[1]/'csrc/operators/lightning_indexer_with_fetch'
    files=[HERE/'aligned_model/CMakeLists.txt',HERE/'aligned_smoke.cpp',op/'op_kernel/indexer.cpp',
           op/'op_kernel/score_window_service.cpp',*sorted((op/'op_kernel/indexer').glob('*.h'))]
    manifest['source_sha256']={str(x.relative_to(HERE.parents[1])):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
    print('Results directory: '+str(out),flush=True)
    try:
        for cmd,name in [(['cmake','-S',str(HERE/'aligned_model'),'-B',str(build),'-DRUN_MODE='+a.mode,
            '-DSOC_VERSION='+a.soc,'-DASCEND_CANN_PACKAGE_PATH='+str(a.cann),'-DKVAF_INCLUDE='+str(include),
            '-DALIGNED_SCORE_MODEL='+('ON' if 'score' in routes else 'OFF')],'configure.log'),
            (['cmake','--build',str(build),'--target','aligned_smoke','--parallel','1'],'build.log')]:
            with (out/name).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest['binary_sha256']=hashlib.sha256((build/'aligned_smoke').read_bytes()).hexdigest()
        simulator=a.cann/'tools/simulator'/a.soc;env=dict(os.environ)
        paths=[str(a.cann/'lib64')]
        if a.mode=='sim':
            paths=[str(simulator/'camodel'),str(simulator/'lib')]+paths
            env['LD_PRELOAD']=str(simulator/'lib/libruntime_camodel.so')
        env['LD_LIBRARY_PATH']=':'.join(paths+[env.get('LD_LIBRARY_PATH','')])
        for b in batches:
            for route in routes:
                name=f'b{b}-{route}';model=out/name;model.mkdir();env['CAMODEL_LOG_PATH']=str(model)
                path=out/(name+'.log');print('Running '+name,flush=True)
                with path.open('w') as log:
                    args=[str(build/'aligned_smoke'),str(b),route]
                    if route=='score' and a.score_parts:args.append(str(a.score_parts))
                    result=subprocess.run(args,env=env,
                        stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout_seconds)
                if result.returncode:raise RuntimeError(f'{name} exit {result.returncode}; inspect {path}')
                contents=path.read_text(errors='replace');row=parse_result(contents)
                row.update(log=str(path),log_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                rows.append(row);print(json.dumps(row),flush=True)
        for b in batches:
            pair=[r for r in rows if r['batch']==b and r['route'] in ('numeric','classify')]
            if len(pair)==2 and any(pair[0][k]!=pair[1][k] for k in ('H','R','selected','hash')):
                raise RuntimeError('Numeric/classify TopK differed on identical scores')
        manifest['status']='PASS'
    except BaseException as error:
        manifest.update(status='FAILED',error=str(error));raise
    finally:(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
