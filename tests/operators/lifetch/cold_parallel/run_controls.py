#!/usr/bin/env python3
"""Print or run interleaved B1/H128K/Q4 full-chain controls on A5.
No SSH or upload. Execution is explicit; each test_lifetch process owns cleanup.
"""
import argparse, hashlib, json, os, random, shlex, subprocess, sys, time
from pathlib import Path
MODES=[('legacy','separate','wave'),('history','separate','wave'),('history','scatter','wave'),
       ('history','direct','wave'),('history','separate','invocation'),('history','direct','invocation')]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def commands(a):
    rng=random.Random(1417)
    for round_no in range(a.rounds+1):
        for windows in a.windows:
            for pf in a.prefetch:
                modes=MODES.copy();rng.shuffle(modes)
                for route,post,prepare in modes:
                    name=f'r{round_no}_W{windows}_pf{pf}_{route}_{post}_{prepare}'
                    cmd=[a.python,str(a.repo/'tests/operators/lifetch/test_lifetch.py'),'bench',
                         '--transport','urma','--device',str(a.device),'--kvaf-root',str(a.kvaf_root),
                         '--batches','1','--history','131072','--resident','16384','--queries','4','--index-heads','32',
                         '--miss-percent','10','--seed','43','--intermediate-fetches',str(windows),
                         '--prefetch-tokens',str(pf),'--warmup',str(a.warmup),'--iterations',str(a.iterations),
                         '--q-workers','2','--aic-budget',str(a.aic_budget),'--score-route',route,
                         '--score-post',post,'--score-prepare',prepare,'--output',str(a.output/name),
                         '--timeout-seconds',str(a.timeout)]
                    if a.aiv_budget is not None:cmd+=['--aiv-budget',str(a.aiv_budget)]
                    if a.kvaf_server:cmd+=['--kvaf-server',a.kvaf_server]
                    if a.valid_history is not None:cmd+=['--valid-history',str(a.valid_history)]
                    yield name,round_no,cmd

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--kvaf-root',type=Path,required=True)
    p.add_argument('--python',default=sys.executable);p.add_argument('--device',type=int,default=5)
    p.add_argument('--kvaf-server');p.add_argument('--output',type=Path,default=Path('cold_parallel_bench'))
    p.add_argument('--windows',type=lambda s:[int(v) for v in s.split(',')],default=[8])
    p.add_argument('--prefetch',type=lambda s:[int(v) for v in s.split(',')],default=[0,512])
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--warmup',type=int,default=4);p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--aic-budget',type=int,default=0);p.add_argument('--aiv-budget',type=int)
    p.add_argument('--valid-history',type=int);p.add_argument('--timeout',type=int,default=300)
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    a.repo=a.repo.resolve();a.kvaf_root=a.kvaf_root.resolve();a.output=a.output.resolve()
    if a.rounds<1 or a.warmup<1 or a.iterations<1 or any(x not in (2,8) for x in a.windows) or any(x<0 for x in a.prefetch):p.error('positive sampling; windows must be 2/8 and prefetch nonnegative')
    cmds=list(commands(a))
    if not a.execute:
        for name,_,cmd in cmds:print(name+': '+shlex.join(cmd))
        return 0
    if a.output.exists():p.error('output already exists; choose a fresh output directory')
    a.output.mkdir(parents=True)
    root=Path(__file__).resolve().parent
    # Verify reviewed source bytes before invoking the existing device harness.
    expected=json.loads((root/'candidate.json').read_text())['files_sha256']
    for path,sha in expected.items():
        if digest(a.repo/path)!=sha:raise RuntimeError('Patched source mismatch: '+path)
    metadata=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip(),
        source_sha256=expected,commands=[dict(name=n,discard=(r==0),argv=c) for n,r,c in cmds],
        note='r0 is discarded; native_chain is SDK/LIDU/KVAF reference, legacy prototype is P2 comparison',
        env={k:os.environ.get(k) for k in ('ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME','LD_LIBRARY_PATH','PYTHONPATH')})
    info=subprocess.run(['npu-smi','info'],text=True,capture_output=True)
    metadata['npu_smi']=dict(exit_code=info.returncode,stdout=info.stdout,stderr=info.stderr)
    (a.output/'manifest.json').write_text(json.dumps(metadata,indent=2))
    for name,round_no,cmd in cmds:
        start=time.time();log=a.output/(name+'.log');print('Running '+name,flush=True)
        with log.open('w') as out:result=subprocess.run(cmd,cwd=a.repo,stdout=out,stderr=subprocess.STDOUT)
        receipt=dict(name=name,discard=(round_no==0),argv=cmd,exit_code=result.returncode,start_unix=start,end_unix=time.time(),log_sha256=digest(log))
        (a.output/(name+'.receipt.json')).write_text(json.dumps(receipt,indent=2))
        if result.returncode:raise RuntimeError(f'{name} failed; stop before other device runs; see {log}')
    print('All runs completed. Discard r0; compare prototype_with_update within matching W/prefetch/resources.')
    return 0
if __name__=='__main__':sys.exit(main())
