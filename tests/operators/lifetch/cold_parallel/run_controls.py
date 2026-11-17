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
                tuning=getattr(a,'matrix','cold')=='tuning'
                modes=[(r,p,d,'separate',0,0,2) for r,p,d in MODES]
                if tuning:
                    modes=[]
                    pairs=getattr(a,'partition_pairs',None) or [(h,c) for h in a.hot_partitions for c in a.cold_partitions]
                    for workers in a.fetch_workers:
                        modes.extend((r,p,d,s,0,0,workers) for r,p,d,s in (
                            ('legacy','separate','wave','separate'),('history','separate','wave','separate'),
                            ('history','direct','wave','separate'),('history','direct','phase','separate'),
                            ('history','direct','phase','inline')))
                        posts=getattr(a,'candidate_posts',['direct'])
                        if 'scatter' in posts:modes.append(('history','scatter','phase','inline',0,0,workers))
                        modes.extend(('history',post,prepare,'inline',hot,cold,workers)
                                     for post in posts for hot,cold in pairs for prepare in a.prepares)
                modes=list(dict.fromkeys(modes));rng.shuffle(modes)
                for route,post,prepare,service,hot,cold,workers in modes:
                    name=f'r{round_no}_W{windows}_pf{pf}_{route}_{post}_{prepare}'
                    if tuning:name+=f'_{service}_Ph{hot}_Pc{cold}_F{workers}'
                    cmd=[a.python,str(a.repo/'tests/operators/lifetch/test_lifetch.py'),'bench',
                         '--transport','urma','--device',str(a.device),'--kvaf-root',str(a.kvaf_root),
                         '--batches','1','--history','131072','--resident','16384','--queries','4','--index-heads','32',
                         '--miss-percent','10','--seed','43','--intermediate-fetches',str(windows),
                         '--prefetch-tokens',str(pf),'--warmup',str(a.warmup),'--iterations',str(a.iterations),
                         '--q-workers',str(workers),'--aic-budget',str(a.aic_budget),'--score-route',route,
                         '--score-post',post,'--score-prepare',prepare,'--output',str(a.output/name),
                         '--timeout-seconds',str(a.timeout)]
                    if tuning:cmd+=['--score-service',service,'--score-hot-partitions',str(hot),'--score-cold-partitions',str(cold)]
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
    p.add_argument('--matrix',choices=('cold','tuning'),default='cold')
    p.add_argument('--hot-partitions',type=lambda s:[int(v) for v in s.split(',')],default=[4,8])
    p.add_argument('--cold-partitions',type=lambda s:[int(v) for v in s.split(',')],default=[4,8,16])
    p.add_argument('--partition-pairs',type=lambda s:[tuple(int(v) for v in x.split(':')) for x in s.split(',')],
                   help='explicit P_hot:P_cold shortlist; overrides the partition Cartesian grid')
    p.add_argument('--prepares',type=lambda s:s.split(','),default=['phase','invocation'])
    p.add_argument('--candidate-posts',type=lambda s:s.split(','),default=['direct'],
                   help='tuning candidates: direct,scatter; retains matched P2/staged controls')
    p.add_argument('--fetch-workers',type=lambda s:[int(v) for v in s.split(',')],default=[2])
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    a.repo=a.repo.resolve();a.kvaf_root=a.kvaf_root.resolve();a.output=a.output.resolve()
    lists=(a.windows,a.prefetch,a.hot_partitions,a.cold_partitions,a.prepares,a.fetch_workers,a.candidate_posts)
    if any(not values or len(set(values))!=len(values) for values in lists):p.error('experiment lists must be nonempty and unique')
    if a.device<0 or a.timeout<=0 or not 0<=a.aic_budget<=32 or (a.aiv_budget is not None and not 2<=a.aiv_budget<=64):
        p.error('invalid device, timeout or hardware budget')
    if a.rounds<1 or a.warmup<1 or a.iterations<1 or any(x not in (2,8) for x in a.windows) or any(x<0 for x in a.prefetch):p.error('positive sampling; windows must be 2/8 and prefetch nonnegative')
    if a.partition_pairs and (any(len(x)!=2 for x in a.partition_pairs) or len(set(a.partition_pairs))!=len(a.partition_pairs)):
        p.error('partition pairs must be unique P_hot:P_cold values')
    partitions=[x for pair in a.partition_pairs for x in pair] if a.partition_pairs else a.hot_partitions+a.cold_partitions
    if (any(not 1<=x<=32 for x in partitions) or
        any(x not in (1,2,4) for x in a.fetch_workers) or any(x not in ('phase','invocation','wave') for x in a.prepares)
        or any(x not in ('direct','scatter') for x in a.candidate_posts)):
        p.error('valid partition, prepare and Fetch lists required')
    if a.matrix=='tuning' and a.aic_budget and max(partitions)>a.aic_budget:
        p.error('requested partition exceeds AIC budget')
    if a.matrix=='tuning' and a.aiv_budget and 2*max(partitions)>a.aiv_budget:
        p.error('requested partition exceeds paired AIV budget')
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
    print('All runs completed. Discard r0; compare prototype_with_update within matching requested windows/prefetch/Fetch/resources; legacy B1 W8 is effective W2.')
    return 0
if __name__=='__main__':sys.exit(main())
