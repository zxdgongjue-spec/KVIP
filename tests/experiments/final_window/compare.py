#!/usr/bin/env python3
"""Compare the original tail against a bounded final-window reuse experiment."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from prepare import prepare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--kvaf-root',required=True,type=Path)
    parser.add_argument('--kvaf-server',default='127.0.0.1:28885')
    parser.add_argument('--device',type=int,default=4)
    parser.add_argument('--iterations',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--windows',type=int,choices=(4,8),default=4)
    parser.add_argument('--prefetch-tokens',type=int,default=1024)
    parser.add_argument('--prefetch-rank-start',type=int,default=1024)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[3]
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.snapshot/'manifest.json').read_text())
    groups=manifest.get('groups',[])
    if len(groups)!=1 or groups[0]['shape']['splitQ']!=1 or groups[0]['shape']['history']>131328:
        raise ValueError('Experiment requires one sequence-split group with history <= 131328')
    if subprocess.run(['git','cat-file','-e','c239791^{commit}'],cwd=repo,
                      stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        subprocess.run(['git','fetch','origin','codex/lifetch-token-get-bounds'],cwd=repo,check=True)
    # Keep the large temporary snapshot utility out of the production PR.
    utility=subprocess.check_output(['git','show','c239791:scripts/lifetch_snapshot_profile.py'],cwd=repo,text=True)
    marker='        selected = topk[valid].contiguous().numpy().tobytes()'
    addition='''        if measured and not any(status) and not int(bad.sum()):
            detail = forward.query_errors.cpu().reshape(shape["batch"],4,32)
            write_json(output / "terminal_counts.json", dict(
                eligible_rows=int(detail[valid.any(dim=1),0,2].sum()),
                active_rows=int(valid.any(dim=1).sum()),
                compact_misses=int(detail[:,0,3].sum()),
                terminal_posted=int(detail[:,0,4].sum()),
                per_row=detail[:,0,2:5].tolist(),
                active_details=[dict(row=i, eligible=int(detail[i,0,2]),
                    misses_per_q=detail[i,0,8:12].tolist(),
                    posted=int(detail[i,0,4]), reason_bits=int(detail[i,0,6]),
                    used_lru=int(detail[i,0,7]))
                    for i in range(shape["batch"]) if bool(valid[i].any())]))
            if iteration == args.warmup:
                logical = update.logical_output.cpu().reshape(shape["batch"],4,2048)[valid]
                slots = update.slot_output.cpu().reshape(shape["batch"],4,2048)[valid].long()
                if bool(((slots < 0) | (slots >= snap.meta["output_stride"])).any()):
                    raise RuntimeError("Selected output slots outside owner allocation")
                owners = inputs.row_owners.cpu().long()[:,None].expand(-1,4)[valid]
                payload = hbm.cpu()[owners[:,None]*snap.meta["output_stride"]+slots]
                order = logical.argsort(dim=1)
                torch.save(dict(logical=logical.gather(1,order),
                    kv=payload.gather(1,order[:,:,None].expand(-1,-1,payload.shape[-1]))),output / "selected_kv.pt")
'''
    if utility.count(marker)!=1:raise ValueError('Snapshot utility layout changed')
    runner=args.output/'snapshot_runner.py';runner.write_text(utility.replace(marker,addition+marker))
    candidate=args.output/'candidate-source'
    prepare(repo,candidate)
    print('A: original final TopK + tail. B: one kernel, bounded terminal prefetch + original completion/fallback.',flush=True)
    results={}
    for name,root in (('A',repo),('B',candidate)):
        output=args.output/name
        command=[sys.executable,str(runner),str(args.snapshot.resolve()),'--output',str(output),
                 '--kvip-root',str(root),'--kvaf-root',str(args.kvaf_root.resolve()),
                 '--kvaf-server',args.kvaf_server,'--device',str(args.device),
                 '--windows',str(args.windows),'--prefetch-tokens',str(args.prefetch_tokens),
                 '--prefetch-rank-start',str(args.prefetch_rank_start),
                 '--warmup',str(args.warmup),'--iterations',str(args.iterations),
                 '--profile-only','--timeout-seconds','900']
        code=subprocess.run(command).returncode
        if code:raise SystemExit(f'{name} failed: {output}; no performance acceptance')
        folder=output/'graph-captured'
        values=json.loads((folder/'kernel_times.json').read_text())
        means={key:sum(v)/len(v) for key,v in values.items() if v}
        final=sum(v for k,v in means.items() if re.fullmatch(r'v9_li_kernel_3(?:_\d+)?',k))
        tail=sum(v for k,v in means.items() if k.startswith('v9_streaming_resolve_update'))
        results[name]=dict(final_topk_us=final,tail_us=tail,combined_us=final+tail,total_us=sum(means.values()),
                           counts=json.loads((folder/'terminal_counts.json').read_text()))
    import torch
    a=torch.load(args.output/'A/graph-captured/selected_kv.pt',weights_only=True)
    b=torch.load(args.output/'B/graph-captured/selected_kv.pt',weights_only=True)
    checks={key:torch.equal(a[key],b[key]) for key in ('logical','kv')}
    results['correctness']=checks
    (args.output/'ab_summary.json').write_text(json.dumps(results,indent=2))
    print('Selected logical TopK / KV equality:',checks)
    print('Path | Final TopK | Tail (B includes final TopK) | Combined us | All kernels us')
    for name in ('A','B'):
        r=results[name]
        print(f"{name} | {r['final_topk_us']:.2f} | {r['tail_us']:.2f} | {r['combined_us']:.2f} | {r['total_us']:.2f}")
    print('B terminal path:',results['B']['counts'])
    print(f"Summary: {args.output/'ab_summary.json'}")
    if not all(checks.values()):raise SystemExit('AB correctness FAIL; timings are not accepted')


if __name__=='__main__':
    main()
