#!/usr/bin/env python3
"""SDK standalone LI reference: exactly one public LI call, no LIDU/KVAF/Fetch.

The SDK chooses its own tiling; this is not the T1 single-group reference.
G stays 32. Unsupported SDK geometry fails rather than padding to G64.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def native_li_only(torch_npu, x):
    result=torch_npu.npu_quant_lightning_indexer(
        query=x['query'],key=x['key'],weights=x['weights'],
        query_dequant_scale=x['qscale'],key_dequant_scale=x['kscale'],
        actual_seq_lengths_query=x['qlen'],actual_seq_lengths_key=x['klen'],block_table=x['pages'],
        query_quant_mode=0,key_quant_mode=0,layout_query='BSND',layout_key='PA_BSND',
        sparse_count=2048,sparse_mode=0)
    return result[0] if isinstance(result,(tuple,list)) else result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fixture',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--device',type=int,default=5);p.add_argument('--warmup',type=int,default=4);p.add_argument('--iterations',type=int,default=20)
    a=p.parse_args()
    if a.output.exists() or a.device<0 or a.warmup<1 or a.iterations<1:p.error('fresh output and valid sampling/device required')
    import numpy as np
    import torch
    import torch_npu
    meta=json.loads((a.fixture/'fixture.json').read_text());h=meta['history']
    if (meta['batch'],meta['heads'])!=(1,32):raise ValueError('SDK control requires same B1/G32 fixture')
    for name,expected in meta['hashes'].items():
        if hashlib.sha256((a.fixture/(name+'.bin')).read_bytes()).hexdigest()!=expected:raise ValueError('fixture hash changed')
    torch_npu.npu.set_device(a.device);device=f'npu:{a.device}'
    def raw(name,dtype,shape):return torch.from_numpy(np.fromfile(a.fixture/(name+'.bin'),dtype).reshape(shape))
    x=dict(query=raw('query',np.uint8,(1,4,32,128)).view(torch.float8_e4m3fn).to(device),
           key=raw('key',np.uint8,(h//128,128,1,128)).view(torch.float8_e4m3fn).to(device),
           weights=raw('weight',np.uint16,(1,4,32)).view(torch.bfloat16).to(device),
           qscale=raw('qscale',np.float32,(1,4,32)).to(device),
           kscale=raw('kscale',np.float32,(h//128,128,1)).to(device),
           qlen=torch.tensor([4],dtype=torch.int32,device=device),
           klen=torch.tensor([h],dtype=torch.int32,device=device),
           pages=torch.arange(h//128,dtype=torch.int32,device=device).reshape(1,-1))
    for _ in range(a.warmup):selected=native_li_only(torch_npu,x)
    torch_npu.npu.synchronize();samples=[]
    start=torch_npu.npu.Event(enable_timing=True);end=torch_npu.npu.Event(enable_timing=True)
    for _ in range(a.iterations):
        start.record();selected=native_li_only(torch_npu,x);end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000)
    indices=selected.cpu().numpy().reshape(4,2048);oracle=np.fromfile(a.fixture/'oracle.bin',np.uint16).reshape(4,h)
    for q in range(4):
        got=indices[q].astype(np.int64)
        if np.any(got<0) or np.any(got>=h) or len(np.unique(got))!=2048:raise ValueError('SDK TopK bounds/duplicate')
        theta=np.partition(oracle[q],h-2048)[h-2048]
        if np.any(oracle[q,got]<theta) or not set(np.flatnonzero(oracle[q]>theta))<=set(got):raise ValueError('SDK TopK oracle mismatch')
    a.output.mkdir(parents=True);indices.astype(np.int32).tofile(a.output/'topk.bin')
    proof=dict(status='PASS',run_mode='npu',batch=1,queries=4,heads=32,history=h,k=2048,warmup=a.warmup,
               iterations=a.iterations,sdk_li_p50_us=statistics.median(samples),sdk_li_samples_us=samples,
               fetch_launches=0,scope='SDK standalone LI score+TopK; SDK owns tiling; no LIDU/KVAF',
               fixture_sha256=meta['hashes'],topk_threshold_oracle=True,
               torch_version=torch.__version__,torch_npu_version=getattr(torch_npu,'__version__','unknown'))
    (a.output/'report.json').write_text(json.dumps(proof,indent=2)+'\n');print('SDK_LI_ORACLE_PASS')


if __name__=='__main__':main()
