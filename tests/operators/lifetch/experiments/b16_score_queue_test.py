#!/usr/bin/env python3
"""LB-P1 isolated B16 Q4 / Q2 group Score queue, CPU exact positive dyadic fixture.
Host preflight is independent of torch; no production runtime/ABI or KVAF service.
First run smoke cases before 512K. Score-only timing is NOT Native QLI TopK timing.
"""
import argparse, ctypes, fcntl, hashlib, json, math, os, re, statistics, subprocess, sys, time
from pathlib import Path

B,Q,G,D,C,TILE=16,4,32,128,32,128
MAGIC=0x42313631
SENTINEL=0xA5A5
GUARD=4096
CASES={
    "smoke":(8192,[4096]*16),
    "mixed-small":(8192,[8192]*8+[4096]*8),
    "odd-shard":(8192,[8064]*8+[3968]*8),
    "18":(524288,[524288]*16),
    "19":(524288,[524288]*8+[131072]*8),
    "21":(524288,[524288]*2+[65536]*14),
    "22":(524288,[524288]+[32768]*15),
    "25":(524288,[524288]*8+[491520]*8),
}

def lpt(tasks):
    q=[[] for _ in range(C)]
    load=[0]*C
    for t in sorted(tasks,key=lambda t:(-(t[3]-t[2]),t)):
        i=min(range(C),key=lambda i:(load[i],i))
        q[i].append(t)
        load[i]+=t[3]-t[2]
    return q

def parts(b,g,n,k):
    return [(b,g,k*j//n,k*(j+1)//n) for j in range(n)]

def plans(name):
    h,ks=CASES[name]
    original=[(b,g,0,k//TILE) for b,k in enumerate(ks) for g in range(2)]
    base=[[t] for t in original]
    result={"count":base}
    if name in ("mixed-small","odd-shard","19"):
        # Equal: split both long and short groups in two; nonuniform:
        # keep each short group whole and fill the paired long group.
        result["equal"]=lpt([t for b,g,lo,hi in original for t in parts(b,g,2,hi)])
        result["fill"]=[[] for _ in range(C)]
        for i in range(16):
            b,g=divmod(i,2)
            long=ks[b]//TILE
            short=ks[b+8]//TILE
            target=(long+short+1)//2
            result["fill"][i]=[(b,g,0,target)]
            result["fill"][i+16]=[(b+8,g,0,short),(b,g,target,long)]
    elif name in ("21","22"):
        n=4 if name=="21" else 8
        result["low_cost"]=lpt([t for b,g,lo,hi in original for t in parts(b,g,n if ks[b]==h else 1,hi)])
        # high-fragment policy deliberately retained as a cost contrast
        if name=="21":
            result["dense_split"]=lpt([t for b,g,lo,hi in original for t in parts(b,g,8,hi)])
    else:
        result["adaptive_noop"]=[list(x) for x in base]
    for label,queues in result.items():
        assert len(queues)==C and all(queues)
        flat=[t for q in queues for t in q]
        assert len(flat)<=256
        seen={(b,g):[] for b in range(B) for g in range(2)}
        for b,g,lo,hi in flat:
            assert (b,g) in seen and 0<=lo<hi<=ks[b]//TILE<=h//TILE
            seen[b,g].append((lo,hi))
        for (b,g),ranges in seen.items():
            ranges.sort()
            assert ranges and ranges[0][0]==0 and ranges[-1][1]==ks[b]//TILE
            assert all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
        assert sum(t[3]-t[2] for t in flat)==2*sum(ks)//TILE
        print("PLAN",name,label,"tasks",len(flat),"max_tiles",max(sum(t[3]-t[2] for t in q) for q in queues),flush=True)
    return result

def fixture(torch,np,device,h,ks):
    # Q is one-hot per query, every one of 32 heads identical. K, scales and
    # weights are positive exact dyadic values; no numerical tolerance needed.
    # Use per-request independent patterns to expose stale cross-request data.
    rng=np.random.default_rng(20261125)
    rawq=np.zeros((B,Q,G,D),np.uint8)
    for q in range(Q): rawq[:,q,:,q]=0x38
    keys=torch.empty((B,h,D),dtype=torch.uint8,device=device)
    bands=np.array([0x20,0x28,0x30,0x38,0x40,0x48],np.uint8)
    values=np.array([.125,.25,.5,1.,2.,4.],np.float32)
    ws=np.ones((B,Q,G),np.float32)
    qs=np.ones_like(ws)
    scales=np.empty((B,h),np.float32)
    expected=np.full((B,Q,h),SENTINEL,np.uint16)
    for b in range(B):
        ids=rng.integers(0,len(bands),(h,Q),dtype=np.int32)
        raw=np.zeros((h,D),np.uint8)
        raw[:,:Q]=bands[ids]
        keys[b].copy_(torch.from_numpy(raw))
        scales[b]=np.where((np.arange(h)+b)%3==0,2.,1.)
        for q in range(Q):
            ws[b,q,:]=2.**(b%3)
            qs[b,q,:]=2.**q
            v=(values[ids[:ks[b],q]]*np.float32(G*2.**(b%3+q))*scales[b,:ks[b]]).astype(np.float32)
            expected[b,q,:ks[b]]=(v.view(np.uint32)>>16).astype(np.uint16)^np.uint16(0x8000)
    def dev(a): return torch.from_numpy(a).to(device)
    tensors=[dev(rawq),keys.reshape(B*h//TILE,TILE,1,D),
        dev(ws).to(torch.bfloat16),dev(qs),dev(scales.reshape(-1,TILE,1)),
        dev(np.full(B,4,np.int32)),dev(np.array(ks,np.int32)),
        dev(np.arange(B*h//TILE,dtype=np.int32).reshape(B,-1)),
        torch.full((B,Q,2048),-1,dtype=torch.int32,device=device)]
    return tensors,expected

def run_case(a,name):
    import numpy as np
    import torch
    import torch_npu
    assert torch.npu.device_count()==1
    torch.npu.set_device(0)
    device=torch.device("npu:0")
    props=torch.npu.get_device_properties(0)
    print("DEVICE",props,flush=True)
    assert props.cube_core_num==32 and props.vector_core_num==64
    stream=torch.npu.current_stream(device)
    h,ks=CASES[name]
    ps=plans(name)
    inputs,expected=fixture(torch,np,device,h,ks)
    lib=ctypes.CDLL(str(a.library.resolve()))
    fn=lib.v9_b16_score_queue
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
    fn.restype=ctypes.c_uint32
    states={}
    for label,queues in ps.items():
        flat=[t for q in queues for t in q]
        offsets=np.cumsum([0]+[len(q) for q in queues],dtype=np.int32)
        out=torch.empty(B*Q*h*2+2*GUARD,dtype=torch.uint8,device=device)
        offs=torch.from_numpy(offsets).to(device)
        tasks=torch.tensor(flat,dtype=torch.int32,device=device)
        ctrl=torch.tensor([MAGIC,1,h,len(flat),1],dtype=torch.int32,device=device)
        status=torch.zeros((64,16),dtype=torch.int32,device=device)
        ptrs=[x.data_ptr() for x in inputs]+[out.data_ptr()+GUARD,offs.data_ptr(),tasks.data_ptr(),ctrl.data_ptr(),status.data_ptr()]
        binding=torch.tensor(ptrs,dtype=torch.uint64,device=device)
        states[label]=dict(out=out,offs=offs,tasks=tasks,ctrl=ctrl,status=status,binding=binding,queues=queues)
    def launch(st,s):
        rc=fn(ctypes.c_void_p(st["binding"].data_ptr()),ctypes.c_void_p(s.npu_stream))
        if rc: raise RuntimeError("launch rc="+str(rc))
    def prepare(st):
        st["out"].fill_(0xA5)
        st["status"].zero_()
    def verify(st,trace):
        raw=st["out"].cpu().numpy()
        assert np.all(raw[:GUARD]==0xA5) and np.all(raw[-GUARD:]==0xA5),"guard overwrite"
        got=raw[GUARD:-GUARD].view(np.uint16).reshape(B,Q,h)
        bad=np.flatnonzero(got.reshape(-1)!=expected.reshape(-1))
        if bad.size:
            pos=np.unravel_index(int(bad[0]),got.shape)
            raise AssertionError(f"Score mismatch count={bad.size} at {pos}: actual={int(got[pos]):#x}, expected={int(expected[pos]):#x}")
        assert bool(torch.all(inputs[8]==-1).item()),"dummy TopK modified"
        if trace:
            stat=st["status"].cpu().numpy()
            for i,q in enumerate(st["queues"]):
                exp=[1,len(q),sum(t[3]-t[2] for t in q)]
                assert list(stat[2*i,:3])==exp and list(stat[2*i+1,:3])==exp,(i,stat[2*i:2*i+2,:3],exp)
        else:
            assert bool(torch.all(st["status"]==0).item()),"trace not disabled"
    graphs={}
    results={}
    try:
        for label,st in states.items():
            prepare(st);stream.synchronize()
            print("EAGER_BEGIN",name,label,flush=True)
            launch(st,stream);stream.synchronize();verify(st,True)
            print("EAGER_EXACT_SCORE_PASS",name,label,flush=True)
            # Untimed diagnostic writes disabled for all perf policies.
            st["ctrl"][4]=0
            prepare(st);stream.synchronize()
            cap=torch.npu.Stream(device=device)
            graph=torch.npu.NPUGraph()
            cap.wait_stream(stream)
            with torch.npu.graph(graph,stream=cap): launch(st,cap)
            cap.synchronize()
            graphs[label]=(graph,cap)
            for i in range(3):
                prepare(st)
                stream.synchronize()
                graph.replay();stream.synchronize();verify(st,False)
            print("GRAPH_EXACT_SCORE_PASS",name,label,flush=True)
            results[label]=[]
        # Equal sample counts; rotate policy order per repeat and sample.
        labels=list(states)
        for r in range(a.repeats):
            for i in range(a.warmup+a.samples):
                for j in range(len(labels)):
                    label=labels[(r+i+j)%len(labels)]
                    graph,_=graphs[label]
                    t=time.perf_counter_ns()
                    graph.replay();stream.synchronize()
                    us=(time.perf_counter_ns()-t)/1000
                    if i>=a.warmup: results[label].append(us)
            for label,st in states.items(): verify(st,False)
    finally:
        for graph,cap in graphs.values(): graph.reset()
    def stats(v):
        v=sorted(v)
        return dict(p50_us=statistics.median(v),p95_us=v[math.ceil(.95*len(v))-1],p99_us=v[math.ceil(.99*len(v))-1])
    report=dict(status="PASS",case=name,history=h,ks=ks,exact_score=True,
        scope="B16 Q4 shared actual K, Score-only, graph replay + synchronize; no TopK",
        samples={k:dict(stats(v),raw_us=v) for k,v in results.items()},
        queues=ps,library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest())
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print("B16_SCORE_QUEUE_PASS",name,json.dumps({k:stats(v) for k,v in results.items()}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("host","device"),default="host")
    p.add_argument("--case",choices=tuple(CASES))
    p.add_argument("--library",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--samples",type=int,default=100)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--warmup",type=int,default=5)
    a=p.parse_args()
    assert not sys.flags.optimize
    if a.mode=="host":
        for name in CASES: plans(name)
        print("B16_SCORE_QUEUE_HOST_PASS",flush=True)
        return
    assert a.case and a.library and a.output and a.library.is_file()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") in ("6","7")
    assert os.environ.get("DEVICE_ID")=="0"
    assert a.samples>=20 and a.repeats>=1 and a.warmup>=0
    run_case(a,a.case)

if __name__=="__main__": main()
