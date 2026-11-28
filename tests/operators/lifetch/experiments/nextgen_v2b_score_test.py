#!/usr/bin/env python3
"""V2b: frozen Score-only binary, real P6 Count/Adaptive queues, same clipping."""
import argparse, ctypes as ct, hashlib, json, os, time
from pathlib import Path
import numpy as np
from b16_score_queue_test import B,Q,C,G,D,TILE,MAGIC,GUARD,SENTINEL,fixture

U=ct.c_uint32
class Packed(ct.Structure):
    _fields_=[("label",U),("mode",U),("n",U),("merge",U),("gate",U),
              ("offsets",U*33),("tasks",(U*4)*256),("visible",U*64),
              ("row_offsets",U*65),("row_ids",U*512),
              ("features",U*9),("loads",U*32),("predicted",ct.c_double),("base",ct.c_double)]
CASES={
    "smoke": (8192,[8192]*8+[4096]*8,[3,4]*8),
    "v05": (196608,[196608]*2+[32768]*14,[4]*16),
    "v07": (524288,[524288]*2+[65536]*14,[4]*16),
    "v08": (524288,[524288]+[32768]*15,[4]*16),
    "v09": (524288,[524288]*8+[131072]*8,[3,4]*8),
    "v10": (524288,[524288]*2+[65536]*14,[3,4,4,4]*4),
}
def geometry(name, planner):
    h,ks,aq=CASES[name]
    # Smoke / MTP cases also test clipped whole tiles and inactive lanes.
    gap=([0,128,129,255]*4 if name in ("smoke","v09","v10") else [0]*16)
    vis=[[k-d-n+q+1 if q<n else 0 for q in range(Q)] for k,d,n in zip(ks,gap,aq)]
    lib=ct.CDLL(str(planner.resolve()))
    fn=lib.mtp_plan
    fn.argtypes=[ct.POINTER(U),ct.POINTER(U),ct.c_int,ct.POINTER(Packed)]
    fn.restype=ct.c_int
    out={}
    for mode,label in enumerate(("count","adaptive")):
        v=Packed()
        start=time.perf_counter_ns()
        assert fn((U*B)(*ks),(U*(B*Q))(*(x for row in vis for x in row)),mode,ct.byref(v))==0
        elapsed=(time.perf_counter_ns()-start)/1000
        assert 32<=v.n<=256 and v.offsets[0]==0 and v.offsets[C]==v.n
        assert list(v.visible)==[x for row in vis for x in row]
        queues=[]
        coverage={(b,g):[] for b in range(B) for g in range(2)}
        for c in range(C):
            lo,hi=v.offsets[c:c+2]
            assert 0<=lo<=hi<=v.n
            queue=[tuple(t) for t in v.tasks[lo:hi]]
            queues.append(queue)
            for b,g,a,z in queue:
                assert b<B and g<2 and 0<=a<z<=ks[b]//TILE
                coverage[b,g].append((a,z))
        for (b,g),spans in coverage.items():
            end=0
            assert 1<=len(spans)<=8
            for a,z in sorted(spans):
                assert a==end
                end=z
            assert end==ks[b]//TILE
        active=[]
        for queue in queues:
            lengths=[max(0,min(z,(max(vis[b][2*g:2*g+2])+TILE-1)//TILE)-a) for b,g,a,z in queue]
            active.append([sum(x>0 for x in lengths),sum(lengths)])
        out[label]=dict(queues=queues,tasks=v.n,selected_label=v.label,selected_mode=v.mode,
                        gate=v.gate,model=v.predicted,planner_one_call_us=elapsed,
                        active=active,max_clipped_tiles=max(x[1] for x in active))
        print("V2B_PLAN",name,label,json.dumps({k:x for k,x in out[label].items() if k not in ("queues","active")}),flush=True)
    return h,ks,aq,vis,out

def device_run(a,h,ks,aq,vis,plans):
    import torch, torch_npu, acl
    physical=int(os.environ["ASCEND_RT_VISIBLE_DEVICES"])
    assert physical in (6,7) and torch.npu.device_count()==1
    libacl=ct.CDLL(str(a.cann/"lib64/libascendcl.so"))
    mapping=libacl.aclrtGetPhyDevIdByLogicDevId
    mapping.argtypes=[ct.c_int32,ct.POINTER(ct.c_int32)]
    mapping.restype=ct.c_int
    mapped=ct.c_int32(-1)
    assert mapping(0,ct.byref(mapped))==0 and mapped.value==physical
    torch.npu.set_device(0)
    props=torch.npu.get_device_properties(0)
    assert props.cube_core_num==32 and props.vector_core_num==64 and acl.get_soc_name()=="Ascend950DT_9582"
    dev=torch.device("npu:0");stream=torch.npu.current_stream(dev)
    inputs,expected=fixture(torch,np,dev,h,ks)
    for b in range(B):
        for g in range(2):
            end=(max(vis[b][2*g:2*g+2])+TILE-1)//TILE*TILE
            expected[b,2*g:2*g+2,end:]=SENTINEL
    lib=ct.CDLL(str(a.library.resolve()))
    fn=lib.v9_b16_score_clip
    fn.argtypes=[ct.c_void_p,ct.c_void_p];fn.restype=ct.c_uint32
    graphs={};states={};samples={}
    def launch(st,s):
        rc=fn(ct.c_void_p(st["binding"].data_ptr()),ct.c_void_p(s.npu_stream))
        if rc:raise RuntimeError(f"launch rc={rc}")
    def reset(st):
        st["out"].fill_(0xa5);st["status"].zero_()
    def verify(st,trace,label):
        raw=st["out"].cpu().numpy()
        assert np.all(raw[:GUARD]==0xa5) and np.all(raw[-GUARD:]==0xa5),"guard overwrite"
        got=raw[GUARD:-GUARD].view(np.uint16).reshape(B,Q,h)
        for b in range(B):
            for q in range(Q):
                bad=np.flatnonzero(got[b,q]!=expected[b,q])
                if bad.size:
                    i=int(bad[0])
                    raise AssertionError(f"{label} score mismatch b={b} q={q} k={i} got={got[b,q,i]:#x} want={expected[b,q,i]:#x} row_errors={len(bad)}")
        assert bool(torch.all(inputs[8]==-1).item()),"dummy topk overwritten"
        stat=st["status"].cpu().numpy()
        if trace:
            for c,(n,t) in enumerate(plans[label]["active"]):
                want=[1 if n else 2,n,t]
                assert stat[2*c,:3].tolist()==want and stat[2*c+1,:3].tolist()==want,(c,stat[2*c:2*c+2,:3],want)
        else:assert not stat.any(),"trace unexpectedly written"
    try:
        for label,plan in plans.items():
            queues=plan["queues"];flat=[t for q in queues for t in q]
            offs=torch.tensor(np.cumsum([0]+[len(q) for q in queues],dtype=np.int32),device=dev)
            tasks=torch.tensor(flat,dtype=torch.int32,device=dev)
            ctrl=torch.tensor([MAGIC,2,h,len(flat),1],dtype=torch.int32,device=dev)
            status=torch.zeros((64,16),dtype=torch.int32,device=dev)
            visible=torch.tensor(vis,dtype=torch.int32,device=dev)
            out=torch.empty(B*Q*h*2+2*GUARD,dtype=torch.uint8,device=dev)
            ptrs=[x.data_ptr() for x in inputs]+[out.data_ptr()+GUARD,offs.data_ptr(),tasks.data_ptr(),ctrl.data_ptr(),status.data_ptr(),visible.data_ptr()]
            assert len(ptrs)==15
            binding=torch.tensor(ptrs,dtype=torch.uint64,device=dev)
            st=dict(out=out,status=status,ctrl=ctrl,binding=binding,keep=(offs,tasks,visible))
            states[label]=st
            reset(st);stream.synchronize()
            print("V2B_EAGER_BEGIN",a.case,label,flush=True)
            launch(st,stream);stream.synchronize();verify(st,True,label)
            print("V2B_EAGER_SCORE_PASS",a.case,label,flush=True)
            ctrl[4]=0;reset(st);stream.synchronize()
            cap=torch.npu.Stream(device=dev);graph=torch.npu.NPUGraph();cap.wait_stream(stream)
            with torch.npu.graph(graph,stream=cap):launch(st,cap)
            cap.synchronize();graphs[label]=(graph,cap)
            for _ in range(3):
                reset(st);stream.synchronize();graph.replay();stream.synchronize();verify(st,False,label)
            print("V2B_GRAPH_SCORE_PASS",a.case,label,flush=True)
            samples[label]=[]
        labels=list(states)
        for i in range(60):
            for label in (labels if i%2==0 else labels[::-1]):
                graph,_=graphs[label]
                start,end=[torch.npu.Event(enable_timing=True) for _ in range(2)]
                start.record(stream);graph.replay();end.record(stream);end.synchronize()
                if i>=10:samples[label].append(start.elapsed_time(end)*1000)
        for label,st in states.items():verify(st,False,label)
    finally:
        for graph,cap in graphs.values():graph.reset()
    result=dict(status="PASS",case=a.case,physical_npu=physical,history=h,actual_k=ks,actual_q=aq,
                visibility=vis,scope="Score-only; both queues use identical Q2 clipping, no TopK or Fetch; planner outside device timing",
                library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),policies={})
    for label,v in samples.items():
        assert len(v)==50 and np.isfinite(v).all() and min(v)>0
        result["policies"][label]=dict(**plans[label],raw_us=v,p50_us=float(np.median(v)),p95_us=float(np.percentile(v,95)),p99_us=float(np.percentile(v,99)),eager_exact=True,graph_exact_replays=3)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print("V2B_CASE_PASS",a.case,json.dumps({k:v["p50_us"] for k,v in result["policies"].items()}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("host","device"),required=True)
    p.add_argument("--planner",type=Path,required=True)
    p.add_argument("--library",type=Path)
    p.add_argument("--cann",type=Path)
    p.add_argument("--case",choices=CASES)
    p.add_argument("--output",type=Path)
    a=p.parse_args();assert __debug__
    if a.mode=="host":
        for name in CASES:geometry(name,a.planner)
        print("V2B_HOST_PLAN_PASS",flush=True)
    else:
        assert a.case and a.library and a.output and a.cann
        device_run(a,*geometry(a.case,a.planner))
if __name__=="__main__":main()
