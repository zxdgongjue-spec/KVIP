#!/usr/bin/env python3
"""V3b: exact Score, task-owned candidates, query-owned global TopK; frozen binaries."""
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
    "smoke_q2": (8192,[8192]*8+[4096]*8,[2]*4+[4]*12),
    "v05": (196608,[196608]*2+[32768]*14,[4]*16),
    "v07": (524288,[524288]*2+[65536]*14,[4]*16),
    "v08": (524288,[524288]+[32768]*15,[4]*16),
    "v09": (524288,[524288]*8+[131072]*8,[3,4]*8),
    "v10": (524288,[524288]*2+[65536]*14,[3,4,4,4]*4),
}
def geometry(name, planner):
    h,ks,aq=CASES[name]
    # Smoke / MTP cases also test clipped whole tiles and inactive lanes.
    gap=([0,128,129,255]*4 if name in ("smoke","smoke_q2","v09","v10") else [0]*16)
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
        # Verify the packed planner row ABI, not just task coverage.
        ro,ri=row_table([t for q in queues for t in q],ks)
        assert ro==list(v.row_offsets) and ri==list(v.row_ids)[:2*v.n]
        if label=="count":assert ri==list(range(B*Q)) and v.n==32
        active=[]
        for queue in queues:
            lengths=[max(0,min(z,(max(vis[b][2*g:2*g+2])+TILE-1)//TILE)-a) for b,g,a,z in queue]
            active.append([sum(x>0 for x in lengths),sum(lengths)])
        out[label]=dict(queues=queues,tasks=v.n,selected_label=v.label,selected_mode=v.mode,
                        gate=v.gate,model=v.predicted,planner_one_call_us=elapsed,
                        active=active,max_clipped_tiles=max(x[1] for x in active))
        print("V3B_PLAN",name,label,json.dumps({k:x for k,x in out[label].items() if k not in ("queues","active")}),flush=True)
    return h,ks,aq,vis,out

K=2048
LOCAL_MAGIC=0x42313632
GLOBAL_MAGIC=0x42313633

def reference(row,lo,hi):
    # Historical P6 cutoff contract: all strictly greater scores first,
    # then earliest physical positions among ties at the cutoff.
    if hi<=lo:return np.empty(0,dtype=np.int64)
    v=row[lo:hi]
    if len(v)<K:return np.arange(lo,hi,dtype=np.int64)
    cut=np.partition(v,len(v)-K)[len(v)-K]
    greater=np.flatnonzero(v>cut)
    equal=np.flatnonzero(v==cut)
    return lo+np.concatenate((greater,equal[:K-len(greater)]))

def row_table(flat,ks):
    rows=[[] for _ in range(B*Q)]
    for i,(b,g,lo,hi) in enumerate(flat):
        for lane in range(2):rows[b*Q+2*g+lane].append((lo,hi,2*i+lane))
    ro=[0];ri=[]
    for row,rs in enumerate(rows):
        rs.sort()
        assert 1<=len(rs)<=8 and rs[0][0]==0 and rs[-1][1]*TILE==ks[row//Q]
        assert all(x[1]==y[0] for x,y in zip(rs,rs[1:]))
        ri.extend(x[2] for x in rs);ro.append(len(ri))
    assert sorted(ri)==list(range(2*len(flat)))
    return ro,ri

def host_oracle():
    # Short, empty, tie, adversarial forbidden suffix, nonaligned boundary.
    row=np.full(8192,0x8100,dtype=np.uint16)
    row[100:400]=0x8200;row[8065:]=0xff00
    assert reference(row,0,0).size==0
    assert np.array_equal(reference(row,128,129),[128])
    want=reference(row,0,8065)
    assert len(want)==K and np.max(want)<8065
    assert np.any(reference(row,0,8192)>=8065)
    for parts in ([(0,8192)],[(0,1024),(1024,4096),(4096,8192)],[(0,4096),(4096,8064),(8064,8192)]):
        ids=np.concatenate([reference(row,lo,min(hi,8065)) for lo,hi in parts])
        # Original position ascending for deterministic cutoff ties.
        order=np.lexsort((ids,-row[ids].astype(np.int32)))
        assert np.array_equal(np.sort(ids[order[:K]]),np.sort(want))
    print("V3B_CPU_ORACLE_PASS",flush=True)

def device_run(a,h,ks,aq,vis,plans):
    import torch, torch_npu, acl
    physical=int(os.environ["ASCEND_RT_VISIBLE_DEVICES"])
    assert physical in (0,6,7) and torch.npu.device_count()==1
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
    refs=[reference(expected[b,q],0,vis[b][q]) for b in range(B) for q in range(Q)]
    libs=[];fns=[]
    for path,sym in ((a.library,"v9_b16_score_clip"),(a.local,"v9_b16_local_topk"),(a.merge,"v9_b16_global_topk")):
        lib=ct.CDLL(str(path.resolve()));fn=getattr(lib,sym)
        fn.argtypes=[ct.c_void_p,ct.c_void_p];fn.restype=ct.c_uint32
        libs.append(lib);fns.append(fn)
    graphs={};states={};samples={};order_exact={}
    def launch(st,s,ops):
        for i in ops:
            rc=fns[i](ct.c_void_p(st["binds"][i].data_ptr()),ct.c_void_p(s.npu_stream))
            if rc:raise RuntimeError(f"stage={i} launch rc={rc}")
    def reset(st):
        for key in ("out","cs","ci","final"):st[key].fill_(0xa5)
        st["status"].zero_()
    def guarded(st,key,dtype,shape):
        raw=st[key].cpu().numpy()
        assert np.all(raw[:GUARD]==0xa5) and np.all(raw[-GUARD:]==0xa5),f"{key} guard overwrite"
        return raw[GUARD:-GUARD].view(dtype).reshape(shape)
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
    def verify_topk(st,label,merged=True):
        flat=st["flat"];n=len(flat)
        cs=guarded(st,"cs",np.uint16,(n,2,K))
        ci=guarded(st,"ci",np.uint32,(n,2,K))
        local_order=True
        for i,(b,g,lo,hi) in enumerate(flat):
            for lane in range(2):
                q=2*g+lane
                want=reference(expected[b,q],lo*TILE,min(hi*TILE,vis[b][q]))
                count=len(want);got=ci[i,lane,:count].astype(np.int64)
                assert np.array_equal(np.sort(got),np.sort(want)),f"{label} local membership task={i} lane={lane}"
                local_order &= bool(np.array_equal(got,want))
                assert np.array_equal(cs[i,lane,:count],expected[b,q,got]),"candidate pair mismatch"
                assert np.all(cs[i,lane,count:]==0xffff) and np.all(ci[i,lane,count:]==0xffffffff),"candidate padding"
        if merged:
            final=guarded(st,"final",np.uint32,(B*Q,K))
        else:
            assert label=="count" and n==32
            final=ci.reshape(B*Q,K)
        final_order=True
        for row,want in enumerate(refs):
            if not len(want):
                assert np.all(final[row]==0xffffffff),f"invalid query row {row}"
            else:
                assert len(want)==K
                got=final[row].astype(np.int64)
                assert np.array_equal(np.sort(got),np.sort(want)),f"{label} final membership row={row}"
                final_order &= bool(np.array_equal(got,want))
        key=label+("_merge" if merged else "_direct")
        pair=dict(local_order=local_order,final_order=final_order)
        prev=order_exact.get(key,dict(local_order=True,final_order=True))
        order_exact[key]={k:prev[k] and pair[k] for k in pair}
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
            n=len(flat)
            def buffer(size):return torch.empty(size+2*GUARD,dtype=torch.uint8,device=dev)
            cs,ci,final=buffer(n*2*K*2),buffer(n*2*K*4),buffer(B*Q*K*4)
            lc=torch.tensor([LOCAL_MAGIC,2,h,n],dtype=torch.int32,device=dev)
            gc=torch.tensor([GLOBAL_MAGIC,2,h,n],dtype=torch.int32,device=dev)
            ro,ri=row_table(flat,ks)
            row_off=torch.tensor(ro,dtype=torch.int32,device=dev)
            row_ids=torch.tensor(ri,dtype=torch.int32,device=dev)
            lb=torch.tensor([out.data_ptr()+GUARD,offs.data_ptr(),tasks.data_ptr(),lc.data_ptr(),cs.data_ptr()+GUARD,ci.data_ptr()+GUARD,inputs[6].data_ptr(),visible.data_ptr()],dtype=torch.uint64,device=dev)
            gb=torch.tensor([cs.data_ptr()+GUARD,ci.data_ptr()+GUARD,tasks.data_ptr(),row_off.data_ptr(),row_ids.data_ptr(),gc.data_ptr(),inputs[6].data_ptr(),final.data_ptr()+GUARD,visible.data_ptr()],dtype=torch.uint64,device=dev)
            st=dict(out=out,status=status,ctrl=ctrl,cs=cs,ci=ci,final=final,flat=flat,
                    binds=(binding,lb,gb),keep=(offs,tasks,visible,lc,gc,row_off,row_ids))
            states[label]=st
            reset(st);stream.synchronize()
            print("V3B_SCORE_BEGIN",a.case,label,flush=True)
            launch(st,stream,(0,));stream.synchronize();verify(st,True,label)
            print("V3B_SCORE_PASS",a.case,label,flush=True)
            launch(st,stream,(1,));stream.synchronize()
            if label=="count":verify_topk(st,label,False)
            launch(st,stream,(2,));stream.synchronize();verify_topk(st,label)
            print("V3B_EAGER_TOPK_PASS",a.case,label,flush=True)
            ctrl[4]=0;stream.synchronize()
            scopes={"score":(0,),"three_stage":(0,1,2)}
            if label=="count":scopes["direct"]=(0,1)
            for scope,ops in scopes.items():
                reset(st);stream.synchronize()
                cap=torch.npu.Stream(device=dev);graph=torch.npu.NPUGraph();cap.wait_stream(stream)
                with torch.npu.graph(graph,stream=cap):launch(st,cap,ops)
                cap.synchronize();graphs[(label,scope)]=(graph,cap)
                for _ in range(3):
                    reset(st);stream.synchronize();graph.replay();stream.synchronize();verify(st,False,label)
                    if scope!="score":verify_topk(st,label,scope=="three_stage")
                samples[(label,scope)]=[]
                print("V3B_GRAPH_PASS",a.case,label,scope,flush=True)
        # Equal samples, rotating policy order. No diagnostic reads inside timed region.
        keys=list(graphs)
        for i in range(60):
            order=keys[i%len(keys):]+keys[:i%len(keys)]
            if i%2:order.reverse()
            for key in order:
                graph,_=graphs[key]
                start,end=[torch.npu.Event(enable_timing=True) for _ in range(2)]
                start.record(stream);graph.replay();end.record(stream);end.synchronize()
                if i>=10:samples[key].append(start.elapsed_time(end)*1000)
        for (label,scope),(graph,cap) in graphs.items():
            st=states[label];reset(st);stream.synchronize();graph.replay();stream.synchronize()
            verify(st,False,label)
            if scope!="score":verify_topk(st,label,scope=="three_stage")
    finally:
        for graph,cap in graphs.values():graph.reset()
    result=dict(status="PASS",case=a.case,physical_npu=physical,history=h,actual_k=ks,actual_q=aq,
                visibility=vis,scope="Score/Local/Global only; same clipping; Count direct separately; planner/upload/Fetch excluded",
                library_sha256={k:hashlib.sha256(path.read_bytes()).hexdigest() for k,path in (("score",a.library),("local",a.local),("global",a.merge))},
                order_exact=order_exact,policies={})
    for label,plan in plans.items():
        result["policies"][label]=dict(**plan,eager_exact_membership=True,graph_exact_replays=3,timing={})
    for (label,scope),v in samples.items():
        assert len(v)==50 and np.isfinite(v).all() and min(v)>0
        result["policies"][label]["timing"][scope]=dict(raw_us=v,p50_us=float(np.median(v)),p95_us=float(np.percentile(v,95)),p99_us=float(np.percentile(v,99)))
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print("V3B_CASE_PASS",a.case,json.dumps(dict(order_exact=order_exact,p50_us={k:{s:t["p50_us"] for s,t in v["timing"].items()} for k,v in result["policies"].items()})),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("host","device"),required=True)
    p.add_argument("--planner",type=Path,required=True)
    p.add_argument("--library",type=Path)
    p.add_argument("--cann",type=Path)
    p.add_argument("--local",type=Path)
    p.add_argument("--merge",type=Path)
    p.add_argument("--case",choices=CASES)
    p.add_argument("--output",type=Path)
    a=p.parse_args();assert __debug__
    if a.mode=="host":
        for name in CASES:geometry(name,a.planner)
        host_oracle()
        print("V3B_HOST_PLAN_PASS",flush=True)
    else:
        assert a.case and a.output and a.cann and all(x and x.is_file() for x in (a.library,a.local,a.merge))
        device_run(a,*geometry(a.case,a.planner))
if __name__=="__main__":main()
