#!/usr/bin/env python3
"""CPU-only checks of actual private ABI, scheduling, pack and prepare code.
Does not validate CANN instructions, NPU numerics or performance.
"""
from pathlib import Path
import ast, ctypes as C, hashlib, importlib, json, os, re, subprocess, sys, tempfile, unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np
TASK=Path(__file__).resolve().parent
SOURCE=TASK.parents[3]
sys.path[:0]=[str(SOURCE/'python'),str(SOURCE/'tests/operators/lifetch/support')]
from kvip.operators.lifetch.tiling import RuntimeShape
from kvip.operators.lifetch.staged import Plan, ScoreWave, PrepareWave, StagedLIFetchForward
from kvip.operators.lifetch.schedule import split_score_tasks
from workloads import configuration
TMP=None
LIB=None

def run(cmd, **kw):
    p=subprocess.run(cmd, text=True, capture_output=True, **kw)
    if p.returncode: raise RuntimeError(f'{cmd}:\n{p.stdout}\n{p.stderr}')
    return p.stdout

def compile_host(source, directory):
    common=source/'csrc/operators/common/lifetch'
    inc=source/'csrc/operators/lightning_indexer_with_fetch/include'
    kernel=source/'csrc/operators/lightning_indexer_with_fetch/op_kernel'
    host=(kernel.parent/'op_host/launch.cpp').read_text()
    host=host[host.index('#include "../../common/lifetch/staged_schedule.h"'):].replace('../../common/lifetch/','')
    declarations=re.findall(r'extern "C" uint32_t (aclrtlaunch_\w+)\(([^;]+)\);', host)
    tags={name:i+1 for i,(name,_) in enumerate(declarations)}
    wrappers=[]
    for name,params in declarations:
        params=params.split(',');named=','.join(f'{p} a{i}' for i,p in enumerate(params))
        args=[f'(uint64_t)a{i}' for i in range(len(params))]
        wrappers.append(f'extern "C" uint32_t {name}({named}){{return record({tags[name]},{{{",".join(args)}}});}}')
    cpp='''#include <cstdint>
#include <cstring>
#include <vector>
#include <initializer_list>
#include "runtime_shape.h"
struct Event{uint64_t tag,args[12];};std::vector<Event> events;uint32_t failAt=0;
uint32_t record(uint32_t tag,std::initializer_list<uint64_t> args){Event e{};e.tag=tag;unsigned i=0;for(auto a:args)e.args[i++]=a;events.push_back(e);return failAt==events.size()?42:0;}
extern "C" void reset(uint32_t failure){events.clear();failAt=failure;}
extern "C" uint32_t count(){return events.size();}
extern "C" uint64_t event(uint32_t i,uint32_t a){return a==0?events.at(i).tag:events.at(i).args[a-1];}
extern "C" uint32_t v9_init(void* b,void* st,const rotating::RuntimeShape* s){return record(0,{s->batch,(uint64_t)st,(uint64_t)b});}
'''+host+'\n'+'\n'.join(wrappers)
    (directory/'host.cpp').write_text(cpp)
    run(['c++','-std=c++17','-shared','-fPIC','-I'+str(common),'-I'+str(inc),str(directory/'host.cpp'),'-o',str(directory/'host.so')])
    lib=C.CDLL(str(directory/'host.so'));lib.event.restype=C.c_uint64
    lib.kvip_lifetch_staged_forward.argtypes=[C.c_void_p]*3+[C.POINTER(RuntimeShape),C.POINTER(Plan)]
    lib.kvip_prepare_staged_shape.argtypes=[C.POINTER(RuntimeShape),C.c_uint32,C.c_uint32]
    return lib,tags

def compile_harnesses():
    common=SOURCE/'csrc/operators/common/lifetch'
    inc=SOURCE/'csrc/operators/lightning_indexer_with_fetch/include'
    kernel=SOURCE/'csrc/operators/lightning_indexer_with_fetch/op_kernel'
    lib,tags=compile_host(SOURCE,TMP)
    # Execute actual VF body with CPU threads; no device timing/synchronization model.
    pack='''#include <cstdint>
#define __simt_vf__
#define __aicore__
#define __simt_callee__
#define __ubuf__
#define LAUNCH_BOUND(x)
struct {uint32_t x;} threadIdx;
#include "staged_post_device.h"
extern "C" void direct_pack(uint16_t* s,uint8_t* b,uint16_t* t,uint32_t q,uint32_t token,uint32_t v0,uint32_t v1){
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)kvip_staged_device::PackColdThresholdUB(s,b,t,q,token,v0,v1);
}
extern "C" uint8_t pack4(uint16_t a,uint16_t b,uint16_t c,uint16_t d,uint16_t t,uint32_t x,uint32_t v){return kvip_staged_post::Pack4(a,b,c,d,t,x,v);}
'''
    (TMP/'pack.cpp').write_text(pack)
    run(['c++','-std=c++17','-shared','-fPIC','-I'+str(kernel),str(TMP/'pack.cpp'),'-o',str(TMP/'pack.so')])
    packlib=C.CDLL(str(TMP/'pack.so'));packlib.pack4.argtypes=[C.c_uint16]*5+[C.c_uint32]*2;packlib.pack4.restype=C.c_uint8
    packlib.direct_pack.argtypes=[C.c_void_p]*3+[C.c_uint32]*4
    # Run actual ready publication with the real Context, Progress and lane mapping.
    vector=(kernel/'staged_indexer.h').read_text()
    publication=vector[vector.index('__aicore__ inline void PublishServiceReady'):vector.index('__aicore__ inline void Service(GM_ADDR')]
    pub='''#include <cstdint>
#include <cassert>
#include <vector>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
#include "context.h"
namespace rotating {RuntimeShape LoadRuntimeShape(Context* p){return p->shape;}}
uint32_t block;std::vector<uint8_t*> cleaned;
namespace AscendC {uint32_t GetBlockIdx(){return block;}}
namespace kvaf_device {void CleanCacheLines(uint8_t* p,uint32_t n){assert(n==128);cleaned.push_back(p);}}
'''+publication+'''
int main(){for(uint32_t batch:{1U,2U,3U,16U,32U})for(uint32_t end:{16384U,32768U,131072U}) {
 rotating::Context ctx{};ctx.shape.batch=batch;ctx.shape.serviceAics=1;ctx.shape.serviceLanes=2;
 std::vector<rotating::Progress> progress(batch*4+4);ctx.progress=(uint64_t)progress.data();
 uint64_t args[16]{};args[14]=(uint64_t)&ctx;
 for(auto& p:progress)for(auto& e:p.end)e=0xA5A5A5A5;
 // Deliberately run lane 1 first: lane 0 can still have unpublished records.
 for(block=1;;--block){cleaned.clear();PublishServiceReady((GM_ADDR)args,end);
  assert(cleaned.size()==((batch+1-block)/2)*4);
  for(uint32_t b=block;b<batch;b+=2)for(uint32_t q=0;q<4;++q) {
   assert(progress[b*4+q].end[0]==end);
   for(uint32_t i=1;i<32;++i)assert(progress[b*4+q].end[i]==0xA5A5A5A5);
  }
  if(block==1)for(uint32_t b=0;b<batch;b+=2)for(uint32_t q=0;q<4;++q)assert(progress[b*4+q].end[0]==0xA5A5A5A5);
  if(!block)break;
 }
 for(uint32_t q=batch*4;q<progress.size();++q)for(auto e:progress[q].end)assert(e==0xA5A5A5A5);
}}
'''
    (TMP/'publication.cpp').write_text(pub)
    (TMP/'batch_get_abi.h').write_text('#pragma once\n')  # Context uses no declarations from this transport header.
    run(['c++','-std=c++17','-I'+str(TMP),'-I'+str(inc),str(TMP/'publication.cpp'),'-o',str(TMP/'publication')])
    run([str(TMP/'publication')])
    # Actual PrepareRow / prepare_all under byte-DMA mocks, including live refresh.
    prepare=(kernel/'staged_indexer.h').read_text();all_vector=prepare;prepare=prepare[prepare.index('__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void StagedPreparePages'):]
    stub=(TASK/'tests/prepare_stub.cpp').read_text()
    (TMP/'prepare.cpp').write_text(stub.replace('// ACTUAL_PREPARE',prepare).replace('// ACTUAL_SCATTER',all_vector[all_vector.index('extern "C" __global__ __aicore__ void v9_staged_scatter_predicate'):all_vector.index('extern "C" __global__ __aicore__ void v9_staged_topk')]))
    run(['c++','-std=c++17','-I'+str(common),'-I'+str(inc),'-I'+str(kernel),str(TMP/'prepare.cpp'),'-o',str(TMP/'prepare')])
    run([str(TMP/'prepare')])
    return lib,tags,packlib

class Tensor:
    def __init__(self,v):self.v=np.asarray(v)
    def data_ptr(self):return self.v.ctypes.data
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.v
    def numel(self):return self.v.size

class Checks(unittest.TestCase):
    def test_01_actual_host_schedule_and_failure(self):
        lib,tags,_=LIB
        self.assertEqual(C.sizeof(Plan),64);self.assertEqual(C.sizeof(PrepareWave),32)
        import itertools
        for windows,post,prepare,service in itertools.product((2,8),range(3),range(3),range(2)):
            if service and not post:continue
            s=RuntimeShape(batch=1,history=131072,resident=16384,indexHeads=32,aicCount=28,rounds=windows,tokenBytes=656)
            self.assertEqual(lib.kvip_prepare_staged_shape(C.byref(s),56,2),0)
            hot=(ScoreWave*1)(ScoreWave(100,101,102,4,4096,4,16384))
            ends=[16384+(114688//128*i//(windows-1))*128 for i in range(1,windows)]
            cold=(ScoreWave*len(ends))(*[ScoreWave(200,201,202,8,14336 if windows==2 else 2048,4,e) for e in ends])
            p=Plan(300,hot,cold,1,len(ends),56,2,400,windows,post,prepare,service)
            def execute(fail=0):
                lib.reset(fail);r=lib.kvip_lifetch_staged_forward(1,2,3,C.byref(s),C.byref(p))
                ev=[(lib.event(i,0),[lib.event(i,a) for a in range(1,13)]) for i in range(lib.count())]
                return r,ev
            r,ev=execute();self.assertEqual(r,0)
            seq=[x[0] for x in ev];tag=lambda n:tags['aclrtlaunch_v9_'+n]
            self.assertEqual(seq.count(tag('staged_prepare')),windows if prepare==0 else 0)
            self.assertEqual(seq.count(tag('staged_prepare_all')),prepare)
            self.assertEqual(seq.count(tag('staged_score_direct')),windows-1 if post==2 else 0)
            self.assertEqual(seq.count(tag('staged_scatter')),windows if post==0 else 1)
            self.assertEqual(seq.count(tag('staged_scatter_predicate')),windows-1 if post==1 else 0)
            svc=tag('staged_service_ready') if service else tag('staged_service')
            self.assertEqual(seq.count(svc),windows)
            self.assertLess(seq.index(tag('staged_topk')),seq.index(svc))
            prep=[(i,a) for i,(t,a) in enumerate(ev) if t==tag('staged_prepare_all')]
            if prepare==1:self.assertEqual(prep[0][1][0],8)
            if prepare==2:
                self.assertEqual([a[0] for _,a in prep],[4,8])
                self.assertEqual([a[3:5] for _,a in prep],[[400,1],[432,windows-1]])
                self.assertGreater(prep[1][0],seq.index(svc))
                self.assertLess(prep[1][0],seq.index(tag('staged_score_direct')) if post==2 else seq.index(tag('staged_score'),seq.index(svc)))
            prog=[a for t,a in ev if t==tag('staged_progress')]
            if service:
                self.assertEqual(prog,[])
                self.assertEqual([a[4] for t,a in ev if t==svc],[16384]+ends)
            else:
                self.assertEqual(prog[0][4:6],[16384,16384])
                self.assertEqual([a[5] for a in prog[1:]],ends)
                if post:self.assertTrue(all(a[4]==a[5] for a in prog))
            self.assertEqual(seq[-1],tag('streaming_owner_wave'))
            for i in range(1,len(seq)+1):
                r,failed=execute(i);self.assertEqual(r,42);self.assertEqual(len(failed),i)
            for field,value in [('postMode',3),('prepareMode',3),('serviceMode',2),('prepareWaves',0),('prepareCount',0)]:
                if prepare==0 and field.startswith('prepare') and field!='prepareMode':continue
                old=getattr(p,field);setattr(p,field,value);r,bad=execute();self.assertEqual(r,100000);self.assertEqual(bad,[]);setattr(p,field,old)
            if not post:
                p.serviceMode=1;r,bad=execute();self.assertEqual(r,100000);self.assertEqual(bad,[])

    def test_01b_multi_owner_waves_publish_only_at_epoch_end(self):
        lib,tags,_=LIB
        for batch in (3,32):
            s=RuntimeShape(batch=batch,history=131072,resident=16384,indexHeads=32,aicCount=28,rounds=8,tokenBytes=656)
            self.assertEqual(lib.kvip_prepare_staged_shape(C.byref(s),56,2),0)
            def phase(begin,end):
                _,_,tasks=split_score_tasks(batch,begin,end,28,56,'history',partitions=16)
                return [ScoreWave(100,101,102,len(chunk),max(t[3]-t[2] for t in chunk),4,
                                  end if first+len(chunk)==len(tasks) else 0)
                        for first in range(0,len(tasks),28) for chunk in [tasks[first:first+28]]]
            hot=phase(0,16384);cold=[]
            for epoch in range(1,8):cold.extend(phase(16384+(epoch-1)*16384,16384+epoch*16384))
            h=(ScoreWave*len(hot))(*hot);c=(ScoreWave*len(cold))(*cold)
            p=Plan(300,h,c,len(h),len(c),56,2,400,len(h)+len(c),2,2,1)
            lib.reset(0);self.assertEqual(lib.kvip_lifetch_staged_forward(1,2,3,C.byref(s),C.byref(p)),0)
            events=[(lib.event(i,0),[lib.event(i,a) for a in range(1,13)]) for i in range(lib.count())]
            tag=lambda name:tags['aclrtlaunch_v9_'+name]
            services=[(i,args) for i,(t,args) in enumerate(events) if t==tag('staged_service_ready')]
            self.assertEqual([args[4] for _,args in services],list(range(16384,131073,16384)))
            prior=services[0][0]
            for (at,args) in services[1:]:
                between=events[prior+1:at]
                self.assertEqual(sum(t==tag('staged_score_direct') for t,_ in between),len(hot))
                prior=at
            prep=[args for t,args in events if t==tag('staged_prepare_all')]
            self.assertEqual([args[3] for args in prep],[400,400+32*len(hot)])
            self.assertEqual([args[4] for args in prep],[len(hot),len(cold)])

    def test_02_pack_exact_oracle_and_direct_ub_banks(self):
        _,_,lib=LIB;rng=np.random.default_rng(17)
        def oracle(keys,theta,token,end):return sum((1<<(2*i)) for i,k in enumerate(keys) if token+i<end and int(k)>theta)
        for theta in (0,1,32768,65535):
            for tail in range(5):
                keys=[theta,max(0,theta-1),min(65535,theta+1),65535]
                self.assertEqual(lib.pack4(*keys,theta,1024,1024+tail),oracle(keys,theta,1024,1024+tail))
        for _ in range(2000):
            keys=rng.integers(0,65536,4,dtype=np.uint16);t=int(rng.integers(0,65536));end=int(rng.integers(0,140))
            self.assertEqual(lib.pack4(*map(int,keys),t,128,end),oracle(keys,t,128,end))
        for q in (1,2):
            for bank in (0,1):
                scores=np.full(1024,0x5555,dtype=np.uint16);bits=np.full(1152,0xA5,dtype=np.uint8);theta=np.zeros(256,dtype=np.uint16)
                token=32768;ends=[token+127,token+65]
                for k in range(q):scores[bank*128+k*256:bank*128+k*256+128]=rng.integers(0,65536,128,dtype=np.uint16);theta[k*128]=32768
                lib.direct_pack(scores.ctypes.data+bank*256,bits.ctypes.data+bank*512,theta.ctypes.data,q,token,*ends)
                touched=set()
                for k in range(q):
                    for x in range(32):
                        off=bank*512+k*256+x;touched.add(off)
                        self.assertEqual(int(bits[off]),oracle(scores[bank*128+k*256+x*4:bank*128+k*256+x*4+4],32768,token+x*4,ends[k]))
                self.assertTrue(all(v==0xA5 for i,v in enumerate(bits) if i not in touched))

    def test_03_paired_coverage_and_sector_exclusivity(self):
        for b in (1,2,4,16,32):
            for route in ('history','bsq','auto'):
                qg,_,tasks=split_score_tasks(b,16384,131072,28,56,route)
                ranges={};sectors=set()
                for owner,qbegin,begin,end in tasks:
                    for lane in (0,1):
                        for q in range(qbegin+lane*(qg//2),qbegin+(lane+1)*(qg//2)):
                            ranges.setdefault((owner,q),[]).append((begin,end))
                            for sector in range(begin//128,end//128):
                                key=(owner,q,sector);self.assertNotIn(key,sectors);sectors.add(key)
                self.assertEqual(len(ranges),b*4)
                for rr in ranges.values():
                    rr.sort();self.assertEqual(rr[0][0],16384);self.assertEqual(rr[-1][1],131072)
                    self.assertTrue(all(a[1]==z[0] for a,z in zip(rr,rr[1:])))
        # Explicit partitions may require multiple waves across owners. Every
        # task still owns complete 32B bitmap sectors and exact Q/H coverage.
        for b in (1,3,16,32):
            for p in (1,2,4,8,16,28):
                qg,producers,tasks=split_score_tasks(b,0,16384,28,56,'history',partitions=p)
                self.assertEqual((qg,producers,len(tasks)),(4,p,b*p))
                for owner in range(b):
                    rr=[(t[2],t[3]) for t in tasks if t[0]==owner]
                    self.assertEqual(rr[0][0],0);self.assertEqual(rr[-1][1],16384)
                    self.assertTrue(all(a[1]==z[0] for a,z in zip(rr,rr[1:])))
        for kwargs in (dict(partitions=29),dict(partitions=2,route='bsq'),dict(partitions=True)):
            options=dict(partitions=0,route='history');options.update(kwargs)
            with self.assertRaises(ValueError):split_score_tasks(1,0,16384,28,56,**options)

    def test_04_real_python_constructor_and_option_inheritance(self):
        staged=importlib.import_module('kvip.operators.lifetch.staged')
        from kvip.operators.lifetch.forward import LIFetchForward
        def make_rot():
            args=[Tensor([0]) for _ in range(14)];args[5]=Tensor(np.array([4],np.int32));args[6]=Tensor(np.array([131072],np.int32))
            fn=lambda value:NS(__call__=lambda:value)
            class Fn:
                def __init__(self,v):self.v=v
                def __call__(self,*a):return self.v
            torch=NS(uint16=np.uint16,uint8=np.uint8,int32=np.int32,bfloat16=np.uint16,float32=np.float32,
                tensor=lambda v,dtype,device:Tensor(np.asarray(v,dtype=dtype)),full=lambda sh,v,dtype,device:Tensor(np.full(sh,v,dtype=dtype)))
            return NS(staged=True,torch=torch,device='cpu',runtime_shape=RuntimeShape(batch=1,history=131072,resident=16384,indexHeads=32,aicCount=28,rounds=8),
                arguments=args,streaming_scratch={},score_route='history',score_post='direct',score_prepare='invocation',prefetch_limit=512,
                library=NS(kvip_staged_plan_bytes=Fn(64),kvip_staged_plan_version=Fn(3),kvip_lifetch_staged_forward=Fn(0)))
        def init(self,r,u,*a,**kw):self.torch=r.torch;self.rotating=r;self.schedule=NS(aiv_budget=56)
        def buffer(pool,torch,device,name,shape,dtype,*layout):return Tensor(np.zeros(shape,dtype=dtype))
        with patch.object(staged.StreamingLIFetchForward,'__init__',init),patch.object(staged.StreamingLIFetchForward,'set_options',lambda s:None),patch.object(staged,'score_buffer',buffer),patch.object(staged,'device_words',lambda torch,v,device:Tensor(np.asarray(v,np.uint64))):
            r=make_rot();a=LIFetchForward(r,None);z=LIFetchForward(r,None)
            self.assertEqual((a.plan.postMode,a.plan.prepareMode),(2,1));self.assertEqual(a.plan.prepareCount,8)
            self.assertNotEqual(a.plan.prepareWaves,z.plan.prepareWaves)
            descriptors=(PrepareWave*8).from_buffer_copy(a.prepare_descriptors.v.tobytes())
            self.assertEqual(descriptors[0].bindings,a.hot[0].bindings)
            self.assertEqual(descriptors[7].tasks,a.cold[-1].tasks)
            self.assertEqual(a.score_geometry['waves'][0]['rows'],28)
            r.score_prepare='phase';r.score_service='inline';r.score_hot_partitions=4;r.score_cold_partitions=8
            optimized=LIFetchForward(r,None)
            self.assertEqual((optimized.plan.prepareMode,optimized.plan.serviceMode),(2,1))
            self.assertEqual(optimized.hot[0].rows,4)
            self.assertTrue(all(w.rows==8 for w in optimized.cold))
            for partitions in (-1,29,True):
                with self.assertRaises(ValueError):LIFetchForward(r,None,score_hot_partitions=partitions)
            with self.assertRaises(ValueError):LIFetchForward(r,None,score_post='separate')
            r.library.kvip_staged_plan_bytes.v=40
            with self.assertRaises(ValueError):LIFetchForward(r,None)

    def test_05_cli_propagation_and_source_integrity(self):
        for post in ('separate','scatter','direct'):
            for prepare in ('wave','invocation','phase'):
                c=configuration(['bench','--batches','1','--score-route','history','--score-post',post,'--score-prepare',prepare])
                self.assertEqual((c.score_post,c.score_prepare),(post,prepare))
        c=configuration(['bench','--score-route','history','--score-post','direct','--score-prepare','phase',
                         '--score-service','inline','--score-hot-partitions','4','--score-cold-partitions','8'])
        self.assertEqual((c.score_service,c.score_hot_partitions,c.score_cold_partitions),('inline',4,8))
        for args in (['bench','--score-post','direct'],['bench','--transport','ubmem','--score-route','history'],['diagnose','--score-route','history']):
            with self.assertRaises(SystemExit):configuration(args)
        manifest=json.loads((TASK/'baseline.json').read_text());changed=[]
        for p,digest in manifest['files_sha256'].items():
            now=hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()
            if now!=digest:changed.append(p)
            if p.startswith('csrc/operators/lightning_indexer_with_fetch/include/') or p in ('scripts/build.sh','scripts/build.py','pyproject.toml'):
                self.assertEqual(now,digest,p)
        for p in SOURCE.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
        e=(SOURCE/'tests/operators/lifetch/support/execution.py').read_text()
        self.assertIn("'score_route', 'score_post', 'score_prepare', 'score_service', 'score_hot_partitions', 'score_cold_partitions')",e)
        self.assertIn("score_hot_partitions=score_hot_partitions, score_cold_partitions=score_cold_partitions, **binding)",e)
        v=(SOURCE/'python/kvip/adapters/vllm/binding.py').read_text();self.assertIn('score_post=score_post, score_prepare=score_prepare',v)
        kernel=(SOURCE/'csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer.cpp').read_text()
        self.assertEqual(kernel.count('op.scoreOnly_=true; op.scoreQueryGroup_=queryGroup;'),2)
        self.assertIn('op.vectorService.SetStagedOutput(original,canonical,tasks);',kernel)
        vector=(SOURCE/'csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer_vector.h').read_text()
        self.assertLess(vector.index('PackColdThresholdUB',vector.index('ProcessVec1(const')),vector.index('SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);',vector.index('ProcessVec1(const')))
        print('Changed tracked source:',', '.join(changed))

    def test_06_repository_control_commands(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('cold_controls',TASK/'run_controls.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        args=NS(rounds=3,windows=[2,8],prefetch=[0,512],python=sys.executable,repo=SOURCE,device=5,kvaf_root=Path('/tmp/KVAF'),warmup=4,iterations=20,aic_budget=28,output=TMP/'results',timeout=300,aiv_budget=56,kvaf_server=None,valid_history=None)
        cmds=list(m.commands(args));self.assertEqual(len(cmds),4*2*2*6)
        for name,r,cmd in cmds:
            c=configuration(cmd[2:]);self.assertEqual(c.batches,(1,));self.assertEqual((c.history,c.resident,c.queries,c.index_heads),(131072,16384,4,32))
            self.assertEqual((c.aic_budget,c.aiv_budget),(28,56))
        for name in ('run_controls.py','summarize_controls.py','verify_cpu.py'):ast.parse((TASK/name).read_text())
        args.matrix='tuning';args.hot_partitions=[4,8];args.cold_partitions=[4,8,16];args.prepares=['phase','invocation'];args.fetch_workers=[1,2,4]
        tuned=list(m.commands(args))
        self.assertEqual(len(tuned),4*2*2*3*(5+2*3*2))
        self.assertEqual(len(tuned),len({name for name,_,_ in tuned}))
        for name,r,cmd in tuned:
            c=configuration(cmd[2:]);self.assertEqual((c.aic_budget,c.aiv_budget),(28,56))
            if c.score_hot_partitions:
                self.assertEqual((c.score_post,c.score_service),('direct','inline'))

    def test_07_summary_requires_complete_evidence_including_warmup(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('summary_controls',TASK/'run_controls.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        out=TMP/'summary-fixtures';out.mkdir()
        args=NS(rounds=1,windows=[8],prefetch=[512],python=sys.executable,repo=SOURCE,device=5,kvaf_root=Path('/tmp/KVAF'),warmup=4,iterations=20,aic_budget=28,output=out,timeout=300,aiv_budget=56,kvaf_server=None,valid_history=None)
        commands=list(m.commands(args));planned=[]
        for name,r,cmd in commands:
            planned.append(dict(name=name,discard=r==0,argv=cmd))
            log=out/(name+'.log');log.write_text('CPU summary fixture: no hardware measurement')
            (out/(name+'.receipt.json')).write_text(json.dumps(dict(name=name,discard=r==0,argv=cmd,exit_code=0,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())))
            folder=out/name;folder.mkdir();(folder/'report.json').write_text(json.dumps(dict(correctness='PASS',score_geometry={},timing=dict(prototype_with_update=dict(p50_us=1.),native_chain=dict(p50_us=1.)))))
        (out/'manifest.json').write_text(json.dumps(dict(commands=planned)))
        command=[sys.executable,str(TASK/'summarize_controls.py'),str(out)]
        run(command);summary=json.loads((out/'summary.json').read_text());self.assertEqual(len(summary),6);self.assertTrue(all(x['rounds']==1 for x in summary))
        victim=out/(commands[-1][0]+'.receipt.json');data=victim.read_bytes();victim.unlink()
        self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0);victim.write_bytes(data)
        report=out/commands[0][0]/'report.json';data=report.read_bytes();failed=json.loads(data);failed['correctness']='FAIL';report.write_text(json.dumps(failed))
        self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0);report.write_bytes(data)
        log=out/(commands[-1][0]+'.log');log.write_text('tampered')
        self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)

    def test_08_generated_kernel_physical_types(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('staged_metadata',SOURCE/'experiments/split_core/verify_staged_metadata.py')
        metadata=importlib.util.module_from_spec(spec);spec.loader.exec_module(metadata)
        folder=TMP/'auto-gen';folder.mkdir()
        aic='\n'.join(f'auto_gen_v9_staged_{k}_kernel=v9_staged_{k}_mix_aic' for k in metadata.SCORE_KERNELS)
        aiv='\n'.join([f'auto_gen_v9_staged_{k}_kernel=v9_staged_{k}_aiv' for k in metadata.AIV_KERNELS]+
                      [f'auto_gen_v9_staged_{k}_kernel=v9_staged_{k}_mix_aiv' for k in metadata.SCORE_KERNELS])
        (folder/'aic_config.cmake').write_text(aic);(folder/'aiv_config.cmake').write_text(aiv)
        (folder/'auto_gen_indexer.cpp').write_text('\n'.join(f'v9_staged_{k}_mix_aic_section K_TYPE_MIX_AIC_MAIN 1, 2;' for k in metadata.SCORE_KERNELS))
        (folder/'auto_gen_staged_vector.cpp').write_text('// AIV-only test metadata, no device compile')
        self.assertEqual(metadata.verify(folder)['status'],'PASSED')
        (folder/'aic_config.cmake').write_text(aic+'\nauto_gen_v9_staged_service_ready_kernel=wrong_mix_aic')
        with self.assertRaises(ValueError):metadata.verify(folder)
        (folder/'aic_config.cmake').write_text(aic)
        (folder/'auto_gen_indexer.cpp').write_text('v9_staged_score_mix_aic_section K_TYPE_MIX_AIC_MAIN 1, 2;')
        with self.assertRaises(ValueError):metadata.verify(folder)

    def test_09_tuning_summary_pairs_and_observed_best(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('tuning_controls',TASK/'run_controls.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        out=TMP/'tuning-summary';out.mkdir()
        args=NS(rounds=2,windows=[2],prefetch=[512],python=sys.executable,repo=SOURCE,device=5,
                kvaf_root=Path('/tmp/KVAF'),warmup=4,iterations=20,aic_budget=28,output=out,timeout=300,
                aiv_budget=56,kvaf_server=None,valid_history=None,matrix='tuning',hot_partitions=[4],
                cold_partitions=[8],prepares=['phase'],fetch_workers=[2])
        planned=[];candidate_report=None
        for name,r,cmd in runner.commands(args):
            config=configuration(cmd[2:]);planned.append(dict(name=name,discard=r==0,argv=cmd))
            log=out/(name+'.log');log.write_text('CPU-only aggregation fixture; no NPU timing')
            (out/(name+'.receipt.json')).write_text(json.dumps(dict(name=name,discard=r==0,argv=cmd,
                exit_code=0,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())))
            latency=3. if config.score_route=='legacy' else (2. if config.score_hot_partitions else 8.)
            folder=out/name;folder.mkdir();report=folder/'report.json'
            report.write_text(json.dumps(dict(correctness='PASS',binary_sha256='same-test-binary',
                resources=dict(effective_aic=28,aiv=56),score_geometry={},
                timing=dict(prototype_with_update=dict(p50_us=latency),native_chain=dict(p50_us=3.)))))
            if r and config.score_hot_partitions:candidate_report=report
        (out/'manifest.json').write_text(json.dumps(dict(commands=planned)))
        command=[sys.executable,str(TASK/'summarize_controls.py'),str(out)]
        run(command);best=json.loads((out/'best_candidates.json').read_text())
        self.assertEqual(len(best),1);self.assertEqual(best[0]['hot_partitions'],'4')
        self.assertEqual(best[0]['rounds'],2)
        self.assertAlmostEqual(best[0]['candidate_over_legacy_mean'],2/3)
        self.assertAlmostEqual(best[0]['candidate_over_staged_before_mean'],1/4)
        changed=json.loads(candidate_report.read_text());changed['resources']['effective_aic']=27
        candidate_report.write_text(json.dumps(changed))
        self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)

    def test_10_source_tradeoffs_and_explicit_shortlist(self):
        sys.path.insert(0,str(TASK))
        from simulate_schedule import execute
        from explore_schedule import conditional_cost,REPORT
        lib,tags,_=LIB
        def row(w,ph,pc,**kw):return execute(SOURCE,lib,tags,w,2,2,1,ph,pc,28,56,splitter=split_score_tasks,compact=True,**kw)
        a,b=row(2,28,28),row(8,28,28)
        self.assertEqual((a['score_critical_128key_iterations'],b['score_critical_128key_iterations']),(37,40))
        for w in (2,8):
            low=row(w,4,8);self.assertEqual(low['score_critical_128key_iterations'],144)
            self.assertEqual(low['logical_qk_macs'],a['logical_qk_macs'])
        lower=row(8,26,26)
        self.assertEqual(lower['score_critical_128key_iterations'],b['score_critical_128key_iterations'])
        self.assertLess(lower['prepared_q_weight_qscale_bytes_each_direction'],b['prepared_q_weight_qscale_bytes_each_direction'])
        q2=row(2,0,0,route='bsq');self.assertEqual(q2['logical_qk_macs'],a['logical_qk_macs'])
        base=execute(SOURCE,lib,tags,8,0,0,0,0,0,28,56,splitter=split_score_tasks,compact=True)
        self.assertAlmostEqual(conditional_cost(base,base,.5,.5,.1,.5),sum(REPORT.values()))
        import run_controls
        args=NS(rounds=1,windows=[2],prefetch=[512],python=sys.executable,repo=SOURCE,device=5,
                kvaf_root=Path('/tmp/KVAF'),warmup=4,iterations=20,aic_budget=28,output=TMP/'pairs',timeout=300,
                aiv_budget=56,kvaf_server=None,valid_history=None,matrix='tuning',hot_partitions=[4,8],
                cold_partitions=[4,8,16],partition_pairs=[(4,8),(26,28)],prepares=['phase'],fetch_workers=[2])
        commands=list(run_controls.commands(args));self.assertEqual(len(commands),14)
        for _,_,cmd in commands:
            c=configuration(cmd[2:])
            if c.score_hot_partitions:self.assertIn((c.score_hot_partitions,c.score_cold_partitions),args.partition_pairs)

    def test_11_model_plan_matches_actual_dispatch(self):
        sys.path.insert(0,str(TASK))
        from run_model_controls import configs
        from simulate_schedule import execute
        lib,tags,_=LIB
        for matrix in ('smoke','full'):
            rows=configs(matrix);self.assertEqual(len(rows),len({r['name'] for r in rows}))
            for r in rows:
                if r['route']=='legacy':continue
                out=execute(SOURCE,lib,tags,r['windows'],{'separate':0,'scatter':1,'direct':2}[r['post']],
                    {'wave':0,'invocation':1,'phase':2}[r['prepare']],int(r['service']=='inline'),r['hot'],r['cold'],
                    r['aics'],r['aivs'],batch=r['batch'],history=r['history'],resident=r['resident'],route=r['route'],
                    workers=r['workers'],splitter=split_score_tasks,compact=True)
                self.assertEqual(out['cold_predicate_comparisons'],r['batch']*4*(r['history']-r['resident']))

if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='kvip-cold-cpu-') as d:
        TMP=Path(d);LIB=compile_harnesses()
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        sys.exit(not result.wasSuccessful())
