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

def compile_harnesses():
    common=SOURCE/'csrc/operators/common/lifetch'
    inc=SOURCE/'csrc/operators/lightning_indexer_with_fetch/include'
    kernel=SOURCE/'csrc/operators/lightning_indexer_with_fetch/op_kernel'
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
    (TMP/'host.cpp').write_text(cpp)
    run(['c++','-std=c++17','-shared','-fPIC','-I'+str(common),'-I'+str(inc),str(TMP/'host.cpp'),'-o',str(TMP/'host.so')])
    lib=C.CDLL(str(TMP/'host.so'));lib.event.restype=C.c_uint64
    lib.kvip_lifetch_staged_forward.argtypes=[C.c_void_p]*3+[C.POINTER(RuntimeShape),C.POINTER(Plan)]
    lib.kvip_prepare_staged_shape.argtypes=[C.POINTER(RuntimeShape),C.c_uint32,C.c_uint32]
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
        for windows in (2,8):
            for post in range(3):
                for prepare in range(2):
                    s=RuntimeShape(batch=1,history=131072,resident=16384,indexHeads=32,aicCount=28,rounds=windows,tokenBytes=656)
                    self.assertEqual(lib.kvip_prepare_staged_shape(C.byref(s),56,2),0)
                    hot=(ScoreWave*1)(ScoreWave(100,101,102,28,640,4,16384))
                    ends=[16384+(114688//128*i//(windows-1))*128 for i in range(1,windows)]
                    cold=(ScoreWave*len(ends))(*[ScoreWave(200,201,202,28,640,4,e) for e in ends])
                    p=Plan(300,hot,cold,1,len(ends),56,2,400,windows,post,prepare,0)
                    def execute(fail=0):
                        lib.reset(fail);r=lib.kvip_lifetch_staged_forward(1,2,3,C.byref(s),C.byref(p))
                        ev=[(lib.event(i,0),[lib.event(i,a) for a in range(1,13)]) for i in range(lib.count())]
                        return r,ev
                    r,ev=execute();self.assertEqual(r,0)
                    seq=[x[0] for x in ev]
                    tag=lambda n:tags['aclrtlaunch_v9_'+n]
                    self.assertEqual(seq.count(tag('staged_prepare')),windows if prepare==0 else 0)
                    self.assertEqual(seq.count(tag('staged_prepare_all')),prepare)
                    self.assertEqual(seq.count(tag('staged_score_direct')),windows-1 if post==2 else 0)
                    self.assertEqual(seq.count(tag('staged_scatter')),windows if post==0 else 1)
                    self.assertEqual(seq.count(tag('staged_scatter_predicate')),windows-1 if post==1 else 0)
                    self.assertLess(seq.index(tag('staged_topk')),seq.index(tag('staged_service')))
                    for t,a in ev:
                        if t==tag('staged_scatter_predicate'):self.assertEqual(a[0],56)
                        if t==tag('staged_prepare_all'):self.assertEqual(a[0],28)
                    prog=[a for t,a in ev if t==tag('staged_progress')]
                    self.assertEqual(prog[0][4:6],[16384,16384])
                    self.assertEqual([a[5] for a in prog[1:]],ends)
                    if post:self.assertTrue(all(a[4]==a[5] for a in prog))
                    self.assertEqual(seq[-1],tag('streaming_owner_wave'))
                    # Every failed launch stops all subsequent work.
                    for i in range(1,len(seq)+1):
                        r,failed=execute(i);self.assertEqual(r,42);self.assertEqual(len(failed),i)
                    for field,value in [('postMode',3),('prepareMode',2),('reserved',1),('prepareWaves',0),('prepareCount',0)]:
                        if prepare==0 and field.startswith('prepare'):continue
                        old=getattr(p,field);setattr(p,field,value);r,bad=execute();self.assertEqual(r,100000);self.assertEqual(bad,[]);setattr(p,field,old)

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
                library=NS(kvip_staged_plan_bytes=Fn(64),kvip_staged_plan_version=Fn(2),kvip_lifetch_staged_forward=Fn(0)))
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
            r.library.kvip_staged_plan_bytes.v=40
            with self.assertRaises(ValueError):LIFetchForward(r,None)

    def test_05_cli_propagation_and_source_integrity(self):
        for post in ('separate','scatter','direct'):
            for prepare in ('wave','invocation'):
                c=configuration(['bench','--batches','1','--score-route','history','--score-post',post,'--score-prepare',prepare])
                self.assertEqual((c.score_post,c.score_prepare),(post,prepare))
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
        self.assertIn("'score_route', 'score_post', 'score_prepare')",e)
        self.assertIn("score_post=score_post, score_prepare=score_prepare, **binding)",e)
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

if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='kvip-cold-cpu-') as d:
        TMP=Path(d);LIB=compile_harnesses()
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        sys.exit(not result.wasSuccessful())
