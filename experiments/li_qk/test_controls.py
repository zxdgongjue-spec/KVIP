"""CPU validation of actual host/plan/guards. Never device performance evidence."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
controls=importlib.import_module('run')

ACL_STUB=r'''
#pragma once
#include <cstdlib>
#include <cstring>
using aclrtStream=void*;using aclrtEvent=void*;
enum {ACL_MEM_MALLOC_HUGE_FIRST=0,ACL_MEMCPY_HOST_TO_DEVICE=0,ACL_MEMCPY_DEVICE_TO_HOST=1};
inline unsigned aclInit(const char*){return 0;}inline unsigned aclFinalize(){return 0;}
inline unsigned aclrtSetDevice(unsigned){return 0;}inline unsigned aclrtResetDevice(unsigned){return 0;}
inline unsigned aclrtCreateStream(aclrtStream* s){*s=(void*)1;return 0;}
inline unsigned aclrtDestroyStream(aclrtStream){return 0;}inline unsigned aclrtSynchronizeStream(aclrtStream){return 0;}
inline unsigned aclrtMalloc(void** p,size_t n,int){*p=malloc(n);return *p?0:1;}
inline unsigned aclrtFree(void* p){free(p);return 0;}
inline unsigned aclrtMemcpy(void* d,size_t n,const void* s,size_t bytes,int){if(bytes>n)return 1;memcpy(d,s,bytes);return 0;}
inline unsigned aclrtCreateEvent(aclrtEvent* e){*e=(void*)1;return 0;}inline unsigned aclrtDestroyEvent(aclrtEvent){return 0;}
inline unsigned aclrtRecordEvent(aclrtEvent,aclrtStream){return 0;}
inline unsigned aclrtEventElapsedTime(float* m,aclrtEvent,aclrtEvent){*m=1;return 0;}
'''
LAUNCH_STUB=r'''
#include "plan.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <vector>
static void mark(char c){std::ofstream f(getenv("LI_QK_LAUNCH_LOG"),std::ios::app);f<<c;}
static std::vector<unsigned short> oracle;
static void load(){if(!oracle.empty())return;std::ifstream f(getenv("LI_QK_ORACLE"),std::ios::binary|std::ios::ate);auto n=f.tellg()/2;oracle.resize(n);f.seekg(0);f.read((char*)oracle.data(),n*2);}
extern "C" unsigned aclrtlaunch_li_qk_score(unsigned,void*,void* binding,unsigned rows,unsigned stride,unsigned heads,unsigned group){
 mark('S');load();auto a=(unsigned long long*)binding;auto query=(unsigned char*)a[0];auto qlen=(unsigned*)a[5];auto klen=(unsigned*)a[6];auto pages=(int*)a[7];auto out=(unsigned short*)a[12];unsigned physical=group==1?2:group;
 for(unsigned r=0;r<rows;++r)for(unsigned q=0;q<qlen[r];++q){auto base=query+(r*physical+q)*heads*128;unsigned d=0;while(d<128&&!base[d])++d;unsigned h=oracle.size()/(4*(getenv("LI_QK_BATCH")?atoi(getenv("LI_QK_BATCH")):1));unsigned begin=pages[r*stride/128]*128;unsigned owner=begin/h;unsigned realQ=(d+128-owner)%128/13;
  for(unsigned x=0;x<klen[r];++x)out[(r*physical+q)*stride+x]=oracle[(owner*4+realQ)*h+begin%h+x];}
 if(getenv("LI_QK_CORRUPT_PADDING")&&group==1)out[stride]=0;
 return 0;
}
extern "C" unsigned aclrtlaunch_li_qk_score_balanced(unsigned,void* stream,void* list,unsigned slots,unsigned rows,unsigned stride,unsigned heads,unsigned group){
 mark('B');auto bindings=(unsigned long long*)list;
 for(unsigned slot=0;slot<slots;++slot)aclrtlaunch_li_qk_score(rows,stream,(void*)bindings[slot],rows,stride,heads,group);return 0;
}
extern "C" unsigned aclrtlaunch_li_qk_scatter(unsigned,void*,void* input,void* canonical,void* tasks,unsigned rows,unsigned stride,unsigned h,unsigned physical){
 mark('C');auto src=(unsigned short*)input;auto dst=(unsigned short*)canonical;auto t=(li_qk::Task*)tasks;
 for(unsigned r=0;r<rows;++r)for(unsigned q=0;q<t[r].queryCount;++q)std::copy_n(src+(r*physical+q)*stride,t[r].end-t[r].begin,dst+(t[r].owner*4+t[r].queryBegin+q)*h+t[r].begin);return 0;
}
extern "C" unsigned aclrtlaunch_li_qk_topk(unsigned,void*,void* canonical,void* indices,void*,unsigned h,unsigned,unsigned queries){
 mark('K');auto src=(unsigned short*)canonical;auto dst=(int*)indices;
 for(unsigned q=0;q<queries;++q){std::vector<int> order(h);std::iota(order.begin(),order.end(),0);std::partial_sort(order.begin(),order.begin()+2048,order.end(),[&](int a,int b){return src[q*h+a]>src[q*h+b]||(src[q*h+a]==src[q*h+b]&&a<b);});std::copy_n(order.begin(),2048,dst+q*2048);}return 0;
}
'''


class Controls(unittest.TestCase):
    def test_01_cpp_python_plans_and_coordinate_coverage(self):
        cpp='#include "plan.h"\n#include <iostream>\nint main(){unsigned pts[][2]={{1,1},{2,1},{4,1},{8,1},{16,1},{1,2},{1,4}};for(auto pt:pts){auto t=pt[0],q=pt[1];auto p=li_qk::MakePlan(131072,32,32,64,t,q);std::cout<<t<<","<<q<<","<<p.stride<<","<<p.physicalGroup<<","<<p.tasks.size()<<"\\n";for(auto r:p.tasks)std::cout<<r.queryBegin<<","<<r.queryCount<<","<<r.begin<<","<<r.end<<"\\n";}}'
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'plan.cpp').write_text(cpp)
            subprocess.run(['c++','-std=c++17','-I'+str(HERE),str(d/'plan.cpp'),'-o',str(d/'plan')],check=True,capture_output=True)
            lines=iter(subprocess.check_output([str(d/'plan')],text=True).splitlines())
            for t,qp in controls.POINTS:
                p=controls.geometry(t,qp);cover=np.zeros((4,131072),np.uint8)
                self.assertEqual(list(map(int,next(lines).split(','))),[t,qp,p['row_stride'],p['physical_query_tile'],p['tasks']])
                for r in p['task_map']:
                    self.assertEqual(list(map(int,next(lines).split(','))),list(r.values()))
                    cover[r['query_begin']:r['query_begin']+r['query_count'],r['begin']:r['end']]+=1
                self.assertTrue(np.all(cover==1))
                self.assertEqual(p['logical_qk_macs'],2147483648)
                self.assertEqual(p['logical_key_read_payload_bytes'],qp*16777216)
                if qp==4:self.assertEqual(p['cube_macs_with_q_alignment'],4294967296)
            self.assertEqual(list(lines),[])
        self.assertEqual(len(controls.POINTS),7)
        self.assertEqual(len(set(controls.POINTS)),7)
        self.assertTrue(all(t==1 or qp==1 for t,qp in controls.POINTS))
        self.assertEqual(controls.geometry(16,1)['score_waves'],1)
        self.assertEqual(controls.geometry(16,1,aics=4,aivs=8)['score_waves'],4)

    def test_02_actual_host_serial_split_and_inactive_q1_guards(self):
        import os
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);(d/'acl').mkdir();(d/'acl/acl.h').write_text(ACL_STUB);(d/'launch.cpp').write_text(LAUNCH_STUB)
            (d/'batch_get_abi.h').write_text('#pragma once\n')  # Context requires no transport declarations.
            cmd=['c++','-std=c++17','-O2','-DLI_QK_RUN_MODE="cpu_stub"','-I'+str(d),'-I'+str(HERE),
                 '-I'+str(ROOT/'csrc/operators/lightning_indexer_with_fetch/include'),
                 '-I'+str(ROOT/'csrc/operators/common/lifetch'),str(HERE/'host.cpp'),str(d/'launch.cpp'),'-o',str(d/'driver')]
            compiled=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(compiled.returncode,0,compiled.stderr)
            fixture=d/'fixture';meta=controls.make_fixture(fixture,1,4096,32)
            env=dict(os.environ,LI_QK_ORACLE=str(fixture/'oracle.bin'),LI_QK_LAUNCH_LOG=str(d/'launch.log'))
            for t,qp in controls.POINTS:
                log=d/'launch.log';log.write_text('');out=d/f'T{t}Q{qp}'
                cmd=[str(d/'driver'),str(fixture),str(out),'4096','32','4','8',str(t),str(qp),'1','2','0']
                r=subprocess.run(cmd,env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr)
                proof=json.loads((out/'report.json').read_text());self.assertEqual(proof['run_mode'],'cpu_stub')
                self.assertIsNone(proof['score_p50_us']);self.assertIsNone(proof['li_p50_us']);self.assertEqual(proof['score_samples_us'],[])
                self.assertEqual(controls.sha(out/'score-keys.bin'),meta['hashes']['oracle'])
                wave=controls.geometry(t,qp,4096,4,8)['score_waves'];calls=log.read_text()
                self.assertEqual(calls.count('S'),6*wave);self.assertEqual(calls.count('K'),3)
                self.assertEqual(calls.count('C'),0 if controls.geometry(t,qp,4096,4,8)['canonical_alias'] else 3*wave)
                self.assertTrue(proof['guards']);self.assertTrue(proof['topk_threshold_oracle'])
            cmd=[str(d/'driver'),str(fixture),str(d/'bad-padding'),'4096','32','4','8','1','4','1','1','0']
            r=subprocess.run(cmd,env=dict(env,LI_QK_CORRUPT_PADDING='1'),capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('padding',r.stderr)

    def test_03_summary_rejects_missing_warmup_and_tampering(self):
        # Synthetic event values test evidence pairing only, not hardware speed.
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);oracle=b'controlled-test-oracle';cases=[]
            for rnd in (0,1):
                for t,qp in controls.POINTS:
                    name=f'r{rnd}_T{t}_Qp{qp}';(d/name).mkdir();argv=['stub',name]
                    item=dict(name=name,round=rnd,history_parts=t,query_parts=qp,argv=argv);cases.append(item)
                    proof=dict(status='PASS',run_mode='npu',batch=1,queries=4,history=131072,heads=32,history_parts=t,query_parts=qp,
                               fetch_launches=0,warmup=4,iterations=2,guards=True,inputs_unchanged=True,topk_threshold_oracle=True,
                               score_p50_us=100,li_p50_us=200,score_samples_us=[90,110],li_samples_us=[190,210],score_waves=1,physical_query_tile=max(2,4//qp))
                    path=d/name/'report.json';path.write_text(json.dumps(proof));(d/name/'score-keys.bin').write_bytes(oracle)
                    log=d/(name+'.log');log.write_text('synthetic test')
                    receipt=dict(exit_code=0,argv=argv,log_sha256=controls.sha(log),report_sha256=controls.sha(path))
                    (d/(name+'.receipt.json')).write_text(json.dumps(receipt))
            manifest=dict(cases=cases,history=131072,warmup=4,iterations=2,rounds=1,fixture=dict(hashes=dict(oracle=hashlib.sha256(oracle).hexdigest())))
            (d/'manifest.json').write_text(json.dumps(manifest));controls.write_summary(d)
            summary=json.loads((d/'summary.json').read_text());self.assertEqual(len(summary),7)
            self.assertTrue(all(r['score_over_serial_mean']==1 for r in summary))
            log=d/'r1_T1_Qp1.log';original=log.read_text();log.write_text('modified')
            with self.assertRaisesRegex(ValueError,'log changed'):controls.write_summary(d)
            log.write_text(original)
            manifest['cases']=cases[:-1]
            (d/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'seven orthogonal'):controls.write_summary(d)
            manifest['cases']=cases
            (d/'manifest.json').write_text(json.dumps(manifest))
            (d/'r0_T1_Qp1.receipt.json').unlink()
            with self.assertRaisesRegex(ValueError,'receipts'):controls.write_summary(d)

    def test_04_scope_and_invalid_geometry(self):
        import re
        launches=set(re.findall(r'aclrtlaunch_\w+', (HERE/'host.cpp').read_text()))
        self.assertEqual(launches,{'aclrtlaunch_li_qk_score','aclrtlaunch_li_qk_scatter','aclrtlaunch_li_qk_topk','aclrtlaunch_li_qk_score_balanced'})
        self.assertIn('-DKVIP_STREAMING_LI=0',(HERE/'CMakeLists.txt').read_text())
        for kw in (dict(t=0,qp=1),dict(t=64,qp=1),dict(t=1,qp=3),dict(t=1,qp=1,aivs=1)):
            with self.assertRaises(ValueError):controls.geometry(**kw)

    def test_05_balanced_cpp_python_and_b16_host(self):
        import os
        for history in (8192,131072):
            p=controls.geometry(2,1,history,30,60,16,'balanced')
            self.assertLessEqual(max(p['core_blocks'])-min(p['core_blocks']),1)
            cover=np.zeros((16,history),np.uint8)
            for r in p['task_map']:
                cover[r['owner'],r['begin']:r['end']]+=1
            self.assertTrue(np.all(cover==1))
            if history==131072:
                self.assertEqual(set(p['core_blocks']),{34,35})
                self.assertEqual(p['active_segments'],60)
        p=controls.geometry(2,1,8192,30,60,16,'balanced')
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);(d/'acl').mkdir();(d/'acl/acl.h').write_text(ACL_STUB);(d/'launch.cpp').write_text(LAUNCH_STUB)
            (d/'batch_get_abi.h').write_text('#pragma once\n')
            subprocess.run(['c++','-std=c++17','-O2','-DLI_QK_RUN_MODE="cpu_stub"','-I'+str(d),'-I'+str(HERE),
                '-I'+str(ROOT/'csrc/operators/lightning_indexer_with_fetch/include'),'-I'+str(ROOT/'csrc/operators/common/lifetch'),
                str(HERE/'host.cpp'),str(d/'launch.cpp'),'-o',str(d/'driver')],check=True,capture_output=True)
            fixture=d/'fixture';meta=controls.make_fixture(fixture,16,8192,32)
            env=dict(os.environ,LI_QK_ORACLE=str(fixture/'oracle.bin'),LI_QK_LAUNCH_LOG=str(d/'calls'),LI_QK_BATCH='16')
            for t,schedule in ((1,'waves'),(2,'waves'),(2,'balanced')):
                out=d/f'T{t}_{schedule}'
                result=subprocess.run([str(d/'driver'),str(fixture),str(out),'8192','32','30','60',str(t),'1','1','1','0','16',schedule],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                report=json.loads((out/'report.json').read_text())
                self.assertEqual(report['batch'],16);self.assertEqual(report['final_topk_aivs'],64)
                self.assertEqual(report['score_launches'],1 if schedule=='balanced' else (2 if t==2 else 1))
                self.assertEqual(controls.sha(out/'score-keys.bin'),meta['hashes']['oracle'])
            code='#include "plan.h"\n#include <iostream>\nint main(int argc,char** argv){auto p=li_qk::MakePlan(std::stoul(argv[1]),32,30,60,2,1,16,true);for(auto t:p.tasks)std::cout<<t.queryBegin<<","<<t.queryCount<<","<<t.begin<<","<<t.end<<","<<t.owner<<"\\n";}'
            (d/'plan.cpp').write_text(code)
            subprocess.run(['c++','-std=c++17','-I'+str(HERE),str(d/'plan.cpp'),'-o',str(d/'plan')],check=True,capture_output=True)
            for history in (8192,131072):
                p=controls.geometry(2,1,history,30,60,16,'balanced')
                actual=[list(map(int,x.split(','))) for x in subprocess.check_output([str(d/'plan'),str(history)],text=True).splitlines()]
                self.assertEqual(actual,[list(t.values()) for t in p['task_map']])
        k32=controls.geometry(32,1)
        self.assertEqual(k32['score_waves'],1);self.assertEqual(k32['key_tokens_per_task'],[4096])

    def test_06_dynamic_issue_denominators_and_fixture_equivalence(self):
        from instructions import analyze
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder)
            lines=[('CUBE','MMAD',10),('MTE2','WAIT_FLAG',9),('SCALAR','ADD',11),('RVECEX','RV_VRELU',12)]
            (d/'core0.cubecore0.instr_log.dump').write_text(''.join(
                f'[info] [00001000] (PC: 0x1000) {pipe} : (Binary: 0x1) (ID: {i:06}) {op} X0\n' for pipe,op,i in lines))
            r=analyze(d)
            self.assertEqual(r['coverage'][0]['instructions'],4)
            self.assertEqual(r['coverage'][0]['id_order_decreases'],1)
            self.assertAlmostEqual(sum(x['share_of_core_instructions'] for x in r['categories']),1)
            self.assertEqual({x['category'] for x in r['categories']},{'cube_math','sync_control','scalar_control_address','vector_math'})
            a=controls.make_fixture(d/'chunked',16,4096,32)
            b=controls.make_small_fixture(d/'all-owner',16,4096,32)
            self.assertEqual(a['hashes'],b['hashes'])

    def test_07_work_grain_and_short_score_coverage(self):
        points=[(h,128) for h in (256,512,1024,2048,4096,8192,131072)]+[(8192,512),(8192,2048)]
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder)
            code='#include "plan.h"\n#include <iostream>\nint main(int argc,char** argv){auto p=li_qk::MakePlan(std::stoul(argv[1]),32,30,60,2,1,16,true,std::stoul(argv[2]),true);for(auto t:p.tasks)std::cout<<t.queryBegin<<","<<t.queryCount<<","<<t.begin<<","<<t.end<<","<<t.owner<<"\\n";}'
            (d/'plan.cpp').write_text(code)
            subprocess.run(['c++','-std=c++17','-I'+str(HERE),str(d/'plan.cpp'),'-o',str(d/'plan')],check=True,capture_output=True)
            for h,grain in points:
                p=controls.geometry(2,1,h,30,60,16,'balanced',grain,True)
                cover=np.zeros((16,h),np.uint8)
                for t in p['task_map']:cover[t['owner'],t['begin']:t['end']]+=1
                self.assertTrue(np.all(cover==1))
                self.assertLessEqual(max(p['core_blocks'])-min(p['core_blocks']),1)
                actual=[list(map(int,x.split(','))) for x in subprocess.check_output([str(d/'plan'),str(h),str(grain)],text=True).splitlines()]
                self.assertEqual(actual,[list(t.values()) for t in p['task_map']])
        with self.assertRaises(ValueError):controls.geometry(2,1,256,30,60,16,'balanced',128)
        with self.assertRaises(ValueError):controls.geometry(2,1,8192,30,60,16,'balanced',192,True)
        with self.assertRaises(ValueError):controls.geometry(1,1,8192,30,60,16,'waves',128)


if __name__=='__main__':unittest.main(verbosity=2)
