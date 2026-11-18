"""CPU checks of real host/oracle and partition plan; no performance evidence."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import radix
from test_controls import ACL_STUB

HERE=Path(__file__).resolve().parent
STUB=r'''
#include "radix_plan.h"
#include "context.h"
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <vector>
static void choose(uint16_t* scores,int32_t* out,uint32_t b,uint32_t h,uint32_t* ends){
 for(unsigned q=0;q<b*4;++q){std::vector<unsigned> ids(ends[q]);std::iota(ids.begin(),ids.end(),0);
  std::partial_sort(ids.begin(),ids.begin()+2048,ids.end(),[&](unsigned i,unsigned j){return scores[q*h+i]>scores[q*h+j];});
  for(unsigned i=0;i<2048;++i)out[q*2048+i]=ids[i];}
 if(getenv("LI_RADIX_CORRUPT"))out[0]=out[1];
}
extern "C" unsigned aclrtlaunch_li_qk_topk(unsigned,void*,void* score,void* out,void* context,unsigned h,unsigned,unsigned rows){
 auto ctx=(rotating::Context*)context;choose((uint16_t*)score,(int32_t*)out,rows/4,h,(uint32_t*)ctx->queryEnds);return 0;}
extern "C" unsigned aclrtlaunch_li_radix_high(unsigned,void*,void* binding){
 auto a=(li_radix::Bindings*)binding;auto score=(uint16_t*)a->scores;auto ends=(uint32_t*)a->ends;auto high=(uint32_t*)a->high;
 std::fill(high,high+a->batch*4*a->parts*256,0);
 for(unsigned q=0;q<a->batch*4;++q)for(unsigned p=0;p<a->parts;++p)
  for(unsigned x=li_radix::Begin(a->history,a->parts,p);x<std::min(ends[q],li_radix::Begin(a->history,a->parts,p+1));++x)
   ++high[(q*a->parts+p)*256+(score[q*a->history+x]>>8)];
 for(unsigned job=0;job<a->batch*4*a->parts;++job)for(unsigned j=1;j<256;++j)high[job*256+j]+=high[job*256+j-1];return 0;}
extern "C" unsigned aclrtlaunch_li_radix_high_reduce(unsigned,void*,void* binding){
 auto a=(li_radix::Bindings*)binding;auto param=(uint32_t*)a->params;auto high=(uint32_t*)a->high;
 for(unsigned q=0;q<a->batch*4;++q){unsigned before=0;for(int j=255;j>=0;--j){unsigned n=0;for(unsigned p=0;p<a->parts;++p)n+=high[(q*a->parts+p)*256+j]-(j?high[(q*a->parts+p)*256+j-1]:0);
  if(before+n>=2048){param[q*32]=j;param[q*32+1]=2048-before;break;}before+=n;}}return 0;}
extern "C" unsigned aclrtlaunch_li_radix_low(unsigned,void*,void* binding){
 auto a=(li_radix::Bindings*)binding;auto low=(uint32_t*)a->low;auto param=(uint32_t*)a->params;auto score=(uint16_t*)a->scores;auto ends=(uint32_t*)a->ends;
 std::fill(low,low+a->batch*4*a->parts*256,0);
 for(unsigned q=0;q<a->batch*4;++q)for(unsigned p=0;p<a->parts;++p)
  for(unsigned x=li_radix::Begin(a->history,a->parts,p);x<std::min(ends[q],li_radix::Begin(a->history,a->parts,p+1));++x)
   if((score[q*a->history+x]>>8)==param[q*32])++low[(q*a->parts+p)*256+(score[q*a->history+x]&255)];
 for(unsigned job=0;job<a->batch*4*a->parts;++job)for(unsigned j=1;j<256;++j)low[job*256+j]+=low[job*256+j-1];return 0;}
extern "C" unsigned aclrtlaunch_li_radix_low_reduce(unsigned,void*,void* binding){
 auto a=(li_radix::Bindings*)binding;auto low=(uint32_t*)a->low;auto high=(uint32_t*)a->high;auto param=(uint32_t*)a->params;auto off=(uint32_t*)a->offsets;
 for(unsigned q=0;q<a->batch*4;++q){unsigned before=0,lo=0;
  for(int j=255;j>=0;--j){unsigned n=0;for(unsigned p=0;p<a->parts;++p)n+=low[(q*a->parts+p)*256+j]-(j?low[(q*a->parts+p)*256+j-1]:0);if(before+n>=param[q*32+1]){lo=j;break;}before+=n;}
  std::vector<unsigned> gt(a->parts);unsigned total=0;
  for(unsigned p=0;p<a->parts;++p){gt[p]=high[(q*a->parts+p)*256+255]-high[(q*a->parts+p)*256+param[q*32]]+low[(q*a->parts+p)*256+255]-low[(q*a->parts+p)*256+lo];total+=gt[p];}
  param[q*32+2]=param[q*32]*256+lo;param[q*32+3]=total;unsigned gp=0,ep=0;
  for(unsigned p=0;p<a->parts;++p){auto o=off+(q*a->parts+p)*8;unsigned eq=low[(q*a->parts+p)*256+lo]-(lo?low[(q*a->parts+p)*256+lo-1]:0);o[0]=gp;o[1]=ep;o[2]=std::min(2048-total-ep,eq);o[3]=gt[p];gp+=gt[p];ep+=o[2];}}
 return 0;}
extern "C" unsigned aclrtlaunch_li_radix_emit(unsigned,void*,void* binding){
 auto a=(li_radix::Bindings*)binding;choose((uint16_t*)a->scores,(int32_t*)a->selected,a->batch,a->history,(uint32_t*)a->ends);return 0;}
'''

class RadixTests(unittest.TestCase):
    def test_cpp_plan_coverage(self):
        code='#include "radix_plan.h"\n#include <iostream>\nint main(){for(unsigned h:{4096,131072})for(unsigned p:{1,2,4,8,16,30,32})for(unsigned i=0;i<=p;++i)std::cout<<h<<","<<p<<","<<i<<","<<li_radix::Begin(h,p,i)<<"\\n";}'
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);(d/'plan.cpp').write_text(code)
            subprocess.run(['c++','-std=c++17','-I'+str(HERE),str(d/'plan.cpp'),'-o',str(d/'plan')],check=True)
            rows=subprocess.check_output([str(d/'plan')],text=True).splitlines()
            for line in rows:
                h,p,i,x=map(int,line.split(','));self.assertEqual(x,(h//128)*i//p*128)
            for h in (4096,131072):
                for p in (1,2,4,8,16,30,32):
                    boundaries=[int(s.split(',')[3]) for s in rows if s.startswith(f'{h},{p},')]
                    lengths=np.diff(boundaries)
                    self.assertEqual((boundaries[0],boundaries[-1]),(0,h))
                    self.assertGreater(int(lengths.min()),0);self.assertLessEqual(int(lengths.max()-lengths.min()),128)

    def test_actual_host_oracle_with_cpu_launch_stubs(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);(d/'acl').mkdir();(d/'acl/acl.h').write_text(ACL_STUB);(d/'stub.cpp').write_text(STUB);(d/'batch_get_abi.h').write_text('#pragma once\n')
            include=HERE.parents[1]/'csrc/operators/lightning_indexer_with_fetch/include'
            support=HERE.parent/'split_core/model_support'
            subprocess.run(['c++','-std=c++17','-DLI_QK_RUN_MODE="sim"','-I'+str(d),'-I'+str(HERE),'-I'+str(include),'-I'+str(support),str(HERE/'radix_host.cpp'),str(d/'stub.cpp'),'-o',str(d/'host')],check=True)
            for distribution in ('uniform','ties','zeros','skew'):
                fixture=d/distribution;radix.fixture(fixture,1,4096,distribution)
                for route,p in (('original',1),('sharded',1),('sharded',30)):
                    out=d/f'{distribution}_{route}_{p}'
                    cmd=[str(d/'host'),str(fixture),str(out),'1','4096',str(p),route,'0','1','0']
                    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
                    self.assertEqual(json.loads((out/'report.json').read_text())['radix_p50_us'],None)
                bad=d/(distribution+'_bad')
                result=subprocess.run([str(d/'host'),str(fixture),str(bad),'1','4096','4','sharded','0','1','0'],env=dict(os.environ,LI_RADIX_CORRUPT='1'),capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0);self.assertIn('duplicate token',result.stderr)
                self.assertFalse((bad/'report.json').exists())

    def test_budget_and_launch_counts(self):
        cases=radix.definitions();self.assertEqual(len(cases),9)
        for c in cases:
            self.assertLessEqual(c['scan_workers'],64)
            self.assertEqual(c['launches'],1 if c['route']=='original' else 5)
        self.assertEqual(next(c for c in cases if c['name']=='B16_sharded_P4')['max_segments_per_aiv'],4)
        self.assertEqual(next(c for c in cases if c['name']=='B1_sharded_P16')['scan_workers'],64)

    def test_summary_rejects_missing_or_changed_evidence(self):
        # Synthetic receipt checks only; these values are never reported as timings.
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);meta=radix.fixture(d/'fixture_b1',1,4096,'zeros')
            c=radix.definitions(4096)[0];rows=[]
            for rnd in (0,1):
                name=f'r{rnd}_{c["name"]}';dest=d/name;dest.mkdir()
                proof=dict(status='PASS',oracle=True,guards=True,inputs_unchanged=True,batch=1,history=4096,parts=1,route='original',aiv_budget=64,k=2048,score_launches=0,fetch_launches=0,
                    launches=[dict(kernel='li_qk_topk',cores=4)],warmup=4,iterations=1,samples_us=[1.0],radix_p50_us=1.0)
                report=dest/'report.json';report.write_text(json.dumps(proof));log=d/(name+'.log');log.write_text('synthetic fixture\n')
                rows.append(dict(c,config=c['name'],name=name,round=rnd,exit_code=0,report=proof,report_sha256=radix.sha(report),log_sha256=radix.sha(log)))
            m=dict(status='PASS',mode='bench',rows=rows,definitions=[c],rounds=1,iterations=1,warmup=4,history=4096,fixtures={'1':meta})
            manifest=d/'manifest.json';manifest.write_text(json.dumps(m));radix.summarize(d)
            m['rows']=rows[1:];manifest.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'complete planned'):radix.summarize(d)
            m['rows']=rows;manifest.write_text(json.dumps(m))
            (d/'fixture_b1/oracle.bin').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'changed input'):radix.summarize(d)

if __name__=='__main__':unittest.main()
