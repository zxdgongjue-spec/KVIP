#!/usr/bin/env python3
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tail import plan
from tail_summary import collect
from tail_instructions import classify_tail
import json

HERE=Path(__file__).resolve().parent

class TailControls(unittest.TestCase):
    def test_cpp_python_validity_and_history_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'valid.cpp').write_text('#include "tail_plan.h"\n#include <iostream>\nint main(){unsigned b,h,r,p,rp;while(std::cin>>b>>h>>r>>p>>rp)std::cout<<li_tail::Valid(b,h,r,p,rp)<<"\\n";}\n')
            subprocess.run(['c++','-std=c++17','-I',str(HERE),str(d/'valid.cpp'),'-o',str(d/'valid')],check=True)
            points=[(b,h,r,p,rp) for b in (1,16,2) for h,r in ((131072,16384),(32768,8192),(8192,4096),(4096,2048),(4096,4096)) for p in (1,2,4,8,16,32) for rp in (1,2,4)]
            actual=subprocess.check_output([str(d/'valid')],input=''.join(' '.join(map(str,x))+'\n' for x in points),text=True).splitlines()
            for point,got in zip(points,actual):
                a=SimpleNamespace(**dict(zip(('batch','history','resident','parts','resident_parts'),point)),controls=['C0','C1','C2','C3'])
                try:rows=plan(a);valid=True
                except ValueError:valid=False
                self.assertEqual(valid,got=='1',point)
                if valid:
                    blocks=(a.history-a.resident)//256
                    intervals=[(a.resident+blocks*i//a.parts*256,a.resident+blocks*(i+1)//a.parts*256) for i in range(a.parts)]
                    self.assertEqual(intervals[0][0],a.resident);self.assertEqual(intervals[-1][1],a.history)
                    self.assertTrue(all(x[1]==y[0] for x,y in zip(intervals,intervals[1:])))

    def test_local_topk_exact_union_ties_and_invalid_tail(self):
        rng=np.random.default_rng(14);k=2048
        for h,p in ((8192,4),(32768,4),(131072,8)):
            scores=rng.integers(32768,33000,size=h,dtype=np.uint16)
            for valid in (h,h-3):
                candidates=[]
                for part in range(p):
                    begin=part*(h//p);end=min(valid,(part+1)*(h//p))
                    ix=np.arange(begin,max(begin,end));chosen=ix[np.argsort(scores[ix],kind='stable')[::-1][:k]];candidates.extend(chosen)
                ids=np.asarray(candidates);selected=ids[np.argsort(scores[ids],kind='stable')[::-1][:k]]
                theta=np.partition(scores[:valid],valid-k)[valid-k]
                self.assertEqual(len(np.unique(selected)),k)
                self.assertTrue(np.all(scores[selected]>=theta))
                self.assertTrue(set(np.flatnonzero(scores[:valid]>theta))<=set(selected))

    def test_resident_local_theta_is_not_global_theta(self):
        scores=np.concatenate((np.arange(1,4097),np.arange(10001,14097)))
        local=[np.partition(x,len(x)-2048)[len(x)-2048] for x in np.split(scores,2)]
        global_theta=np.partition(scores,len(scores)-2048)[len(scores)-2048]
        self.assertNotEqual(local[0],global_theta)
        candidate=np.concatenate([np.sort(x)[-2048:] for x in np.split(scores,2)])
        self.assertEqual(np.partition(candidate,len(candidate)-2048)[len(candidate)-2048],global_theta)

    def test_shape_boundaries_and_scope(self):
        a=SimpleNamespace(batch=16,history=131072,resident=16384,parts=2,resident_parts=1,controls=['C0','C1','C2','C3'])
        rows=plan(a);self.assertEqual([x['cold_aiv_jobs'] for x in rows],[16,64,16,64])
        self.assertEqual([x['final_local_aiv_jobs'] for x in rows],[64,64,128,128])
        self.assertTrue(all(x['aiv_budget']==64 and 'no QK/Fetch' in x['scope'] for x in rows))

    def test_summary_rejects_incomplete_and_isolates_fixture_binary(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folders=[]
            geometry=dict(batch=1,history=32768,resident=8192,parts=4,resident_parts=1,stage='all',cold_parallel=0,final_split=0)
            for i,(control,binary,fixture,ticks) in enumerate((('C0','a','x',100),('C1','a','x',40),('C1','b','x',35),('C1','a','y',45))):
                f=root/str(i);f.mkdir();folders.append(f)
                r=dict(geometry,name=control+'_all',status='MODEL_PASSED',exit_code=0,whole_model_ticks=ticks,issued_instructions=100)
                r['cold_parallel']=int(control=='C1')
                r['report']=dict(geometry,cold_parallel=r['cold_parallel'],status='PASS',tail_p50_us=None,oracle=True,guards=True,inputs_unchanged=True,fetch_launches=0)
                m=dict(status='MODEL_PASSED',logged_core_mask='0xffffffff',build_manifest=dict(binary_sha256=binary),fixture=dict(hashes=dict(oracle=fixture)),rows=[r])
                (f/'manifest.json').write_text(json.dumps(m))
            result=collect(folders)['rows']
            self.assertEqual([x['ticks_over_C0'] for x in result],[1.0,0.4,None,None])
            m=json.loads((folders[1]/'manifest.json').read_text());m['status']='RUNNING';(folders[1]/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):collect(folders)

    def test_resident_candidates_and_cold_gt_theta_cover_final(self):
        rng=np.random.default_rng(914);k=2048;r=4096;h=8192
        for tied in (False,True):
            scores=rng.integers(32768,32800 if tied else 60000,size=h,dtype=np.uint16)
            resident=np.argsort(scores[:r],kind='stable')[::-1][:k];theta=scores[resident].min()
            candidates=np.concatenate((resident,np.flatnonzero(scores[r:]>theta)+r))
            selected=candidates[np.argsort(scores[candidates],kind='stable')[::-1][:k]]
            tau=np.partition(scores,h-k)[h-k]
            self.assertTrue(np.all(scores[selected]>=tau))
            self.assertTrue(set(np.flatnonzero(scores>tau))<=set(selected))

    def test_instruction_category_uses_opcode_for_gm_memory(self):
        self.assertEqual(classify_tail('RVECLD','SIMT_LDG'),'global_memory_load_store')
        self.assertEqual(classify_tail('RVECST','SIMT_STG'),'global_memory_load_store')
        self.assertEqual(classify_tail('RVECEX','SIMT_IADD_X_I'),'integer_arithmetic_addressing')
        self.assertEqual(classify_tail('RVECEX','SIMT_ISETP'),'predicate_integer_logic')
        self.assertEqual(classify_tail('ALL','BAR'),'sync_control')

if __name__=='__main__':unittest.main(verbosity=2)
