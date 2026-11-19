# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual C++ window plan and MR14 layout against Python metadata."""
import ctypes
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tests/operators/lifetch/support')]
from kvip.operators.lifetch.score_policy import score_window_schedule
from diagnostics import account
from performance import split_graph_invocations
from workloads import configuration
from run_a5 import main


class Windows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(); path=Path(cls.temp.name)
        (path/'plan.cpp').write_text('''
#include "csrc/operators/lightning_indexer_with_fetch/include/runtime_shape.h"
namespace rotating {
#include "csrc/operators/common/lifetch/sequence_schedule.h"
}
#include "csrc/operators/common/lifetch/score_window_plan.h"
#include "csrc/operators/common/lifetch/staged_post.h"
extern "C" void wave(unsigned b,unsigned h,unsigned r,unsigned w,unsigned p,unsigned epoch,unsigned* out) {
 rotating::RuntimeShape shape{};shape.batch=b;shape.history=h;shape.resident=r;shape.rounds=w;
 auto s=kvip_score_t2::PlanWindow(shape,epoch,p);
 out[0]=s.begin;out[1]=s.end;out[2]=s.end0;out[3]=s.end1;out[4]=s.parts;
}
extern "C" void slice(unsigned task,unsigned p,unsigned start,unsigned end,unsigned* out) {
 auto s=kvip_score_t2::WindowPartition(task,p,start,end);out[0]=s.owner;out[1]=s.beginTile*128;out[2]=s.endTile*128;
}
extern "C" unsigned shard(unsigned token,unsigned r,unsigned half) {return rotating::SequenceSliceAt(token,r,half).shard;}
extern "C" unsigned pack(unsigned a,unsigned b,unsigned c,unsigned d,unsigned theta,unsigned token,unsigned end) {
 return kvip_staged_post::Pack4(a,b,c,d,theta,token,end);
}
''')
        compiler=shutil.which('c++');lib=path/'plan.so'
        subprocess.run([compiler,'-std=c++17','-shared','-fPIC','-I',str(ROOT),str(path/'plan.cpp'),'-o',str(lib)],check=True)
        cls.lib=ctypes.CDLL(str(lib))
        cls.lib.wave.argtypes=[ctypes.c_uint]*6+[ctypes.POINTER(ctypes.c_uint)]
        cls.lib.slice.argtypes=[ctypes.c_uint]*4+[ctypes.POINTER(ctypes.c_uint)]
        cls.lib.shard.argtypes=[ctypes.c_uint]*3;cls.lib.shard.restype=ctypes.c_uint
        cls.lib.pack.argtypes=[ctypes.c_uint]*7;cls.lib.pack.restype=ctypes.c_uint

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_real_window_partition_matches_metadata_and_ready_cursors(self):
        for b,p,w in [(1,16,2),(1,32,2),(16,2,4),(16,2,8),(16,2,16)]:
            h,r=131072,16384
            shape=SimpleNamespace(batch=b,history=h,resident=r,rounds=w,indexHeads=32,splitQ=1,liCores=2*b,aicCount=32)
            plan=score_window_schedule(shape,p); previous=0
            for item in plan['waves']:
                values=(ctypes.c_uint*5)();self.lib.wave(b,h,r,w,p,item['epoch'],values)
                self.assertEqual(tuple(values),(item['begin'],item['end'],item['ready_end0'],item['ready_end1'],item['parts']))
                self.assertEqual(item['begin'],previous);previous=item['end']
                self.assertLessEqual(item['score_aics'],32)
                self.assertEqual(item['predicate'],item['begin']>=r)
                owner_ranges={owner:[] for owner in range(b)}
                for task in range(b*item['parts']):
                    out=(ctypes.c_uint*3)();self.lib.slice(task,item['parts'],item['begin'],item['end'],out)
                    owner,start,end=out;self.assertEqual(start%256,0);self.assertEqual(end%256,0)
                    self.assertLess(start,end);owner_ranges[owner].append((start,end))
                for ranges in owner_ranges.values():
                    cursor=item['begin'];lengths=[]
                    for start,end in ranges:
                        self.assertEqual(start,cursor);cursor=end;lengths.append(end-start)
                    self.assertEqual(cursor,item['end']);self.assertLessEqual(max(lengths)-min(lengths),256)
                if item['epoch']:
                    # Every token that the original T2 consumer can scan up to
                    # cap has been produced, including both alternating halves.
                    for token in range(r,item['end'],256):
                        shard=self.lib.shard(token,r,plan['half_window'])
                        ready=item['ready_end0'] if shard==0 else item['ready_end1']
                        self.assertGreater(ready,token)
            self.assertEqual(previous,h);self.assertEqual(plan['li_launches'],2*w+2)
            self.assertEqual(plan['scatter_launches'],0)

    def test_mr14_predicate_pack_strict_tie_and_valid_end(self):
        self.assertEqual(self.lib.pack(9,10,11,12,10,100,104),16|64)
        self.assertEqual(self.lib.pack(11,12,13,14,10,100,102),1|4)

    def test_whole_graph_accounts_for_all_new_launches(self):
        for w in (2,4):
            names=[]
            for epoch in range(w):
                names.append('v9_li_kernel_score_window_t2_7')
                if epoch==0:names.append('v9_li_kernel_1')
                names.append('v9_li_window_service_2')
            names += ['v9_li_kernel_3','v9_streaming_resolve_update']
            stride=6*len(names)+4
            events=[dict(name='replay',ph='X',ts=0,dur=2*stride)]
            for call in range(2):
                for i,name in enumerate(names):events.append(dict(name=name,ph='X',cat='kernel',ts=call*stride+i*6+1,dur=5))
            result=account(events,'replay',2*stride,full=True,fused_final=True)
            self.assertNotIn('error',result)
            self.assertEqual(len(result['intervals_us']['li']),2*(2*w+2))
            graph=split_graph_invocations(result,2)
            self.assertEqual(graph['timing']['stage_mean_us']['li'],5*(2*w+2))

    def test_window_cli_rejects_unvalidated_replay_and_no_parts(self):
        good=configuration(['bench','--li-score-parts','16','--li-score-mode','window','--batches','1'])
        self.assertEqual(good.li_score_mode,'window')
        for args in [['bench','--li-score-mode','window'],['bench','--li-score-parts','2','--li-score-mode','window','--input','replay']]:
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):configuration(args)

    def test_runner_defaults_to_integrated_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'
            with redirect_stdout(io.StringIO()):main(['--kvaf-root',temp,'--dry-run','--rounds','1','--output',str(output)])
            jobs=json.loads((output/'jobs.json').read_text());self.assertEqual(len(jobs),8)
            self.assertEqual([(j['case'],j['batch'],j['mode'],j['parts']) for j in jobs[:4]],
                             [('mr23',1,'original',0),('mr23',16,'original',0),
                              ('window_p16',1,'window',16),('window_p2',16,'window',2)])
            for job in jobs:
                if job['case']=='mr23':self.assertNotIn('--li-score-mode',job['args'])
                else:self.assertIn('window',job['args'])


if __name__=='__main__':unittest.main()
