# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual C++ window plan and MR14 layout against Python metadata."""
import ctypes
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from decimal import Decimal
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
from run_a5 import main, batch_configuration, plan_routes, li_components, save, summarize


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
extern "C" void runtime(unsigned b,unsigned w,unsigned* out) {
 rotating::RuntimeShape s{};s.batch=b;s.history=131072;s.resident=16384;s.rounds=w;
 s.indexHeads=32;s.splitQ=1;s.aicCount=32;s.tokenBytes=656;
 out[0]=rotating::PrepareShape(&s);out[1]=s.rounds;out[2]=s.liCores;
 out[3]=s.serviceAics;out[4]=s.serviceLanes;out[5]=s.finalCores;
}
''')
        compiler=shutil.which('c++');lib=path/'plan.so'
        subprocess.run([compiler,'-std=c++17','-shared','-fPIC','-I',str(ROOT),str(path/'plan.cpp'),'-o',str(lib)],check=True)
        cls.lib=ctypes.CDLL(str(lib))
        cls.lib.wave.argtypes=[ctypes.c_uint]*6+[ctypes.POINTER(ctypes.c_uint)]
        cls.lib.slice.argtypes=[ctypes.c_uint]*4+[ctypes.POINTER(ctypes.c_uint)]
        cls.lib.shard.argtypes=[ctypes.c_uint]*3;cls.lib.shard.restype=ctypes.c_uint
        cls.lib.pack.argtypes=[ctypes.c_uint]*7;cls.lib.pack.restype=ctypes.c_uint
        cls.lib.runtime.argtypes=[ctypes.c_uint]*2+[ctypes.POINTER(ctypes.c_uint)]

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_real_window_partition_matches_metadata_and_ready_cursors(self):
        for b,p,w in [(1,16,2),(1,32,2),(2,16,4),(4,8,4),(8,4,4),
                       (16,2,4),(2,16,8),(4,8,8),(8,4,8),(16,2,8),(16,2,16)]:
            h,r=131072,16384
            runtime=(ctypes.c_uint*6)();self.lib.runtime(b,w,runtime)
            self.assertEqual(runtime[0],0)
            self.assertEqual(runtime[1],min(w,2*b))
            self.assertEqual(runtime[2],2*b)
            self.assertEqual(runtime[3],2*b//runtime[1])
            self.assertEqual(runtime[4],2*runtime[3])
            self.assertEqual(runtime[5],4*b)
            w=runtime[1]
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
            jobs=json.loads((output/'jobs.json').read_text());self.assertEqual(len(jobs),20)
            self.assertEqual([(j['case'],j['batch'],j['mode'],j['parts']) for j in jobs[:10]],
                             [item for b,p in [(1,16),(2,16),(4,8),(8,4),(16,2)] for item in
                              [('mr23',b,'original',0),('window_budget',b,'window',p)]])
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['requested_batches'],[1,2,4,8,16])
            self.assertIn('SerialBefore',manifest['serial_reference'])
            for job in jobs:
                if job['case']=='mr23':self.assertNotIn('--li-score-mode',job['args'])
                else:self.assertIn('window',job['args'])

    def test_budget_controls_and_subbatch_plan(self):
        batches,parts=batch_configuration('2,8','2:8,8:4')
        routes=plan_routes(['mr23','window_budget','full_budget'],batches,parts)
        self.assertEqual(routes,[('mr23','2',0,'original'),('mr23','8',0,'original'),
            ('window_budget','2',8,'window'),('window_budget','8',4,'window'),
            ('full_budget','2',8,'full'),('full_budget','8',4,'full')])
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'
            with redirect_stdout(io.StringIO()):main(['--kvaf-root',temp,'--dry-run','--rounds','1',
                '--batches','2,8','--parts-map','2:8,8:4','--cases','mr23,window_budget,full_budget','--output',str(output)])
            jobs=json.loads((output/'jobs.json').read_text());self.assertEqual(len(jobs),12)
            for j in jobs:
                self.assertEqual(j['args'][j['args'].index('--batches')+1],str(j['batch']))
                if j['case']=='full_budget':self.assertEqual(j['mode'],'full')

    def test_bad_configuration_stops_before_output_creation(self):
        invalid=[['--batches','1,1'],['--batches','3'],['--parts-map','1:16'],
                 ['--parts-map','1:16,2:17,4:8,8:4,16:2'],['--parts-map','1:16,1:8'],
                 ['--parts-map','1:16:2'],['--parts-map','x'],
                 ['--batches','2','--cases','mr23,window_p16']]
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'
            for extra in invalid:
                with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    main(['--kvaf-root',temp,'--dry-run','--output',str(output)]+extra)
                self.assertFalse(output.exists())

    def measured_trace(self, folder, waves=4):
        names=[]
        for epoch in range(waves):
            names.append('v9_li_kernel_score_window_t2_7_2')
            if epoch==0:names.append('v9_li_kernel_1_2')
            names.append('v9_li_window_service_2')
        names += ['v9_li_kernel_3','v9_streaming_resolve_update']
        stride=6*len(names)+4
        origin=Decimal('1000000000000000.013')
        events=[dict(name='lifetch_measured_graph',ph='X',ts=str(origin),dur=2*stride)]
        for call in range(2):
            for i,name in enumerate(names):events.append(dict(name=name,ph='X',cat='kernel',
                ts=str(origin+call*stride+i*6+Decimal('1.125')),dur=5))
        # A CPU launch with the same name must not be charged again.
        events.append(dict(name=names[0],ph='X',cat='cpu_op',ts=str(origin+1),dur=5))
        result=account(events,'lifetch_measured_graph',2*stride,full=True,fused_final=True)
        graph=split_graph_invocations(result,2)
        directory=folder/'diagnose/round-1';directory.mkdir(parents=True)
        save(directory/'trace.json',dict(traceEvents=events));save(directory/'graph_profile.json',graph)
        return graph,events,directory

    def test_same_graph_components_precision_and_kernel_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            graph,events,directory=self.measured_trace(folder)
            geometry=dict(waves=list(range(4)))
            result=li_components(folder,'window',geometry)
            self.assertEqual(result,dict(score_hot_us=5,score_cold_us=15,theta_us=5,service_us=20,final_us=5))
            self.assertEqual(sum(result.values()),graph['timing']['stage_mean_us']['li'])
            self.assertIsNone(li_components(folder,'full',None))
            original=json.loads((directory/'trace.json').read_text())
            events.append(events[1].copy())
            save(directory/'trace.json',dict(traceEvents=events))
            with self.assertRaisesRegex(RuntimeError,'Incomplete/duplicate'):
                li_components(folder,'window',geometry)
            missing=original['traceEvents'];del missing[1]
            save(directory/'trace.json',dict(traceEvents=missing))
            with self.assertRaisesRegex(RuntimeError,'Incomplete/duplicate'):
                li_components(folder,'window',geometry)

    def test_window_summary_components_and_batch_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);save(output/'manifest.json',{})
            jobs=[]
            for b,p in [(2,16),(4,8)]:
                shape=SimpleNamespace(batch=b,history=131072,resident=16384,rounds=4,indexHeads=32,
                                      splitQ=1,liCores=2*b,aicCount=32)
                geometry=score_window_schedule(shape,p)
                for case,parts,mode in [('mr23',0,'original'),('window_budget',p,'window')]:
                    run=output/f'{case}-b{b}';folder=run/'miss10'/f'case-b{b}'
                    folder.mkdir(parents=True);fixture=folder.parent/'fixture';fixture.mkdir()
                    save(fixture/'fixture.json',{});(fixture/'arg0.bin').write_bytes(bytes([b]))
                    save(folder/'report.json',dict(batch=b,miss_percent=10,correctness='PASS',execution='COMPLETED'))
                    (folder/'serial').mkdir()
                    save(folder/'serial/path_result.json',dict(all_rounds=dict(timing_protocol='one_profiled_graph',
                         event_mean_us=100,p50_us=90,stage_mean_us=dict(li=60,lidu_prepare=10,fetch=20))))
                    graph,_,_=self.measured_trace(folder/'fused')
                    timing=graph['timing']
                    save(folder/'fused/path_result.json',dict(all_rounds=timing,
                         score_geometry=geometry if mode=='window' else None,
                         li_schedule=dict(rounds=4,serviceLanes=b,finalWorkers=4)))
                    jobs.append(dict(name=run.name,output=str(run),case=case,batch=b,parts=parts,mode=mode,
                                     pf=1024,round=1,log='unit test only'))
            with redirect_stdout(io.StringIO()):table=summarize(output,jobs,1)
            for row in [r for r in table if r['case']=='window_budget']:
                self.assertEqual(row['score_total_us'],20)
                self.assertEqual(row['score_peak_aic'],32)
                self.assertEqual(row['li_kernel_launches'],10)
                self.assertEqual(row['net_gain_vs_mr23_us'],0)
            self.assertEqual(len(json.loads((output/'fixture_hashes.json').read_text())),2)


if __name__=='__main__':unittest.main()
