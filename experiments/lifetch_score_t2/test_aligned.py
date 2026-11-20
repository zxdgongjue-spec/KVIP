# SPDX-License-Identifier: Apache-2.0
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from test_window import ROOT
from kvip.operators.lifetch.score_policy import score_aligned_schedule
from workloads import configuration
from diagnostics import account
from performance import split_graph_invocations
from run_a5 import li_components, plan_routes, DEFAULT_PARTS, DEFAULT_MISSES, ROUTE_DESCRIPTIONS, main, summarize, save


class Aligned(unittest.TestCase):
    def test_three_integrated_routes_plan_sixty_jobs_with_default_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            with redirect_stdout(io.StringIO()):
                main(['--kvaf-root',tmp,'--dry-run','--cases','mr23,window_budget,aligned_budget',
                      '--prefetch','0','--output',str(output)])
            jobs=json.loads((output/'jobs.json').read_text())
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(len(jobs),60)
            self.assertEqual(manifest['requested_batches'],[1,2,4,8,16])
            self.assertEqual(manifest['configuration']['miss_percent'],'5,10,15,20,25,30')
            self.assertEqual(set(manifest['route_descriptions']),{'mr23','window_budget','aligned_budget'})
            self.assertIn('algorithm/schedule alignment',manifest['integration'])
            self.assertIn('integrated MR14 candidates are Fused rows',manifest['serial_reference'])
            for round_id in range(4):
                round_jobs=[j for j in jobs if j['round']==round_id]
                self.assertEqual(len(round_jobs),15)
                self.assertEqual({(j['case'],j['batch']) for j in round_jobs},
                                 {(case,b) for case in ROUTE_DESCRIPTIONS for b in DEFAULT_PARTS})
            for job in jobs:
                self.assertEqual(job['args'][job['args'].index('--miss-percent')+1],DEFAULT_MISSES)
                self.assertEqual(job['args'][job['args'].index('--batches')+1],str(job['batch']))

    def test_extended_miss_percentages_and_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            with redirect_stdout(io.StringIO()):
                main(['--kvaf-root',tmp,'--dry-run','--cases','mr23','--batches','1',
                      '--miss-percent','0,25,30,100','--rounds','1','--output',str(output)])
            jobs=json.loads((output/'jobs.json').read_text())
            self.assertEqual(len(jobs),2)
            self.assertTrue(all(j['args'][j['args'].index('--miss-percent')+1]=='0,25,30,100' for j in jobs))

    def test_invalid_miss_percentages_stop_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            for misses in ['nan','inf','-inf','-1','101','5,5','5,5.0','']:
                with self.subTest(misses=misses),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    main(['--kvaf-root',tmp,'--dry-run','--output',str(output),'--miss-percent='+misses])
                self.assertFalse(output.exists())

    def test_aligned_wrapper_defaults_remain_overrideable(self):
        wrapper=ROOT/'experiments/lifetch_score_t2/run_aligned_a5.sh'
        with tempfile.TemporaryDirectory() as tmp:
            for name,extra,expected_batches,expected_misses,expected_count in [
                ('defaults',[],[1,2,4,8,16],DEFAULT_MISSES,60),
                ('override',['--batches','2,4','--miss-percent','0,10','--rounds','1'],[2,4],'0,10',12),
            ]:
                output=Path(tmp)/name
                result=subprocess.run(['bash',str(wrapper),'--kvaf-root',tmp,'--dry-run','--output',str(output)]+extra,
                                      env=dict(os.environ,PYTHON=sys.executable),capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                jobs=json.loads((output/'jobs.json').read_text())
                manifest=json.loads((output/'manifest.json').read_text())
                self.assertEqual(len(jobs),expected_count)
                self.assertEqual(manifest['requested_batches'],expected_batches)
                self.assertEqual(manifest['configuration']['miss_percent'],expected_misses)
                self.assertEqual(manifest['configuration']['prefetch'],'0')

    def test_geometry_and_prefetch_guard(self):
        for b,p in DEFAULT_PARTS.items():
            s=SimpleNamespace(batch=b,history=131072,resident=16384,rounds=min(4,2*b),
                              indexHeads=32,splitQ=1,liCores=2*b,aicCount=32)
            g=score_aligned_schedule(s,p,0)
            self.assertEqual(g['topk_aivs'],4*b)
            self.assertEqual(g['actual_windows'],0)
            self.assertEqual(g['li_launches'],2)
            self.assertEqual(g['waves'][0]['end'],131072)
            with self.assertRaises(ValueError):score_aligned_schedule(s,p,1024)
        config=configuration(['bench','--batches','2','--li-score-parts','16',
                              '--li-score-mode','aligned','--prefetch-tokens','0'])
        self.assertEqual(config.li_score_mode,'aligned')
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            configuration(['bench','--li-score-parts','16','--li-score-mode','aligned'])
        self.assertEqual(plan_routes(['aligned_budget'],list(DEFAULT_PARTS),DEFAULT_PARTS),
                         [('aligned_budget',str(b),p,'aligned') for b,p in DEFAULT_PARTS.items()])

    def test_same_graph_counts_full_score_topk_and_completed_fetch(self):
        events=[dict(name='lifetch_measured_graph',ph='X',ts=0,dur=100)]
        for call in range(2):
            for i,(name,duration) in enumerate([('v9_li_kernel_score_aligned_7',5),
                    ('v9_li_full_topk_8',6),('v9_streaming_resolve_update',7)]):
                events.append(dict(name=name,ph='X',cat='kernel',ts=call*40+i*10+1,dur=duration))
        result=account(events,'lifetch_measured_graph',100,full=True,fused_final=True)
        self.assertNotIn('error',result)
        graph=split_graph_invocations(result,2)
        self.assertEqual(graph['timing']['stage_mean_us']['li'],11)
        self.assertEqual(graph['timing']['stage_mean_us']['resolve_update'],7)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);out=folder/'diagnose/round-1';out.mkdir(parents=True)
            (out/'trace.json').write_text(json.dumps(dict(traceEvents=events)))
            (out/'graph_profile.json').write_text(json.dumps(graph))
            parts=li_components(folder,'aligned',dict(waves=[{}]))
            self.assertEqual(parts,dict(score_full_us=5,full_topk_classify_us=6))
            events.append(dict(name='v9_li_full_topk_8',ph='X',cat='kernel',ts=11,dur=6))
            (out/'trace.json').write_text(json.dumps(dict(traceEvents=events)))
            with self.assertRaisesRegex(RuntimeError,'duplicate'):
                li_components(folder,'aligned',dict(waves=[{}]))

    def test_summary_preserves_alignment_and_completed_tail(self):
        shape=SimpleNamespace(batch=2,history=131072,resident=16384,rounds=4,
                              indexHeads=32,splitQ=1,liCores=4,aicCount=32)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'manifest.json',{});jobs=[]
            for case,mode,parts in [('mr23','original',0),('aligned_budget','aligned',16)]:
                folder=root/case/'miss10/case-b2';folder.mkdir(parents=True)
                fixture=folder.parent/'fixture';fixture.mkdir();save(fixture/'fixture.json',{})
                (fixture/'arg0.bin').write_bytes(b'identical')
                save(folder/'report.json',dict(batch=2,miss_percent=10,execution='COMPLETED',correctness='PASS'))
                (folder/'serial').mkdir();(folder/'fused').mkdir()
                events=[dict(name='lifetch_measured_graph',ph='X',ts=0,dur=100)]
                names=[('v9_li_kernel_score_aligned',5),('v9_li_full_topk',6)] if mode=='aligned' else [('v9_li_kernel',20)]
                for i,(name,duration) in enumerate(names+[('v9_streaming_resolve_update',7)]):
                    events.append(dict(name=name,ph='X',cat='kernel',ts=i*25+1,dur=duration))
                graph=split_graph_invocations(account(events,'lifetch_measured_graph',100,full=True,fused_final=True),1)
                if mode=='aligned':
                    out=folder/'fused/diagnose/round-1';out.mkdir(parents=True)
                    save(out/'trace.json',dict(traceEvents=events));save(out/'graph_profile.json',graph)
                save(folder/'fused/path_result.json',dict(all_rounds=graph['timing'],
                     score_geometry=score_aligned_schedule(shape,16,0) if parts else None,
                     li_schedule=dict(rounds=4,serviceLanes=2,finalWorkers=4)))
                save(folder/'serial/path_result.json',dict(all_rounds=dict(timing_protocol='one_profiled_graph',
                     event_mean_us=100,p50_us=90,stage_mean_us=dict(li=60,lidu_prepare=10,fetch=20))))
                jobs.append(dict(round=1,case=case,name=case,pf=0,batch=2,mode=mode,parts=parts,output=str(root/case),log='test'))
            with redirect_stdout(io.StringIO()):rows=summarize(root,jobs,1)
            row=next(r for r in rows if r['case']=='aligned_budget')
            self.assertEqual(row['route_description'],ROUTE_DESCRIPTIONS['aligned_budget'])
            self.assertEqual(row['actual_windows'],0);self.assertEqual(row['service_aivs'],0)
            self.assertEqual(row['score_total_us'],5);self.assertEqual(row['full_topk_classify_us'],6)
            self.assertEqual(row['li_kernel_launches'],2);self.assertEqual(row['resolve_fetch_us'],7)


if __name__=='__main__':unittest.main()
