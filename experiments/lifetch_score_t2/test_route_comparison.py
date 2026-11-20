# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the direct, shared-pool route comparison."""
from collections import Counter
from contextlib import redirect_stdout
import io
import difflib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_window import ROOT
from workloads import configuration
import route_comparison as comparison
import numpy as np
from pipeline_counts import collect_pipeline_counts, new_pipeline_count_report
from kvip.operators.lifetch.score_policy import score_window_schedule


PARTS = {1:16, 2:16, 4:8, 8:4, 16:2}


def config(root, *extra):
    return configuration(['bench', '--input', 'synthetic', '--transport', 'urma',
        '--device', '4', '--kvaf-root', str(root/'KVAF'), '--kvaf-server', '127.0.0.1:28885',
        '--batches', '1,2,4,8,16', '--miss-percent', '5,10,15,20,25,30',
        '--intermediate-fetches', '4', '--warmup', '2', '--iterations', '10',
        '--timeout-seconds', '3600', '--output', str(root/'results'),
        '--compare-routes', 'mr23,window_budget', *extra])


def option(argv, name):
    return argv[argv.index(name)+1]


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def component_trace(folder,geometry,li_us,calls):
    query=geometry['mode']=='window_query'
    names=[]
    for epoch in range(len(geometry['waves'])):
        names.append('v9_li_kernel_score_window_t2_7_2')
        if epoch==0:names.append('v9_li_kernel_1_2')
        names.extend(['v9_li_window_topk_3','v9_li_window_post_2'] if query else ['v9_li_window_service_2'])
    names.append('v9_li_kernel_3_1')
    duration=li_us/len(names);stride=li_us+10
    events=[dict(name='lifetch_measured_graph',ph='X',ts=1000,dur=stride*calls)]
    samples=[]
    for call in range(calls):
        intervals=[]
        for index,name in enumerate(names):
            begin=call*stride+index*duration
            events.append(dict(name=name,ph='X',cat='kernel',ts=1000+begin,dur=duration))
            intervals.append([begin,begin+duration])
        samples.append(dict(invocation=call,intervals_us=dict(li=intervals),stages_us=dict(li=li_us)))
    write_json(folder/'diagnose/round-1/trace.json',dict(traceEvents=events))
    write_json(folder/'diagnose/round-1/graph_profile.json',dict(samples=samples))


def write_job(job, cfg, *, bad_case=None):
    """Fabricate completed graphs, never an NPU or transport substitute."""
    bench=Path(job['output'])/'recorded-run/cases/bench'
    misses=list(cfg.miss_percent) if isinstance(cfg.miss_percent,(tuple,list)) else [cfg.miss_percent]
    write_json(bench/'pool.json',dict(transport='urma',max_batch=max(cfg.batches),
        runtime_workers=40,put_bytes=max(cfg.batches)*cfg.history*656,
        cases=2*len(cfg.batches)*len(misses),additional_case_put_bytes=0))
    for miss in misses:
        fixture=bench/f'miss{miss:g}'/'fixture'
        fixture.mkdir(parents=True,exist_ok=True)
        write_json(fixture/'fixture.json',dict(history=cfg.history,resident=cfg.resident,seed=cfg.seed))
        for i in range(5):
            (fixture/f'arg{i}.bin').write_bytes(f'operand-{i}-miss{miss:g}'.encode())
        for batch in cfg.batches:
            folder=fixture.parent/f'case-b{batch}'
            okay=(batch,miss)!=bad_case
            write_json(folder/'report.json',dict(batch=batch,miss_percent=miss,
                execution='COMPLETED',correctness='PASS' if okay else 'FAIL'))
            for name in ('serial','fused'):
                fused_base={'mr23':90.,'window_budget':60.,'query_budget':50.}[job['case']]
                event=(100. if name=='serial' else fused_base)+batch+miss
                timing=dict(timing_protocol='one_profiled_graph',event_mean_us=event,p50_us=event-1,
                            stage_mean_us=(dict(li=event-30,resolve_update=20) if name=='fused' else
                                           dict(li=event-30,lidu_prepare=10,fetch=15)))
                result=dict(correctness='PASS' if okay or name=='serial' else 'FAIL',
                            failures=[] if okay else ['fabricated numerical mismatch'],all_rounds=timing)
                if name=='fused':
                    waves=min(cfg.intermediate_fetches,2*batch)
                    lanes=max(1,2*batch//waves)
                    shape=SimpleNamespace(batch=batch,history=cfg.history,resident=cfg.resident,
                        rounds=waves,indexHeads=cfg.index_heads,splitQ=1,liCores=2*batch,
                        aicCount=32,serviceLanes=lanes)
                    query=job['case']=='query_budget'
                    geometry=(None if job['case']=='mr23' else
                              score_window_schedule(shape,job['parts_map'][batch],query))
                    result.update(score_parts=0 if job['case']=='mr23' else job['parts_map'][batch],
                        score_mode='full' if job['case']=='mr23' else 'window_query' if query else 'window',
                        score_geometry=geometry,effective_aic=32,
                        li_schedule=dict(liCores=2*batch,rounds=waves,serviceAics=lanes,
                                         serviceLanes=lanes,finalWorkers=2))
                    if geometry:component_trace(folder/name,geometry,event-30,cfg.iterations)
                write_json(folder/name/'path_result.json',result)
    return bench


class RouteComparison(unittest.TestCase):
    def test_default_plan_keeps_all_batches_in_each_route_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=config(root)
            sources={'mr23':root/'baseline', 'window_budget':ROOT}
            jobs=comparison.build_commands(cfg,root/'run',sources,PARTS)
            self.assertEqual(len(jobs),2)
            self.assertEqual(Counter(j['case'] for j in jobs),{'mr23':1,'window_budget':1})
            self.assertEqual({j['round'] for j in jobs},{1})
            for job in jobs:
                argv=job['args']
                self.assertEqual(option(argv,'--batches'),'1,2,4,8,16')
                self.assertEqual(option(argv,'--miss-percent'),'5,10,15,20,25,30')
                self.assertEqual(option(argv,'--prefetch-tokens'),'512')
                self.assertEqual(option(argv,'--prefetch-rank-start'),'1024')
                for name,value in [('--device','4'),('--intermediate-fetches','4'),
                                   ('--warmup','2'),('--iterations','10'),('--timeout-seconds','3600'),
                                   ('--kvaf-server','127.0.0.1:28885')]:
                    self.assertEqual(option(argv,name),value)
                self.assertNotIn('--compare-routes',argv)
                self.assertNotIn('--comparison-rounds',argv)
                self.assertNotIn('--comparison-discard-first',argv)
                self.assertNotIn('--compare-mr14-serial',argv)
                if job['case']=='mr23':
                    self.assertEqual(Path(argv[2]),sources['mr23']/'tests/operators/lifetch/test_lifetch.py')
                    self.assertNotIn('--li-score-parts-map',argv)
                    self.assertNotIn('--li-score-mode',argv)
                else:
                    self.assertEqual(option(argv,'--li-score-mode'),'window')
                    self.assertEqual(option(argv,'--li-score-parts-map'),'1:16,2:16,4:8,8:4,16:2')

    def test_discard_control_adds_one_whole_route_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=config(root,'--comparison-rounds','3','--comparison-discard-first')
            jobs=comparison.build_commands(cfg,root/'run',{'mr23':root/'baseline','window_budget':ROOT},PARTS)
            self.assertEqual(len(jobs),8)
            self.assertEqual(Counter(j['round'] for j in jobs),{0:2,1:2,2:2,3:2})
            self.assertTrue(all(option(j['args'],'--batches')=='1,2,4,8,16' for j in jobs))

    def test_sampling_overrides_are_shared_by_both_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=config(root,'--prefetch-tokens','1024','--intermediate-fetches','8',
                                  '--warmup','3','--iterations','7','--timeout-seconds','1200')
            jobs=comparison.build_commands(cfg,root/'run',{'mr23':root/'baseline','window_budget':ROOT},PARTS)
            for job in jobs:
                for name,value in [('--prefetch-tokens','1024'),('--intermediate-fetches','8'),
                                   ('--warmup','3'),('--iterations','7'),('--timeout-seconds','1200')]:
                    self.assertEqual(option(job['args'],name),value)


    def paired_results(self, root, cfg):
        jobs=comparison.build_commands(cfg,root,{case:root/'baseline' if case=='mr23' else ROOT for case in cfg.compare_routes},PARTS)
        for job in jobs:
            job['returncode']=0
            write_job(job,cfg)
        return jobs

    def test_summary_requires_completed_graphs_and_preserves_all_requested_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=config(root)
            jobs=self.paired_results(root,cfg)
            with redirect_stdout(io.StringIO()):
                rows=comparison.summarize(root,jobs,cfg)
            self.assertEqual(len(rows),60)
            self.assertTrue(all(row['accepted'] for row in rows))
            candidates=[r for r in rows if r['case']=='window_budget']
            self.assertTrue(all(r['net_gain_vs_mr23_us']==30 for r in candidates))
            for row in candidates:
                self.assertAlmostEqual(row['over_mr23'],row['fused_event_us']/(row['fused_event_us']+30))
                self.assertEqual(row['resolve_fetch_us'],20)
            self.assertEqual(json.loads((root/'summary.json').read_text()),rows)

    def test_fixture_mismatch_rejects_gain_for_each_affected_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=config(root);jobs=self.paired_results(root,cfg)
            candidate=next(j for j in jobs if j['case']=='window_budget')
            path=next(Path(candidate['output']).rglob('miss10/fixture/arg0.bin'))
            path.write_bytes(b'changed operand')
            with redirect_stdout(io.StringIO()):rows=comparison.summarize(root,jobs,cfg)
            affected=[r for r in rows if r['case']=='window_budget' and r['miss_percent']==10]
            self.assertEqual(len(affected),5)
            self.assertTrue(all(not r['accepted'] and r['over_mr23'] is None and
                                r['net_gain_vs_mr23_us'] is None for r in affected))
            self.assertTrue(all('operand mismatch' in r['reason'] for r in affected))
            self.assertTrue(all(r['accepted'] for r in rows if r['miss_percent']!=10))

    def test_pool_capacity_workers_and_case_coverage_must_match(self):
        for change in [dict(max_batch=1),dict(runtime_workers=2),dict(cases=2)]:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg=config(root);jobs=self.paired_results(root,cfg)
                candidate=next(j for j in jobs if j['case']=='window_budget')
                path=next(Path(candidate['output']).rglob('pool.json'))
                pool=json.loads(path.read_text());pool.update(change);write_json(path,pool)
                with redirect_stdout(io.StringIO()):rows=comparison.summarize(root,jobs,cfg)
                affected=[r for r in rows if r['case']=='window_budget']
                self.assertTrue(all(not r['accepted'] and r['net_gain_vs_mr23_us'] is None for r in affected))
                self.assertTrue(all('pool' in r['reason'] for r in affected))

    def test_invalid_graph_event_timing_is_not_a_performance_sample(self):
        for value in [float('nan'),float('inf'),0.,-1.]:
            with self.subTest(value=value),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg=config(root);jobs=self.paired_results(root,cfg)
                candidate=next(j for j in jobs if j['case']=='window_budget')
                path=next(Path(candidate['output']).rglob('miss5/case-b1/fused/path_result.json'))
                result=json.loads(path.read_text());result['all_rounds']['event_mean_us']=value
                write_json(path,result)
                with redirect_stdout(io.StringIO()):rows=comparison.summarize(root,jobs,cfg)
                row=next(r for r in rows if r['case']=='window_budget' and r['batch']==1 and r['miss_percent']==5)
                self.assertFalse(row['accepted']);self.assertIsNone(row['net_gain_vs_mr23_us'])

    def test_baseline_numerical_failure_keeps_case_results_and_runs_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=config(root)
            def execute(job,environment):
                write_job(job,cfg,bad_case=(1,5) if job['case']=='mr23' else None)
                return 1 if job['case']=='mr23' else 0
            with patch.object(comparison,'dependency_preflight',return_value=dict(commit=comparison.KVAF,binaries={})), \
                 patch.object(comparison,'dependency_fingerprint',return_value={}), \
                 patch.object(comparison,'device_preflight',return_value=dict(effective_aic=32)), \
                 patch.object(comparison,'execute_job',side_effect=execute) as launched, \
                 redirect_stdout(io.StringIO()):
                code=comparison.run(cfg)
            self.assertEqual(code,1);self.assertEqual(launched.call_count,2)
            output=next((root/'results').glob('route_comparison_*'))
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['status'],'FAILED');self.assertEqual(len(manifest['completed_jobs']),2)
            rows=json.loads((output/'summary.json').read_text())
            self.assertEqual(len(rows),60)
            bad=[r for r in rows if r['batch']==1 and r['miss_percent']==5]
            self.assertTrue(all(not r['accepted'] and r['net_gain_vs_mr23_us'] is None for r in bad))
            self.assertTrue(all(r['accepted'] for r in rows if (r['batch'],r['miss_percent'])!=(1,5)))
            self.assertEqual(len(list(output.rglob('case-b*/report.json'))),60)

    def test_dependency_rejection_happens_before_device_or_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=config(root)
            with patch.object(comparison,'dependency_preflight',side_effect=RuntimeError('wrong KVAF')), \
                 patch.object(comparison,'device_preflight') as device, \
                 patch.object(comparison,'execute_job') as worker, \
                 redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'wrong KVAF'):
                comparison.run(cfg)
            device.assert_not_called();worker.assert_not_called()
            manifest=json.loads(next((root/'results').rglob('manifest.json')).read_text())
            self.assertEqual(manifest['status'],'FAILED')


    def test_parts_map_survives_worker_json_key_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=config(root,'--li-score-parts-map','1:8,2:8,4:4,8:2,16:1')
            restored=SimpleNamespace(**json.loads(json.dumps(vars(cfg),default=str)))
            parts=comparison.parts_configuration(restored)
            self.assertEqual(parts,{1:8,2:8,4:4,8:2,16:1})
            jobs=comparison.build_commands(restored,root,{'mr23':root/'baseline','window_budget':ROOT},parts)
            candidate=next(j for j in jobs if j['case']=='window_budget')
            self.assertEqual(option(candidate['args'],'--li-score-parts-map'),'1:8,2:8,4:4,8:2,16:1')

    def test_missing_same_graph_stage_does_not_produce_gain(self):
        for stage in ('li','resolve_update'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg=config(root);jobs=self.paired_results(root,cfg)
                candidate=next(j for j in jobs if j['case']=='window_budget')
                path=next(Path(candidate['output']).rglob('miss5/case-b1/fused/path_result.json'))
                result=json.loads(path.read_text());del result['all_rounds']['stage_mean_us'][stage]
                write_json(path,result)
                with redirect_stdout(io.StringIO()):rows=comparison.summarize(root,jobs,cfg)
                row=next(r for r in rows if r['case']=='window_budget' and r['batch']==1 and r['miss_percent']==5)
                self.assertFalse(row['accepted']);self.assertIsNone(row['net_gain_vs_mr23_us'])

    def test_changed_exported_source_invalidates_the_paired_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=config(root)
            def execute(job,environment):
                write_job(job,cfg)
                if job['case']=='mr23':
                    path=Path(job['cwd'])/'csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer.cpp'
                    path.write_bytes(path.read_bytes()+b'\n// changed during comparison\n')
                return 0
            with patch.object(comparison,'dependency_preflight',return_value=dict(commit=comparison.KVAF,binaries={})), \
                 patch.object(comparison,'dependency_fingerprint',return_value={}), \
                 patch.object(comparison,'device_preflight',return_value=dict(effective_aic=32)), \
                 patch.object(comparison,'execute_job',side_effect=execute), \
                 redirect_stdout(io.StringIO()):
                code=comparison.run(cfg)
            self.assertEqual(code,1)
            output=next((root/'results').glob('route_comparison_*'))
            rows=json.loads((output/'summary.json').read_text())
            self.assertEqual(len(rows),60)
            self.assertTrue(all(not r['accepted'] and r['net_gain_vs_mr23_us'] is None for r in rows))
            self.assertEqual(json.loads((output/'manifest.json').read_text())['status'],'FAILED')


    def test_malformed_result_keeps_requested_rows_without_inventing_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=config(root);jobs=self.paired_results(root,cfg)
            candidate=next(j for j in jobs if j['case']=='window_budget')
            path=next(Path(candidate['output']).rglob('miss5/case-b1/fused/path_result.json'))
            path.write_text('{malformed json')
            with redirect_stdout(io.StringIO()):rows=comparison.summarize(root,jobs,cfg)
            self.assertEqual(len(rows),60)
            affected=[r for r in rows if r['case']=='window_budget']
            self.assertTrue(all(not r['accepted'] and r['net_gain_vs_mr23_us'] is None for r in affected))
            self.assertTrue(all(r['accepted'] for r in rows if r['case']=='mr23'))
            self.assertTrue(candidate['collection_error'])
            self.assertEqual(len(list(root.rglob('case-b*/report.json'))),60)


    def query_config(self,root):
        return config(root,'--batches','1,2,4','--compare-routes','mr23,window_budget,query_budget')

    def test_query_plan_has_three_workers_each_with_the_complete_small_batch_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.query_config(root)
            parts=comparison.parts_configuration(cfg)
            self.assertEqual(parts,{1:16,2:16,4:8})
            jobs=comparison.build_commands(cfg,root,{case:root/case for case in cfg.compare_routes},parts)
            self.assertEqual(len(jobs),3)
            self.assertEqual({j['case'] for j in jobs},{'mr23','window_budget','query_budget'})
            for job in jobs:
                self.assertEqual(option(job['args'],'--batches'),'1,2,4')
                for name,value in [('--prefetch-tokens','512'),('--intermediate-fetches','4'),
                                   ('--warmup','2'),('--iterations','10'),('--timeout-seconds','3600')]:
                    self.assertEqual(option(job['args'],name),value)
                if job['case']!='mr23':
                    self.assertEqual(option(job['args'],'--li-score-parts-map'),'1:16,2:16,4:8')
                    self.assertEqual(option(job['args'],'--li-score-mode'),
                                     'window_query' if job['case']=='query_budget' else 'window')
                self.assertNotIn('--compare-routes',job['args'])

    def test_query_components_and_full_chain_gain_are_paired_with_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.query_config(root);jobs=self.paired_results(root,cfg)
            with redirect_stdout(io.StringIO()):rows=comparison.summarize(root,jobs,cfg)
            self.assertEqual(len(rows),54);self.assertTrue(all(r['accepted'] for r in rows))
            for row in [r for r in rows if r['case']=='query_budget']:
                windows=min(4,2*row['batch']);launches=windows*3+2
                self.assertEqual(row['li_kernel_launches'],launches)
                self.assertAlmostEqual(row['query_topk_us'],windows*row['li_us']/launches)
                self.assertAlmostEqual(row['net_gain_vs_window_us'],10)
                self.assertAlmostEqual(row['over_window_budget'],row['fused_event_us']/(row['fused_event_us']+10))
                self.assertAlmostEqual(row['net_gain_vs_mr23_us'],40)
            window=[r for r in rows if r['case']=='window_budget']
            self.assertTrue(all(r['query_topk_us'] is None for r in window))
            self.assertEqual({r['li_kernel_launches'] for r in window},{6,10})

    def test_failed_window_does_not_invalidate_query_against_a_valid_mr23(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.query_config(root);jobs=self.paired_results(root,cfg)
            window=next(j for j in jobs if j['case']=='window_budget')
            write_job(window,cfg,bad_case=(1,5));window['returncode']=1
            with redirect_stdout(io.StringIO()):rows=comparison.summarize(root,jobs,cfg)
            query=next(r for r in rows if r['case']=='query_budget' and r['batch']==1 and r['miss_percent']==5)
            self.assertTrue(query['accepted']);self.assertAlmostEqual(query['net_gain_vs_mr23_us'],40)
            self.assertIsNone(query['over_window_budget']);self.assertIsNone(query['net_gain_vs_window_us'])
            good=next(r for r in rows if r['case']=='query_budget' and r['batch']==2 and r['miss_percent']==5)
            self.assertAlmostEqual(good['net_gain_vs_window_us'],10)

    def test_query_to_window_gain_requires_matching_parts_and_service_schedule(self):
        for mismatch in ('parts','schedule'):
            with self.subTest(mismatch=mismatch),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg=self.query_config(root);jobs=self.paired_results(root,cfg)
                query=next(j for j in jobs if j['case']=='query_budget')
                if mismatch=='parts':
                    query['parts_map']={**query['parts_map'],1:8}
                    write_job(query,cfg)
                else:
                    path=next(Path(query['output']).rglob('miss5/case-b1/fused/path_result.json'))
                    result=json.loads(path.read_text());result['li_schedule']['serviceLanes']+=1
                    write_json(path,result)
                with redirect_stdout(io.StringIO()):rows=comparison.summarize(root,jobs,cfg)
                row=next(r for r in rows if r['case']=='query_budget' and r['batch']==1 and r['miss_percent']==5)
                self.assertTrue(row['accepted']);self.assertIsNotNone(row['over_mr23'])
                self.assertIsNone(row['over_window_budget']);self.assertIsNone(row['net_gain_vs_window_us'])

    def test_query_topk_component_must_exist_in_the_same_measured_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.query_config(root);jobs=self.paired_results(root,cfg)
            query=next(j for j in jobs if j['case']=='query_budget')
            path=next(Path(query['output']).rglob('miss5/case-b1/fused/diagnose/round-1/trace.json'))
            trace=json.loads(path.read_text())
            index=next(i for i,e in enumerate(trace['traceEvents']) if e['name'].startswith('v9_li_window_topk'))
            del trace['traceEvents'][index];write_json(path,trace)
            with redirect_stdout(io.StringIO()):rows=comparison.summarize(root,jobs,cfg)
            row=next(r for r in rows if r['case']=='query_budget' and r['batch']==1 and r['miss_percent']==5)
            self.assertFalse(row['accepted']);self.assertIsNone(row['query_topk_us'])
            self.assertIsNone(row['net_gain_vs_window_us'])


    def test_b124_wrapper_dry_run_uses_unified_bench_and_preserves_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            wrapper=ROOT/'experiments/lifetch_score_t2/run_b124_a5.sh'
            for name,extra,sampling in [('defaults',[],('2','10','3600')),
                                       ('overrides',['--warmup','3','--iterations','7','--timeout-seconds','1200'],('3','7','1200'))]:
                parent=root/name
                result=subprocess.run(['bash',str(wrapper),'--device','0','--kvaf-root',str(root/'not-built'),
                    '--kvaf-server','127.0.0.1:28985','--comparison-dry-run','--output',str(parent),*extra],
                    env=dict(os.environ,PYTHON=sys.executable),capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                output=next(parent.glob('route_comparison_*'))
                jobs=json.loads((output/'jobs.json').read_text())
                manifest=json.loads((output/'manifest.json').read_text())
                self.assertEqual(manifest['status'],'NOT_RUN');self.assertEqual(len(jobs),3)
                self.assertEqual({j['case'] for j in jobs},{'mr23','window_budget','query_budget'})
                for job in jobs:
                    self.assertEqual(option(job['args'],'--batches'),'1,2,4')
                    self.assertEqual(option(job['args'],'--miss-percent'),'5,10,15,20,25,30')
                    for flag,value in [('--device','0'),('--kvaf-server','127.0.0.1:28985'),
                                       ('--prefetch-tokens','512'),('--prefetch-rank-start','1024'),
                                       ('--intermediate-fetches','4'),('--warmup',sampling[0]),
                                       ('--iterations',sampling[1]),('--timeout-seconds',sampling[2])]:
                        self.assertEqual(option(job['args'],flag),value)
                    if job['case']!='mr23':
                        self.assertEqual(option(job['args'],'--li-score-parts-map'),'1:16,2:16,4:8')
                        self.assertEqual(option(job['args'],'--li-score-mode'),
                                         'window_query' if job['case']=='query_budget' else 'window')
                self.assertFalse((output/'summary.csv').exists())


    def test_baseline_reporting_overlay_preserves_fixed_operator_and_timer_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);reference=comparison.reference_module()
            source=reference.baseline(output)
            archive=subprocess.check_output(['git','-C',str(ROOT),'archive',reference.BASE])
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                fixed={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile() and
                       (m.name.startswith('csrc/operators/') or
                        m.name=='tests/operators/lifetch/support/performance.py')}
                original=tar.extractfile('tests/operators/lifetch/support/execution.py').read()
            evidence=comparison.baseline_reporting_overlay(source,output)
            for name,data in fixed.items():
                self.assertEqual((source/name).read_bytes(),data,name)
            effective=(source/evidence['path']).read_bytes()
            self.assertEqual(evidence['original_sha256'],hashlib.sha256(original).hexdigest())
            self.assertEqual(evidence['effective_sha256'],hashlib.sha256(effective).hexdigest())
            helper=(source/evidence['helper']).read_bytes()
            self.assertEqual(helper,(ROOT/evidence['helper']).read_bytes())
            self.assertEqual(evidence['helper_sha256'],hashlib.sha256(helper).hexdigest())
            before=original.decode().splitlines();after=effective.decode().splitlines()
            changes=[c for c in difflib.SequenceMatcher(a=before,b=after).get_opcodes() if c[0]!='equal']
            self.assertEqual([c[0] for c in changes],['insert','insert','replace'])
            first,second,third=changes
            self.assertEqual(before[first[1]-1].strip(),'rounds = {}')
            self.assertEqual(before[second[1]-1].strip(),"validate_all(inputs, 'measured')")
            self.assertTrue(all('pipeline_counts' in line or line.strip()=='if fused:'
                                for change in changes[:2] for line in after[change[3]:change[4]]))
            self.assertEqual(after[third[3]:third[4]],
                             [line.replace('sequence_graph_sum=', 'pipeline_counts=pipeline_counts, sequence_graph_sum=')
                              for line in before[third[1]:third[2]]])
            self.assertEqual(json.loads((output/'baseline_reporting_override.json').read_text()),evidence)

    def measured_counter_report(self,inputs,batch):
        class Readback:
            def __init__(self,array):self.array=array
            def cpu(self):return self
            def numpy(self):return self.array
        states=[]
        for index in range(inputs):
            rows=np.zeros((batch,256),np.int32);posts=np.zeros((batch,2,32),np.int32)
            for request in range(batch):
                rows[request,9]=3*(index+1)+request
                posts[request,0,0]=index+request+1
                posts[request,1,0]=2*request+2
                rows[request,247]=posts[request,:,0].sum()
            states.append(SimpleNamespace(rotating=SimpleNamespace(
                runtime_shape=SimpleNamespace(batch=batch,finalWorkers=2),
                rows=Readback(rows),final_posts=Readback(posts))))
        report=new_pipeline_count_report()
        report['records']=collect_pipeline_counts(states,1)
        return report

    def test_actual_get_counts_sum_requests_per_complete_batch_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=config(root,'--batches','2','--miss-percent','5')
            jobs=self.paired_results(root,cfg)
            report=self.measured_counter_report(10,2)
            self.assertEqual(len(report['records']),20)
            for job in jobs:
                path=next(Path(job['output']).rglob('case-b2/fused/path_result.json'))
                result=json.loads(path.read_text());result['pipeline_counts']=report
                write_json(path,result)
            with redirect_stdout(io.StringIO()):rows=comparison.summarize(root,jobs,cfg)
            self.assertEqual(len(rows),2)
            for row in rows:
                self.assertEqual(row['pipeline_count_status'],'PASS')
                self.assertEqual(row['prefetch_requests_per_call'],34)
                self.assertEqual(row['final_get_requests_per_call'],18)
                self.assertEqual(row['total_get_requests_per_call'],52)
                self.assertNotEqual(row['prefetch_requests_per_call'],cfg.prefetch_tokens)
            window=next(j for j in jobs if j['case']=='window_budget')
            for path in [next(Path(window['output']).rglob('case-b2/report.json')),
                         next(Path(window['output']).rglob('case-b2/fused/path_result.json'))]:
                result=json.loads(path.read_text());result['correctness']='FAIL';write_json(path,result)
            with redirect_stdout(io.StringIO()):rows=comparison.summarize(root,jobs,cfg)
            row=next(r for r in rows if r['case']=='window_budget')
            self.assertFalse(row['accepted']);self.assertIsNone(row['net_gain_vs_mr23_us'])
            self.assertIsNone(row['fused_event_us'])
            self.assertEqual(row['pipeline_count_status'],'PASS')
            self.assertEqual([row[field] for field in comparison.COUNT_FIELDS],[34,18,52])

    def test_missing_or_partial_actual_get_records_are_null_instead_of_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(Path(tmp),'--batches','2','--miss-percent','5')
            partial=self.measured_counter_report(2,2)
            complete=self.measured_counter_report(10,2)
            broken=json.loads(json.dumps(complete));del broken['records'][0]['final_get_requests']
            for result,status in [({},'MISSING'),({'pipeline_counts':partial},'INVALID'),
                                  ({'pipeline_counts':broken},'INVALID')]:
                with self.subTest(status=status,result=result):
                    summary=comparison.pipeline_count_summary(result,cfg,2)
                    self.assertEqual(summary['pipeline_count_status'],status)
                    self.assertTrue(all(summary[field] is None for field in comparison.COUNT_FIELDS))


if __name__=='__main__':
    unittest.main()
