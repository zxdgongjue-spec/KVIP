# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the direct, shared-pool route comparison."""
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_window import ROOT
from workloads import configuration
import route_comparison as comparison


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
                event=100.+batch+miss if name=='serial' else (90. if job['case']=='mr23' else 60.)+batch+miss
                timing=dict(timing_protocol='one_profiled_graph',event_mean_us=event,p50_us=event-1,
                            stage_mean_us=(dict(li=event-30,resolve_update=20) if name=='fused' else
                                           dict(li=event-30,lidu_prepare=10,fetch=15)))
                result=dict(correctness='PASS' if okay or name=='serial' else 'FAIL',
                            failures=[] if okay else ['fabricated numerical mismatch'],all_rounds=timing)
                if name=='fused':
                    result.update(score_parts=0 if job['case']=='mr23' else job['parts_map'][batch],
                                  score_mode='full' if job['case']=='mr23' else 'window',
                                  score_geometry=None if job['case']=='mr23' else dict(mode='window',waves=[dict(parts=PARTS[batch])]),
                                  effective_aic=32)
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
        jobs=comparison.build_commands(cfg,root,{'mr23':root/'baseline','window_budget':ROOT},PARTS)
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


if __name__=='__main__':
    unittest.main()
