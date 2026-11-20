# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the exact bench CLI and batch-specific history producers."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tests/operators/lifetch/support'), str(ROOT/'python')]
from workloads import configuration, score_parts_for_batch
from kvip.operators.lifetch.geometry import li_geometry
from kvip.operators.lifetch.score_policy import score_window_schedule

USER_ARGS = ['bench', '--input', 'synthetic', '--transport', 'urma', '--device', '4',
             '--kvaf-root', './third_party/KVAF', '--kvaf-server', '127.0.0.1:28885',
             '--batches', '1,2,4,8,16', '--miss-percent', '5,10,15,20,25,30',
             '--intermediate-fetches', '4', '--warmup', '2', '--iterations', '10',
             '--timeout-seconds', '3600']
PARTS_MAP = '1:16,2:16,4:8,8:4,16:2'


class DirectScoreParts(unittest.TestCase):
    def reject(self, extra, base=USER_ARGS):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            configuration([*base, *extra])

    def test_original_user_command_keeps_default_route_and_sampling(self):
        c = configuration(USER_ARGS)
        self.assertEqual((c.device, c.kvaf_server), (4, '127.0.0.1:28885'))
        self.assertEqual(c.batches, (1, 2, 4, 8, 16))
        self.assertEqual(c.miss_percent, (5., 10., 15., 20., 25., 30.))
        self.assertEqual((c.history, c.resident, c.queries, c.index_heads, c.seed), (131072, 16384, 4, 32, 43))
        self.assertEqual((c.prefetch_tokens, c.prefetch_rank_start, c.intermediate_fetches), (512, 1024, 4))
        self.assertEqual((c.warmup, c.iterations, c.timeout_seconds), (2, 10, 3600))
        self.assertIsNone(c.compare_routes)
        self.assertIsNone(c.li_score_parts_map)
        self.assertEqual([score_parts_for_batch(c, b) for b in c.batches], [0]*5)

    def test_batch_parts_survive_worker_json_and_match_real_window_contract(self):
        c = configuration([*USER_ARGS, '--li-score-parts-map', PARTS_MAP, '--li-score-mode', 'window'])
        # Internal workers deserialize config JSON: integer map keys become strings.
        serialized = json.loads(json.dumps(vars(c), default=str))
        worker = SimpleNamespace(**serialized)
        self.assertEqual(set(worker.li_score_parts_map), {'1', '2', '4', '8', '16'})
        for b, p, w in [(1, 16, 2), (2, 16, 4), (4, 8, 4), (8, 4, 4), (16, 2, 4)]:
            actual = score_parts_for_batch(worker, b)
            self.assertEqual(actual, p)
            g = li_geometry(b, intermediate_fetches=worker.intermediate_fetches, aic_count=32)
            shape = SimpleNamespace(batch=b, history=worker.history, resident=worker.resident,
                rounds=g['intermediate_fetches'], indexHeads=worker.index_heads, splitQ=1,
                liCores=2*b, aicCount=32, serviceLanes=g['service_aivs_per_window'])
            schedule = score_window_schedule(shape, actual)
            self.assertEqual(len(schedule['waves']), w)
            self.assertEqual(max(wave['score_aics'] for wave in schedule['waves']), b*p)
            self.assertEqual(schedule['li_launches'], 2*w+2)
            for wave in schedule['waves']:
                self.assertLess(wave['begin'], wave['end'])
                self.assertLessEqual(wave['score_aics'], 32)

    def test_scalar_remains_usable_but_cannot_oversubscribe_any_requested_batch(self):
        c = configuration(['bench', '--batches', '1,2', '--li-score-parts', '16'])
        self.assertEqual([score_parts_for_batch(c, b) for b in c.batches], [16, 16])
        self.reject(['--li-score-parts', '16'])
        self.reject(['--li-score-parts-map', PARTS_MAP, '--li-score-parts', '2'])

    def test_map_requires_complete_unique_valid_integer_allocation(self):
        for mapping in ['1:16', '1:16,2:16,4:9,8:4,16:2', '1:16,01:8',
                        '1:0', '0:16', '17:1', '1:33', '1:x', '1:16:2', '1', '']:
            with self.subTest(mapping=mapping):
                self.reject(['--li-score-parts-map', mapping])
        self.reject(['--li-score-parts-map', PARTS_MAP, '--index-heads', '64'])
        self.reject(['--li-score-parts-map', PARTS_MAP, '--transport', 'ubmem'])
        self.reject(['--li-score-parts-map', '1:16'], base=['bench', '--input', 'real',
                    '--trace-root', '/trace', '--group', '0', '--round-range', '1:1'])

    def test_missing_worker_map_entry_does_not_silently_use_scalar_default(self):
        worker = SimpleNamespace(li_score_parts=0, li_score_parts_map={'1':16})
        with self.assertRaisesRegex(ValueError, 'missing batch 2'):
            score_parts_for_batch(worker, 2)

    def test_actual_run_path_binds_batch_parts_to_first_state_and_every_replica(self):
        # Exercise the real host run_path; only device buffers/transport/graph
        # execution are replaced. This checks the selector reaches all ten
        # state constructors and persisted evidence, rather than just the parser.
        import execution
        class Buffer:
            def __init__(self, values): self.values = np.asarray(values)
            def to(self, device): return self
            def contiguous(self): return self
            def cpu(self): return self
            def numpy(self): return self.values
            def __getitem__(self, key):
                return Buffer(self.values[key.values if isinstance(key, Buffer) else key])
        stream = SimpleNamespace(synchronize=lambda: None)
        torch = SimpleNamespace(from_numpy=Buffer, device=lambda value: SimpleNamespace(index=4),
            npu=SimpleNamespace(set_device=lambda value: None, current_stream=lambda device: stream))
        registration = SimpleNamespace(require_a5_dsa_registration=lambda names: None)
        with tempfile.TemporaryDirectory() as tmp:
            for b, p in [(1,16), (2,16), (4,8), (8,4), (16,2)]:
                c = configuration([*USER_ARGS, '--li-score-parts-map', PARTS_MAP, '--li-score-mode', 'window',
                    '--history', '8192', '--resident', '4096'])
                c = SimpleNamespace(**json.loads(json.dumps(vars(c), default=str)))
                c.kvaf_root = Path(c.kvaf_root)
                g = dict(batch=b, history=c.history, resident=c.resident, heads=c.index_heads,
                         token_bytes=656, output_stride=c.resident, sparse_mode=0)
                initial = SimpleNamespace(geometry=g)
                class Sequence:
                    rounds = ((SimpleNamespace(manifest_path=Path(tmp)/'fixture.json'),),)
                    def __iter__(self): yield initial
                page_count = b*c.history//128
                view = SimpleNamespace(addresses=Buffer(np.arange(page_count, dtype=np.uint64)*83968),
                                       valid=Buffer(np.ones(page_count, dtype=np.uint8)))
                pool = SimpleNamespace(workers=40, borrow=lambda *args: SimpleNamespace(
                    runtime=object(), table=object(), device_table=view, cpu_payload=None), release=lambda: None)
                pages = SimpleNamespace(page_indices=np.arange(page_count).reshape(b,-1))
                created = []
                def construct(*args, **options):
                    created.append(options)
                    shape = SimpleNamespace(liCores=2*b, rounds=min(4,2*b), serviceAics=1,
                                            serviceLanes=2, finalWorkers=4)
                    return SimpleNamespace(forward=lambda: None, rotating=SimpleNamespace(runtime_shape=shape),
                        reset=lambda: None, validate=lambda inputs: None, close=lambda: None, lease=object())
                with patch.dict(sys.modules, {'acl':SimpleNamespace(get_soc_name=lambda: 'test_soc'),
                    'kvaf':SimpleNamespace(), 'torch':torch, 'torch_npu':SimpleNamespace()}), \
                        patch('kvip.runtime.build.kvaf_build_paths', return_value=(Path('/cann'),Path('/headers'))), \
                        patch('kvip.runtime.build.build_libraries', return_value=(Path('/library'),)), \
                        patch('kvip.runtime.device.query_device_resources', return_value={'effective_aic':32}), \
                        patch.object(execution, '_load_native_operators', return_value=(torch,registration,None)), \
                        patch.object(execution, 'progress'), patch.object(execution, 'FusedReplayState', side_effect=construct), \
                        patch.object(execution, 'measure_replay_batch', return_value={'correctness':'PASS','failures':[]}):
                    folder = Path(tmp)/f'case-b{b}'
                    result = execution.run_path(c, folder, Sequence(), pages, fused=True, shared_pool=pool)
                self.assertEqual(len(created), 10)
                self.assertTrue(all(item['score_parts']==p and item['score_mode']=='window' for item in created))
                self.assertEqual(result['score_parts'], p)
                evidence = json.loads((folder/'path_result.json').read_text())
                self.assertEqual(evidence['score_parts'], p)
                self.assertEqual(evidence['requested_score_parts_map'], c.li_score_parts_map)

    def test_comparison_flags_need_explicit_supported_comparison_scope(self):
        c = configuration([*USER_ARGS, '--compare-routes', 'mr23,window_budget',
            '--comparison-rounds', '3', '--comparison-discard-first', '--comparison-dry-run'])
        self.assertEqual(c.compare_routes, ('mr23', 'window_budget'))
        self.assertEqual(c.comparison_rounds, 3)
        self.assertTrue(c.comparison_discard_first and c.comparison_dry_run)
        self.assertEqual(c.li_score_mode, 'full')
        self.assertEqual(c.li_score_parts, 0)
        for extra in [['--comparison-rounds', '2'], ['--comparison-discard-first'],
                      ['--comparison-dry-run'], ['--compare-routes', 'mr23'],
                      ['--compare-routes', 'window_budget'], ['--compare-routes', 'mr23,mr23'],
                      ['--compare-routes', 'mr23,unknown'],
                      ['--compare-routes', 'mr23,window_budget', '--comparison-rounds', '0'],
                      ['--compare-routes', 'mr23,window_budget', '--transport', 'ubmem'],
                      ['--compare-routes', 'mr23,window_budget', '--index-heads', '64']]:
            with self.subTest(extra=extra):
                self.reject(extra)
        self.reject(['--compare-routes', 'mr23,window_budget'], base=['bench', '--input', 'real',
                    '--trace-root', '/trace', '--group', '0', '--round-range', '1:1'])
        self.reject(['--compare-routes', 'mr23,window_budget'], base=['diagnose'])

    def test_aligned_comparison_requires_zero_prefetch(self):
        self.reject(['--compare-routes', 'mr23,window_budget,aligned_budget'])
        c = configuration([*USER_ARGS, '--compare-routes', 'mr23,aligned_budget', '--prefetch-tokens', '0'])
        self.assertEqual(c.compare_routes, ('mr23', 'aligned_budget'))
        self.assertEqual(c.prefetch_tokens, 0)

    def test_three_small_batch_routes_keep_user_prefetch_and_graph_configuration(self):
        c = configuration([*USER_ARGS, '--batches', '1,2,4',
            '--compare-routes', 'mr23,window_budget,query_budget',
            '--li-score-parts-map', '1:16,2:16,4:8', '--comparison-dry-run'])
        self.assertEqual(c.batches, (1,2,4))
        self.assertEqual(c.compare_routes, ('mr23','window_budget','query_budget'))
        self.assertEqual((c.prefetch_tokens,c.intermediate_fetches,c.warmup,c.iterations), (512,4,2,10))
        self.assertEqual([score_parts_for_batch(c,b) for b in c.batches], [16,16,8])
        self.assertEqual(c.li_score_mode, 'full')
        for b in c.batches:
            g = li_geometry(b, intermediate_fetches=c.intermediate_fetches, aic_count=32)
            shape = SimpleNamespace(batch=b, history=c.history, resident=c.resident,
                rounds=g['intermediate_fetches'], indexHeads=c.index_heads, splitQ=1,
                liCores=2*b, aicCount=32, serviceLanes=g['service_aivs_per_window'])
            query = score_window_schedule(shape, score_parts_for_batch(c,b), query_parallel=True)
            self.assertEqual(query['mode'], 'window_query')
            self.assertEqual(query['topk_aivs'], 4*b)
            self.assertEqual(query['li_launches'], 3*shape.rounds+2)

    def test_mr23_and_query_only_comparison_does_not_require_window_route(self):
        c = configuration([*USER_ARGS, '--batches', '1,2,4',
            '--compare-routes', 'mr23,query_budget'])
        self.assertEqual(c.compare_routes, ('mr23','query_budget'))
        self.assertEqual((c.prefetch_tokens,c.intermediate_fetches,c.warmup,c.iterations), (512,4,2,10))
        self.assertEqual(c.li_score_parts, 0)
        self.assertIsNone(c.li_score_parts_map)

    def test_query_zero_prefetch_ablation_is_valid(self):
        c = configuration([*USER_ARGS, '--batches','1,2,4',
            '--compare-routes','mr23,window_budget,query_budget', '--prefetch-tokens','0'])
        self.assertEqual(c.compare_routes, ('mr23','window_budget','query_budget'))
        self.assertEqual(c.prefetch_tokens, 0)
        self.assertEqual((c.intermediate_fetches,c.warmup,c.iterations), (4,2,10))

    def test_query_comparison_requires_the_same_supported_input_scope(self):
        for extra in [['--transport','ubmem'],
                      ['--index-heads','64'], ['--queries','2']]:
            with self.subTest(extra=extra):
                self.reject(['--compare-routes','mr23,query_budget',*extra])
        self.reject(['--compare-routes','query_budget'])
        self.reject(['--compare-routes','mr23,query_budget,query_budget'])
        self.reject(['--compare-routes','mr23,query_budget'], base=['bench','--input','real',
                    '--trace-root','/trace','--group','0','--round-range','1:1'])
        self.reject(['--compare-routes','mr23,query_budget'], base=['diagnose'])


if __name__ == '__main__':
    unittest.main()
