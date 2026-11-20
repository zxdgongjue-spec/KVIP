# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for readback from completed measured states, outside timing."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tests/operators/lifetch/support'), str(ROOT/'python')]
from pipeline_counts import collect_pipeline_counts, new_pipeline_count_report


class Buffer:
    def __init__(self, values, read=None): self.values, self.read = values, read
    def cpu(self):
        if self.read: self.read()
        return self
    def numpy(self): return self.values


def state_counts(prefetch, final):
    batch, workers = len(prefetch), len(final[0])
    rows = np.zeros((batch,256), dtype=np.int32)
    posts = np.zeros((batch,workers,32), dtype=np.int32)
    rows[:,9] = prefetch
    rows[:,247] = np.asarray(final).sum(axis=1)
    posts[:,:,0] = final
    # Non-count words catch accidental uint64 reads or wrong FinalPost offsets.
    rows[:,8] = 1000000; rows[:,10] = 2000000
    posts[:,:,1] = 999999
    return SimpleNamespace(rotating=SimpleNamespace(runtime_shape=SimpleNamespace(batch=batch,finalWorkers=workers),
        rows=Buffer(rows), final_posts=Buffer(posts)))


class PipelineCounts(unittest.TestCase):
    def test_uint32_request_counts_and_final_worker_counts_are_not_budgets(self):
        states = [state_counts([17,0], [[1,2,3,4],[0,0,0,0]]),
                  state_counts([29,31], [[7,0,11,0],[1,1,1,1]])]
        records = collect_pipeline_counts(states, source_round=8)
        self.assertEqual([(r['input'],r['request']) for r in records], [(0,0),(0,1),(1,0),(1,1)])
        self.assertEqual([r['prefetch_requests'] for r in records], [17,0,29,31])
        self.assertEqual([r['final_get_requests'] for r in records], [10,0,18,4])
        self.assertEqual(records[2]['final_get_by_worker'], [7,0,11,0])
        self.assertEqual(records[2]['total_get_requests'], 47)
        self.assertTrue(all(r['source_round']==8 for r in records))
        self.assertTrue(all(r['row_final_requests']==r['final_get_requests'] for r in records))
        report = new_pipeline_count_report()
        self.assertIn('after measured event', report['collection'])
        self.assertIn('uint32', report['sources']['prefetch'])
        self.assertNotIn('first_submission', report)

    def test_invalid_or_missing_counters_raise_instead_of_becoming_zero(self):
        s = state_counts([1], [[1,2,3,4]])
        s.rotating.rows.values[0,247] += 1
        with self.assertRaisesRegex(ValueError, 'counters disagree'):
            collect_pipeline_counts([s], 1)
        for field, bad in [('rows',np.zeros((1,256),np.uint64)),
                           ('final_posts',np.zeros((1,4,32),np.float32)),
                           ('rows',np.zeros((1,255),np.int32))]:
            s = state_counts([1], [[1,2,3,4]])
            setattr(s.rotating, field, Buffer(bad))
            with self.assertRaisesRegex(ValueError, 'ABI mismatch'):
                collect_pipeline_counts([s], 1)
        s = state_counts([1], [[1,2,3,4]])
        def failed_read(): raise RuntimeError('device readback failed')
        s.rotating.rows = Buffer(s.rotating.rows.values, failed_read)
        with self.assertRaisesRegex(RuntimeError, 'device readback failed'):
            collect_pipeline_counts([s], 1)
        with self.assertRaises(ValueError): collect_pipeline_counts([], 1)

    def test_real_measurement_function_reads_all_measured_inputs_after_validation(self):
        import execution
        import reporting
        phase = {'name':'setup'}
        reads, validated, graph_replays = [], [], []
        class State:
            def __init__(self, index):
                self.index, self.device, self.source_round = index, 'cpu-contract', 5
                self.rotating = state_counts([0,0], [[0]*4,[0]*4]).rotating
                self.rotating.rows.read = lambda: reads.append((phase['name'],self.source_round,self.index,'rows'))
                self.rotating.final_posts.read = lambda: reads.append((phase['name'],self.source_round,self.index,'posts'))
                self.lease = SimpleNamespace(capture_scope=nullcontext, retain_graph=lambda graph: None)
                class Forward:
                    def __call__(forward):
                        seed = (1000 if phase['name']=='measured' else 10)+100*self.source_round+self.index
                        self.rotating.rows.values[:,9] = [seed,seed+1]
                        self.rotating.final_posts.values[:,:,0] = [[self.index+1,2,3,4],[5,6,7,8]]
                        self.rotating.rows.values[:,247] = self.rotating.final_posts.values[:,:,0].sum(axis=1)
                    def set_diagnostics(forward, enabled):
                        if enabled: raise AssertionError('readback enabled device instrumentation')
                self.forward = Forward()
            def reset(self):
                self.rotating.rows.values.fill(0); self.rotating.final_posts.values.fill(0)
            def prepare(self, inputs): self.source_round = inputs.source_round
            def validate(self, inputs): validated.append((phase['name'],inputs.source_round,self.index))
            def check_completion(self): pass
        states = [State(i) for i in range(10)]
        class Graph:
            def replay(self):
                graph_replays.append(phase['name'])
                for state in states: state.forward()
            def reset(self):
                for state in states: state.reset()
        stream = SimpleNamespace(synchronize=lambda: None)
        torch = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda device: stream,
            NPUGraph=Graph, Stream=lambda: stream, graph=lambda graph,stream: nullcontext()))
        class Sequence:
            rounds = (5,8)
            def __iter__(self):
                for number in self.rounds: yield SimpleNamespace(source_round=number)
        timing = dict(samples_us=list(range(10,20)), mean_us=77.0,
            stage_mean_us={'li':12.0,'resolve_update':20.0}, inter_kernel_gap_mean_us=3.0,
            overlap_mean_us=0.0, outside_invocations_mean_us=42.0)
        expected_timing = deepcopy(timing)
        def profile(torch, graph, **kwargs):
            self.assertEqual(kwargs['calls'],10)
            phase['name'] = 'measured'
            graph.replay(); graph.completed()
            return dict(timing=deepcopy(timing),samples=[{'us':value} for value in range(10,20)],
                        accounting={},trace='/cpu-contract/trace.json')
        with tempfile.TemporaryDirectory() as tmp:
            config = SimpleNamespace(mode='bench',warmup=2,iterations=10,transport='urma')
            with patch('performance.profile_graph_batch',side_effect=profile):
                # Capture/prewarmup forwards use the distinct setup counter values.
                result = execution.measure_replay_batch(torch,states,Sequence(),config,Path(tmp),fused=True)
            evidence = result['pipeline_counts']
            self.assertEqual(evidence['status'],'PASS')
            self.assertEqual(len(evidence['records']),40)
            self.assertEqual({r['source_round'] for r in evidence['records']},{5,8})
            for record in evidence['records']:
                self.assertEqual(record['prefetch_requests'],1000+100*record['source_round']+record['input']+record['request'])
                self.assertIn(('measured',record['source_round'],record['input']), validated)
            self.assertEqual(len(reads),40)
            self.assertTrue(all(item[0]=='measured' for item in reads))
            self.assertEqual(len(graph_replays),6)  # 2 warmups x 2 rounds + 2 measured graphs.
            self.assertEqual(result['rounds'],{5:expected_timing,8:expected_timing})
            self.assertEqual(result['all_rounds']['event_mean_us'],77.0)
            self.assertEqual(result['all_rounds']['graph_calls'],10)
            self.assertEqual(result['all_rounds']['stage_mean_us'],expected_timing['stage_mean_us'])
            # The existing run_path persists the measure result in path_result.
            reporting.atomic_json(Path(tmp)/'path_result.json',result)
            self.assertTrue((Path(tmp)/'path_result.json').is_file())
            reads.clear()
            with patch('performance.profile_graph_batch',side_effect=profile), \
                    patch('pipeline_counts.collect_pipeline_counts',side_effect=AssertionError('serial count readback')):
                serial = execution.measure_replay_batch(torch,states,Sequence(),config,Path(tmp)/'serial',fused=False)
            self.assertIsNone(serial['pipeline_counts'])
            self.assertEqual(reads,[])
            self.assertEqual(serial['rounds'],result['rounds'])


if __name__ == '__main__':
    unittest.main()
