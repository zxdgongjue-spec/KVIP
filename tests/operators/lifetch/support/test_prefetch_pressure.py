import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyze_prefetch_pressure import analyze


class PressureTest(unittest.TestCase):
    def test_read_counts_and_cycle_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'current-b16-m15/case0'
            path.mkdir(parents=True)
            report = dict(status='pass', native_misses_per_request=[2]*16,
                          timing={k: {'p50_us': 100} for k in ('native_chain','prototype_with_update')},
                          read_samples={'prototype_with_update': [dict(prefetch_hot=4,prefetch_cold=20,
                              unused_hot=4,unused_cold=0,final=12,total=36)]})
            (path/'report.json').write_text(json.dumps(report))
            sample = dict(round=0, prefetch=24, final=12, accounting={'stages_us': {'resolve_update': 50}},
                          streaming_cycles=[{'core_timeline_raw': {'batch':16}}])
            (path/'full_attribution.json').write_text(json.dumps({'samples':[sample]}))
            cores = [dict(block=0,role='resolver+finalize b0',start_us=0,end_us=50,stages=[]),
                     dict(block=1,role='index_update b0',start_us=0,end_us=20,stages=[]),
                     dict(block=2,role='fetch Q1 lane0',start_us=0,end_us=40,stages=[],published_misses=[12],
                          KVAF_counters={'safe_post_cycles':2000,'safe_CQ_cycles':10000})]
            timeline = dict(cores=cores,last_block=0,clock_mhz=1000,observed_span_us=50)
            with patch('analyze_prefetch_pressure.decode', return_value=timeline):
                text, _, records = analyze(root,16,[15])
                self.assertIn('native_reads=32',text)
                self.assertIn('tail_after_fetch=10.000',text)
                self.assertIn('| 2.00 | 10.00',text)
                self.assertEqual(len(records),1)
                report['read_samples']['prototype_with_update'][0]['total'] = 99
                (path/'report.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'Read accounting'):
                    analyze(root,16,[15])


if __name__ == '__main__':
    unittest.main()
