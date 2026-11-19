"""Do not publish a numeric contract PASS as timing if the runtime failed."""
import unittest
from run_query_model import parse_result


class ModelEvidence(unittest.TestCase):
    LINE=('QUERY_TOPK_PASS B2 Q4 H8192 R2048 W4 route=query topk_keys=16384 '
          'candidate_count=512 candidate_hash=1741888167578971299 '
          'work_span_cycles=14137 max_lane_cycles=13811 scope=incremental_topk_rank_only\n')

    def test_contract_and_timing_evidence(self):
        row=parse_result(self.LINE)
        self.assertEqual((row['batch'],row['route'],row['work_span_cycles']),(2,'query',14137))
        for invalid in ('',self.LINE+self.LINE,
                        self.LINE.replace('work_span_cycles=14137','work_span_cycles=0')):
            with self.assertRaises(RuntimeError):parse_result(invalid)

    def test_runtime_errors_invalidate_numerical_pass(self):
        for error in ('[ERROR] pem_ccu execute_set_flag already has same set_flag!',
                      'double free or corruption (fasttop)\nAborted'):
            with self.assertRaisesRegex(RuntimeError,'Model/runtime error'):
                parse_result(self.LINE+error)


if __name__=='__main__':unittest.main()
