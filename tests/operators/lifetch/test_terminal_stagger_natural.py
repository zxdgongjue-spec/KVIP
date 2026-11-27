# SPDX-License-Identifier: Apache-2.0
"""Regression checks for advancing a fixed-list result to the ordinary graph."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "support"))
import terminal_stagger_natural as study
from terminal_first_cq_clocks import measure_first


class NaturalConfirmation(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.out = self.root("new", "new-reader")
        self.ref = self.base/"archive"
        self.root0 = self.root("prior", "new-reader")
        self.pair = self.root0/"g7/bs32/block1"
        self.ident = {"group":7, "batch":32, "physical_npu":4, "host":"test"}
        self.checks = [dict(first_CQ=dict(pilot_delays_us=[17.], pilot_poll_us=[12.], partner_done_us=[32.],
                                         first_team_done_us=58., peak_Q=1, peak_tokens=1), actual_window_sha256="win", actual_final_sha256="end", ordered_match=True,
                            native_final_guard_failures=0, index_assignment_by_row=["ok"]*32) for _ in range(10)]
        for i in (0,1):
            arm = self.pair/f"gate{i}"
            arm.mkdir(parents=True)
            self.put(arm/"thp.json", dict(status="measured", page_mode="thp", coverage_min_pct=99.5,
                numa_pages_overlapping_vmas={"3":1024}, ub_device="ut"))
            (arm/"placement_verified").touch()
            self.put(arm/"stagger.json", dict(identity=self.ident, arm="first-receipt-pair", gate_word=i,
                correctness="PASS", calls=10, full_KV_Index_checked_calls=10, frozen_journal_sha256="journal",
                expected_window_sha256="win", expected_final_sha256="end", tail_text="frozen_tail",
                guard_sha256="guard", cold_build="guarded_LI", cold_build_fingerprint="guarded_LI",
                original_journal=str(self.pair/"journal"), compiler_sha256="compiler",
                traffic=[dict(row_window_prefetch=[[0]*4]*32, row_query_final=[[0]*4]*31+[[96]*4])]*10,
                timed_journal_checks=[dict(c,pilot_pairs_confirmed=2*i) for c in self.checks], graph_mean_us=100-i, mean_us={"LI":50, "tail":45-i},
                p50_us={"LI":50,"tail":45-i}))
        (self.pair/"journal").mkdir()
        self.put(self.pair/"journal/journal.json", {"journal_sha256":"journal","cold_build":"ordinary_LI"})

    def root(self, name, reader):
        r=self.base/("fixed-q4-audit-first-receipt-pair."+name)
        r.mkdir()
        for k,v in dict(native=study.BASE, kvaf="kvaf", kvaf_tree="tree", machine_id="this-host",
                        physical_npu="4", host="test", audit_reader=reader, audit_reference=str(self.base/"archive")).items():
            (r/f"{k}.txt").write_text(v)
        return r

    def put(self,p, obj):
        p.write_text(json.dumps(obj))

    def test_cache_rejects_other_host_or_software(self):
        self.assertEqual(study.archived_pair(self.out,self.ref,7,32),self.pair)
        (self.root0/"machine_id.txt").write_text("different")
        self.assertIsNone(study.archived_pair(self.out,self.ref,7,32))
        (self.root0/"machine_id.txt").write_text("this-host")
        (self.root0/"audit_reader.txt").write_text("unknown")
        self.assertIsNone(study.archived_pair(self.out,self.ref,7,32))

    def test_advance_only_if_same_work_and_both_times_improve(self):
        with redirect_stdout(io.StringIO()):
            self.assertTrue(study.eligible(self.pair,7,32))
            p=self.pair/"gate1/stagger.json"
            doc=json.loads(p.read_bytes())
            doc["graph_mean_us"]=102
            self.put(p,doc)
            self.assertFalse(study.eligible(self.pair,7,32))
            doc["graph_mean_us"]=99
            doc["timed_journal_checks"][0]["actual_final_sha256"]="different"
            self.put(p,doc)
            self.assertFalse(study.eligible(self.pair,7,32))

    def test_natural_compares_ordinary_li_even_without_timed_control(self):
        arm=self.out/"g7/bs32/block1/fused"
        arm.mkdir(parents=True)
        self.put(arm/"packed_result.json", dict(identity=self.ident, get_workers=32,
            stagger_natural_image=dict(gate_word=1,tail_text="ordinary_tail",cold_build="ordinary_LI"),
            measured_cold_symbol="v9_li_large_four_ranks",cold_build_fingerprint="ordinary_LI",
            p50_us={"LI":5,"tail":4}))
        observed=({"graph":10,"LI":5,"tail":4}, (self.ident,("3","ut")))
        with patch("export_large_rank_evidence.observation",return_value=observed), \
                patch("four_rank_followup.serial_reference",return_value=None), redirect_stdout(io.StringIO()) as log:
            study.summary(self.out,self.ref,7,32,self.pair)
            self.assertIn("vs_same_card_serial UNAVAILABLE",log.getvalue())
            self.assertIn("vs_original_natural UNAVAILABLE",log.getvalue())
            p=arm/"packed_result.json"
            doc=json.loads(p.read_bytes())
            doc["cold_build_fingerprint"]="guarded_LI"
            self.put(p,doc)
            with self.assertRaises(ValueError):
                study.summary(self.out,self.ref,7,32,self.pair)


class FirstReceiptClocks(unittest.TestCase):
    def test_pilot_and_partner_have_distinct_timing_origins(self):
        import numpy as np
        stamps = np.zeros((1,4,32), dtype=int)
        stamps[0,:,0] = 10
        stamps[0,:,10] = [60,80,66,86]
        stamps[0,:,27:30] = [[20,25,35],[36,38,50],[22,24,34],[35,39,55]]
        counts = [[64]*4]
        record = measure_first(stamps,counts,[0],1.,True)
        self.assertEqual(record["peak_Q"],2)
        self.assertEqual(record["peak_tokens"],8)
        self.assertEqual(record["pilot_delays_us"],[15.,12.])
        self.assertEqual(record["pilot_poll_us"],[10.,10.])
        self.assertEqual(record["partner_done_us"],[40.,45.])
        self.assertEqual(record["first_team_done_us"],76.)
        stamps[0,1,27] = 31
        with self.assertRaisesRegex(ValueError,"before its pilot"):
            measure_first(stamps,counts,[0],1.,True)

    def test_empty_row_needs_no_receipt(self):
        import numpy as np
        stamps = np.zeros((1,4,32), dtype=int)
        stamps[0,:,0], stamps[0,:,10] = 20, 24
        record = measure_first(stamps,[[0]*4],[0],2.,True)
        self.assertEqual((record["peak_Q"],record["pilot_delays_us"],record["first_team_done_us"]),(0,[],2.))


if __name__ == "__main__":
    unittest.main()
