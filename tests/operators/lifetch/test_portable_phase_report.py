# SPDX-License-Identifier: Apache-2.0
"""Synthetic archived-timing fixtures only; no NPU or external archives needed."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import report_large_li_phases as r


def save(path, data):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(data))


class PortablePhaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.area = Path(self.tmp.name)
        self.root = self.area / "fixed-q4-four-window-carry-one-block.B20abc"
        self.control = self.area / "fixed-q4-packed.B20abc"
        self.source = self.area / "fixed-q4.B20abc"
        self.ident = dict(group=2,batch=20,host="b20",physical_npu=4,native=r.CARRY,sequence_sha256="g2",
                          protocol="ten_independent_states_one_graph_one_event_v1")
        self.mark(self.root, r.CARRY)
        self.packed(self.root, r.CARRY)
    def mark(self, path, rev):
        path.mkdir(exist_ok=True)
        for k,v in dict(native=rev,host="b20",physical_npu="4",kvaf="sdk",kvaf_tree="tree",machine_id="node20").items():
            (path / (k+".txt")).write_text(v)
    def packed(self, root, rev):
        path = root / "g2/bs20/block1/fused/packed_result.json"
        raw = dict(identity=dict(self.ident,native=rev),arm="fused",correctness="PASS",calls=10,
                   full_KV_Index_checked_calls=10,checked_queries=800,checked_rows=200,
                   samples=[dict(LI1=10,LI23=21,LI=31)]*10,
                   large_window_policy="four-window-carry" if rev==r.CARRY else None)
        save(path, raw)
        save(path.parent / "thp.json", dict(status="measured", page_mode="thp", coverage_min_pct=100,
                                           ub_device="nic", numa_pages_overlapping_vmas={"3": 100}))
        (path.parent / "placement_verified").touch()
        return path
    def run_report(self, root=None, rank_revision=None):
        buf=StringIO()
        with redirect_stdout(buf):
            cells=r.report(root or self.root,rank_revision, batches=[20],groups=[2])
        return cells,buf.getvalue()
    def test_missing_archive(self):
        cells,text=self.run_report(self.area/"fixed-q4-four-window-carry-one-block.VFkffz")
        self.assertEqual(cells,[]);self.assertIn("ARCHIVE_UNAVAILABLE",text)
    def test_new_only_no_links_no_old_roots(self):
        cells,text=self.run_report()
        self.assertEqual(cells[0]["current"]["LI1"],[10,10])
        for term in ("CONTROL unavailable","ORIGINAL_DIAG unavailable","CURRENT_DIAG unavailable","PHASE_REPORT_CELLS=1/1"):
            self.assertIn(term,text)
    def test_current_rank_timing_without_diagnostics(self):
        rank="1"*40
        self.mark(self.root,rank)
        path=self.packed(self.root,rank); data=json.loads(path.read_text());data["large_window_policy"]="four-ranks";save(path,data)
        cells,text=self.run_report(rank_revision=rank)
        self.assertEqual(cells[0]["current_calls"],10);self.assertIn("detail_unavailable",text)
    def test_moved_controls_and_ambiguous_fallback(self):
        self.mark(self.control,r.CONTROL);self.packed(self.control,r.CONTROL)
        other=self.area/"fixed-q4-packed.other";self.mark(other,r.CONTROL);self.packed(other,r.CONTROL)
        self.assertIsNone(self.run_report()[0][0]["control"])
        (self.root/"reference.txt").write_text("/old/mount/"+self.control.name)
        cells,_=self.run_report();self.assertEqual(cells[0]["control"],cells[0]["current"])
        (self.control/"physical_npu.txt").write_text("0")
        cells,_=self.run_report();self.assertEqual(cells[0]["control_root"],str(other))
        (other/"machine_id.txt").write_text("node24")
        self.assertIsNone(self.run_report()[0][0]["control"])
    def test_unequal_registered_memory_placement(self):
        self.mark(self.control, r.CONTROL)
        cell = self.packed(self.control, r.CONTROL).parent
        record = json.loads((cell / "thp.json").read_text())
        record["ub_device"] = "other"
        save(cell / "thp.json", record)
        self.assertIsNone(self.run_report()[0][0]["control"])

    def test_source_corrupt_diag_and_mismatched_input(self):
        self.mark(self.source,r.ORIGINAL);self.mark(self.control,r.CONTROL)
        path=self.packed(self.control,r.CONTROL); rec=json.loads(path.read_text());rec["identity"]["sequence_sha256"]="other";save(path,rec)
        src=self.source/"g2/bs20/fused/ARM_RESULT.json";save(src,dict(arm="fused",identity=dict(self.ident,native=r.ORIGINAL,protocol="one_restored_graph_per_device_event_with_profiler")))
        cells,text=self.run_report();self.assertEqual(cells[0]["current_calls"],10);self.assertIsNone(cells[0]["control"])
        self.assertIn("ORIGINAL_DIAG unavailable=",text)
if __name__=="__main__":
    unittest.main()
