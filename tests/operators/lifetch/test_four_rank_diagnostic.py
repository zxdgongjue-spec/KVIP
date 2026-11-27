# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).with_name("support")))
from four_rank_diagnostic import package
from four_rank_tail_clocks import decode


def query(own, cuts):
    t = [0]*32
    stages = [0,1,2,3,6,11,12,7] + ([13] if own else []) + [14,15,4,5,8,26,16,24,17,9,10]
    for k, word in enumerate(stages, 1): t[word] = k*1000
    t[20] = t[21] = own
    extra = [0]*32
    extra[0] = len(cuts)
    extra[1:1+len(cuts)] = cuts
    for i in range(len(cuts)):
        tick = t[16]+8*i
        extra[8+i*4:12+i*4] = ([0,0] if i == 0 else [tick,tick+1]) + [tick+2,tick+5]
    return t,extra


class DiagnosticTest(unittest.TestCase):
    def test_mixed_counts_and_overlapping_prefixes(self):
        packed = [query(0,[]),query(1,[1]), query(2,[1,2]),query(320,[20,120,220,320])]
        t,c = map(list,zip(*packed))
        result = decode([t],[c],1)
        self.assertEqual([len(q["batches"]) for q in result["qs"]],[0,1,2,4])
        self.assertIsNone(result["qs"][0]["last_final_cq_to_exit_us"])
        last = result["qs"][3]
        self.assertEqual(last["last_final_cq_to_exit_us"], t[3][10]-c[3][23])
        c[3][12] = t[3][16]
        with self.assertRaisesRegex(ValueError,"chronology"):
            decode([t],[c],1)

    def test_two_prefixes_in_flight_with_in_order_receipts(self):
        queries = [query(0, []) for _ in range(3)] + [query(320, [20,120,220,320])]
        stamps, detail = map(list, zip(*queries))
        extra = detail[-1]
        extra[5] = 2
        index_end = stamps[-1][16]
        extra[12:16] = [index_end, index_end+1, index_end+12, index_end+15]
        extra[16:20] = [index_end+6, index_end+7, index_end+18, index_end+20]
        extra[20:24] = [index_end+15, index_end+16, index_end+21, index_end+23]
        self.assertEqual(decode([stamps], [detail], 1)["qs"][-1]["private_credit"], 2)
        extra[16] = index_end+3
        with self.assertRaisesRegex(ValueError, "chronology"):
            decode([stamps], [detail], 1)

    def test_packages_current_and_reused_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            area=Path(tmp)/"peer-baseline"
            root=area/"four-rank-diagnostic.latest"
            for group, source in [(7,root), (10,area/"four-rank-diagnostic.previous")]:
                dst=source/f"g{group}/bs32/fused"
                dst.mkdir(parents=True)
                (dst/"placement_verified").touch()
                (dst/"single_diagnostic.json").write_text(json.dumps({"group":group}))
                (dst/"thp.json").write_text("{}")
                if group==10:
                    d=dst/"li-pmu/PROF_123"
                    d.mkdir(parents=True);(d/"op_summary_1.csv").write_text("test")
                idx=root/f"g{group}/bs32/diagnostic_source.txt"
                idx.parent.mkdir(parents=True,exist_ok=True);idx.write_text(str(dst))
            (root/"summary.txt").write_text("2")
            package(root)
            with tarfile.open(root/"four-rank-diagnostics.tar.gz") as archive:
                self.assertEqual(len(archive.getnames()),6)
                self.assertIn("g10/bs32/li-pmu/PROF_123/op_summary_1.csv",archive.getnames())
