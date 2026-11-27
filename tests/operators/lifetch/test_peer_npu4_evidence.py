# SPDX-License-Identifier: Apache-2.0
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from peer_npu4_evidence import BASE, TREE, SDK, origin, write_package

CURRENT = ("localhost.localdomain", "server20", "boot20")

def field(root, key, value):
    (root/(key+".txt")).write_text(value + "\n")

class HostEvidence(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.area = Path(self.tmp.name)

    def archive(self, suffix, card, mid):
        root = self.area/("fixed-q4-four-ranks-one-block."+suffix)
        root.mkdir()
        for k,v in dict(native=BASE, kvaf=SDK, kvaf_tree=TREE, host=CURRENT[0], machine_id=mid,
                        boot_id=CURRENT[2], physical_npu=card, candidate_blocks="1",
                        selection_oracle="captured_logical_scores").items():
            field(root,k,v)
        arm = root/"g2/bs28/block1/fused"
        arm.mkdir(parents=True)
        for name in ("packed_result.json","placement_verified"):
            (arm/name).touch()
        return root

    def test_wrong_host_same_hostname_wrong_card_and_partial(self):
        self.archive("MRDDaH", "4", "server24")
        self.archive("othercard", "3", "server20")
        self.assertIsNone(origin(self.area, CURRENT))
        correct = self.archive("random", "4", "server20")
        self.assertEqual(origin(self.area, CURRENT), correct)
        field(correct,"machine_id","")
        self.assertEqual(origin(self.area, CURRENT), correct)
        field(correct,"boot_id","foreign")
        self.assertIsNone(origin(self.area, CURRENT))

    def test_unpaired_package_without_base(self):
        root = self.archive("localdiag", "4", "server20")
        for b in (28,32):
            arm = root/f"g2/bs{b}/block1/clocks"
            arm.mkdir(parents=True)
            (arm/"local_placement_measured").touch()
            (arm/"thp.json").write_text(json.dumps(dict(status="measured", page_mode="thp", ub_device="local",
                coverage_min_pct=99.9, numa_pages_overlapping_vmas={"2":100})))
            (arm/"single_diagnostic.json").write_text(json.dumps(dict(identity=dict(physical_npu=4,native=BASE,
                group=2,batch=b), controller_commit="test", correctness="PASS", measured_calls=0, diagnostic_calls=1)))
            field(self.area,f"g2-bs{b}",str(arm))
        with patch("peer_npu4_evidence.machine",return_value=CURRENT):
            write_package(self.area,None,"test")
        with gzip.open(self.area/"window-evidence.json.gz", "rt") as fp:
            result=json.load(fp)
        self.assertIsNone(result["reference"])
        self.assertEqual(set(result["local_diagnostics"]), {"28","32"})
        self.assertEqual(result["archived"], {})
