import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent/"support"))
import g7_x4_local as study

class LocalNaturalGraph(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.area=Path(self.temp.name)
        self.reader="1"*40
        self.machine=patch.object(study,"content",return_value="local_id")
        self.machine.start()
        self.addCleanup(self.machine.stop)

    def root(self,name,who="local_id"):
        root=self.area/name
        root.mkdir()
        for k,v in (("machine_id",who),("boot_id","boot"),("host",os.uname().nodename),
                    ("physical_npu","4"),("native",study.BASE),("kvaf",study.SDK),("kvaf_tree",study.TREE)):
            (root/(k+".txt")).write_text(v)
        return root

    def original(self,who):
        root=self.root("fixed-q4-four-ranks-one-block."+who,who)
        arm=root/"g7/bs32/block1/fused"
        arm.mkdir(parents=True)
        for filename in ("thp.json","packed_result.json","placement_verified"):
            (arm/filename).write_text("{}")
        return root

    def test_each_batch_selects_only_this_host(self):
        self.original("other-host")
        self.assertEqual(study.choose(self.area,32,self.reader)[0],"local-run")
        root=self.original("local_id")
        self.assertEqual(study.choose(self.area,32,self.reader)[:2],("paired-run",root))
        self.assertEqual(study.choose(self.area,28,self.reader)[0],"local-run")
        cached=self.root("fixed-q4-audit-window-x4-early.cached")
        (cached/"audit_reader.txt").write_text(study.ORIGINAL)
        (cached/"audit_reference.txt").write_text(str(root))
        arm=cached/f"g7/bs32/block1/{study.MODE}"
        arm.mkdir(parents=True)
        for filename in ("placement_verified","window-study.json"):
            (arm/filename).touch()
        self.assertEqual(study.choose(self.area,32,self.reader),("paired-cache",root,arm))

    def test_local_cache_requires_correct_graph_and_measured_placement(self):
        root=self.root("fixed-q4-audit-window-x4-early.local")
        (root/"audit_reader.txt").write_text(self.reader)
        arm=root/f"g7/bs28/block1/{study.MODE}"
        arm.mkdir(parents=True)
        (arm/"local_placement_measured").touch()
        (arm/"thp.json").write_text(json.dumps(dict(status="measured",page_mode="thp",ub_device="local",
            coverage_min_pct=99.9,numa_pages_overlapping_vmas={"3":2000})))
        (arm/"window-study.json").write_text(json.dumps(dict(mode=study.MODE,reference=None)))
        ranks,grants=study.policy(study.MODE)
        data=dict(identity=dict(group=7,batch=28,native=study.BASE,physical_npu=4,host=os.uname().nodename,
            calls=10,sequence_sha256="a"*64),tail_text={"tag":"tail"},original_library="lib",correctness="PASS",
            calls=10,get_workers=32,full_KV_Index_checked_calls=10,checked_rows=280,checked_queries=1120,
            checked_rank_queries=112,actual_ranks=ranks,window_grants_per_row=grants)
        (arm/"packed_result.json").write_text(json.dumps(data))
        self.assertEqual(study.choose(self.area,28,self.reader),("local-cache","-",arm))
        data["checked_rows"]=279
        (arm/"packed_result.json").write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            study.choose(self.area,28,self.reader)

if __name__=="__main__":
    unittest.main()
