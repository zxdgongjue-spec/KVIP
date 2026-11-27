# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
from tempfile import TemporaryDirectory
import json
import sys

sys.path.insert(0, str(Path(__file__).with_name("support")))
import b24_window_tail_diagnostic as m


class MatchedWindowTailTest(TestCase):
    def test_only_the_four_archived_arms(self):
        for mode,group,batch in [("window-tail-x3",7,32),("window-tail-x4",10,32),("window-tail-x2",7,28)]:
            with self.assertRaises(ValueError):
                m.bind(mode,Path("unused"),group,batch)
        ref=Path("fake/ordinary")
        with mock.patch.object(m,"cached",return_value=None):
            with self.assertRaisesRegex(FileNotFoundError,"same-card"):
                m.bind("window-tail-x2",ref,7,32)

    def test_match_current_library_and_each_cold_image(self):
        state=dict(cold_build="norm/cold-a",tail_text={"sha":"x"})
        natural=dict(cold_build_fingerprint="norm/cold-a")
        bind=dict(original_library="norm",tail_text={"sha":"x"})
        m.verify_launch(state,natural,bind)
        for replacement in (dict(cold_build="norm/wrong"),dict(cold_build="other/cold-a"),dict(tail_text={"sha":"y"})):
            with self.assertRaisesRegex(ValueError,"differs"):
                m.verify_launch(dict(state,**replacement),natural,bind)

    def test_four_pinned_sources_and_control_tail_evidence(self):
        def fake_values(arm,ref,g,b,name):
            base=json.loads((ref/f"g{g}/bs{b}/block1/fused/packed_result.json").read_text())
            return json.loads((arm/"packed_result.json").read_text()),base,None
        def local_cache(reader,ref,g,b,name):
            first=next(ref.parent.glob("fixed-q4-audit-"+name+".*"),None)
            return first/f"g{g}/bs{b}/block1/{name}" if first else None
        with TemporaryDirectory() as area, mock.patch.object(m,"values",side_effect=fake_values), mock.patch.object(m,"cached",side_effect=local_cache):
            area=Path(area);ref=area/"ordinary";ref.mkdir();(ref/"native.txt").write_text("base")
            for mode,gs in m.CELLS.items():
                tag=mode.replace("window-tail-x","window-x")+"-early"
                root=ref if mode=="window-tail-x3" else area/({"window-tail-x2":"fixed-q4-audit-window-x2-early.Qb575M","window-tail-x4":"fixed-q4-audit-window-x4-early.ON4wPF"}[mode])
                root.mkdir(exist_ok=True); (root/"audit_reader.txt").write_text(m.STUDY_READER); (root/"native.txt").write_text("base")
                for g in gs:
                    path=root/f"g{g}/bs32/block1"/(tag if root!=ref else "fused");path.mkdir(parents=True,exist_ok=True)
                    template=dict(calls=10,traffic=[{}]*10,cold_build_fingerprint="N/"+tag)
                    if root!=ref:template["tail_text"]={"sha":"shared"}
                    (path/"packed_result.json").write_text(json.dumps(template))
                    base=(ref/f"g{g}/bs32/block1/fused");base.mkdir(exist_ok=True,parents=True)
                    if not (base/"packed_result.json").exists():
                        (base/"packed_result.json").write_text(json.dumps(dict(template,cold_build_fingerprint="N/base")))
            for mode, gs in m.CELLS.items():
                for group in gs:
                    _, o=m.bind(mode,ref,group,32)
                    self.assertEqual(o["tail_text"],{"sha":"shared"});self.assertEqual(len(o["sha256"]),64)

    def test_only_one_forward_from_existing_eager_call(self):
        calls=[]
        class Real:
            def __init__(self):self.forward=lambda: calls.append("real")
        ex=SimpleNamespace(FusedReplayState=Real)
        def inspect(s):
            calls.append("probe")
            s.forward()
        with m.capture_first(ex,inspect):
            s=ex.FusedReplayState();s.forward()
        self.assertIs(ex.FusedReplayState,Real)
        self.assertEqual(calls,["probe","real"])
        with self.assertRaisesRegex(ValueError,"did not run"):
            with m.capture_first(ex,inspect):
                ex.FusedReplayState()
        self.assertIs(ex.FusedReplayState,Real)
