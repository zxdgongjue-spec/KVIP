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
    def test_accept_only_supported_shapes(self):
        for mode,group,batch in [("window-tail-x3",7,32),("window-tail-x4",10,32),("window-tail-x2",7,28),
                                 ("window-tail-x3-g7",7,24),("window-tail-x3-g7",10,32)]:
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
            return next((arm for arm in ref.parent.glob(f"fixed-q4-audit-{name}.*/g{g}/bs{b}/block1/{name}")
                         if (arm/"packed_result.json").is_file()), None)

        with TemporaryDirectory() as area, mock.patch.object(m,"values",side_effect=fake_values), mock.patch.object(m,"cached",side_effect=local_cache):
            area=Path(area)
            ref=area/"ordinary"
            cells=[(mode,g,b) for mode,gs in m.CELLS.items() for g in gs for b in (28,32) if m.supported(mode,g,b)]
            self.assertCountEqual(cells,[("window-tail-x2",7,32),("window-tail-x2",10,32),("window-tail-x4",7,32),
                                         ("window-tail-x3",10,32),("window-tail-x3-g7",7,28),("window-tail-x3-g7",7,32)])
            for mode,g,b in cells:
                name=m.policy_name(mode)
                common=dict(calls=10,traffic=[{}]*10,correctness="PASS",full_KV_Index_checked_calls=10,
                            checked_queries=b*40,checked_rows=b*10)
                base=ref/f"g{g}/bs{b}/block1/fused"
                base.mkdir(parents=True,exist_ok=True)
                (base/"packed_result.json").write_text(json.dumps(dict(common,cold_build_fingerprint="N/base")))
                if name != m.CONTROL:
                    arm=area/f"fixed-q4-audit-{name}.sample/g{g}/bs{b}/block1/{name}"
                    arm.mkdir(parents=True,exist_ok=True)
                    (arm/"packed_result.json").write_text(json.dumps(dict(common,cold_build_fingerprint="N/"+name,
                                                                          tail_text={"sha":"shared"})))
            for mode,g,b in cells:
                _, receipt=m.bind(mode,ref,g,b)
                self.assertEqual(receipt["tail_text"],None if b==28 else {"sha":"shared"})
                self.assertEqual(len(receipt["sha256"]),64)
            corrupt=ref/"g7/bs28/block1/fused/packed_result.json"
            checked=json.loads(corrupt.read_text())
            corrupt.write_text(json.dumps(dict(checked,full_KV_Index_checked_calls=9)))
            with self.assertRaisesRegex(ValueError,"checked natural ten-call"):
                m.bind("window-tail-x3-g7",ref,7,28)

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
