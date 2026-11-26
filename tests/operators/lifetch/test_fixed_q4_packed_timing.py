# SPDX-License-Identifier: Apache-2.0
"""Ten-call device attribution without per-invocation event estimates."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from fixed_q4_packed_timing import packed_account
from large_batch_pair import window_traffic


def fixture(names):
    # Use the same base timestamp precision as an exported CANN trace.
    base = 1_800_000_000_000_000
    events = [dict(name="scope", ph="X", ts=str(base), dur=1100),
              dict(name="process_name", ph="M", pid=991, args={"name": "Ascend Hardware"})]
    for call in range(10):
        for j, name in enumerate(names):
            t = base + call * 100 + 3 + j * 16
            def event(label, offset, dur, kind):
                return dict(name=label, ph="X", pid=991, tid=24, ts=str(t + offset), dur=dur,
                            args={"Task Type": kind, "Physic Stream Id": 24, "Task Id": call * 10 + j})
            # MIX duration is the union of overlapping AIC/AIV records, and a
            # device-queue suffix need not distinguish the BS16 LI symbols.
            if j == 0:
                events.append(event(name + "_mix_aic_1", 1, 4, "MIX_AIC"))
                events.append(event(name + "_mix_aiv_1", 0, 4, "MIX_AIV"))
            else:
                events.append(event(name + "_1", 0, 3 + j, "AI_VECTOR_CORE"))
    return events


class PackedAttribution(unittest.TestCase):
    def test_fused_by_batch_and_native(self):
        cases = [
            (1, True, ["v9_li_fixed_hot", "v9_li_kernel_1", "v9_li_fixed_cold", "v9_parallel_terminal"]),
            (2, True, ["v9_li_fixed_hot", "v9_li_kernel_1", "v9_li_fixed_cold", "v9_parallel_terminal_fixed"]),
            (16, True, ["v9_li_kernel", "v9_li_kernel_1", "v9_li_kernel_2", "v9_parallel_terminal_fixed"]),
            (16, True, ["v9_li_kernel_1", "v9_li_kernel_1", "v9_li_kernel_2", "v9_parallel_terminal"]),
            (32, True, ["v9_li_large_hot", "v9_li_large_threshold_cold", "v9_parallel_terminal_large"]),
            (32, True, ["v9_li_large_hot", "v9_li_large_two_cold", "v9_parallel_terminal_large"]),
            (32, False, ["QuantLightningIndexer", "A5LiduFusedBatchGetPrepareMtp", "KvafExecuteBatchGetAivKernelV3"]),
        ]
        for batch, fused, names in cases:
            with self.subTest(batch=batch, fused=fused):
                result = packed_account(fixture(names), "scope", 1000, batch=batch, fused=fused)
                means = result["mean_us"]
                self.assertEqual((result["calls"], means["graph"]), (10, 100))
                self.assertNotIn("graph", result["p50_us"])
                self.assertTrue(all("graph" not in s for s in result["samples"]))
                self.assertEqual(means["LI"], (9 if len(names) == 3 else 14) if fused else 5)
                self.assertEqual(means["tail"], (len(names) + 2) if fused else 9)
                self.assertEqual(result["p50_us"]["LI"], means["LI"])
                self.assertAlmostEqual(sum(means[k] for k in ("LI", "tail", "inter_kernel_gap",
                                                              "between_calls", "event_edges")), means["graph"])
                self.assertAlmostEqual(means["between_calls"] + means["event_edges"], means["event_minus_span"])

    def test_each_request_obeys_the_selected_window_grants(self):
        for grants in ((256, 384, 384), (256, 256, 256, 256)):
            with self.subTest(grants=grants):
                rows = np.zeros((2, 7 + len(grants)), dtype=np.int32)
                rows[0, 7:] = grants
                rows[0, 0] = sum(grants)
                counts = np.zeros((2, 4), dtype=np.int32)
                self.assertEqual(window_traffic(rows, counts, grants)["posted_per_window"], list(grants))
                rows[0, 7] -= 1
                with self.assertRaisesRegex(ValueError, "observed per-row"):
                    window_traffic(rows, counts, grants)
                rows[0, 8] += 1
                with self.assertRaisesRegex(ValueError, "observed per-row"):
                    window_traffic(rows, counts, grants)

    def test_experimental_cold_receipt_checks_all_ten_calls(self):
        symbol = "v9_li_large_bs32_split_role"
        names = ["v9_li_large_hot", symbol, "v9_parallel_terminal_large"]
        events = fixture(names)
        result = packed_account(events, "scope", 1000, fused=True, batch=32, expected_li23=symbol)
        self.assertEqual(result["calls"], 10)
        # One regular launch among nine experimental ones must fail attribution.
        next(e for e in events if e["name"].startswith(symbol))["name"] = "v9_li_large_threshold_cold"
        with self.assertRaisesRegex(ValueError, "Unattributed"):
            packed_account(events, "scope", 1000, fused=True, batch=32, expected_li23=symbol)
        for batch in (1, 2, 4, 8, 16):
            with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, "large fused batch"):
                packed_account(events, "scope", 1000, fused=True, batch=batch, expected_li23=symbol)

        for batch in (20, 24, 28):
            with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, "requires BS32"):
                packed_account(events, "scope", 1000, fused=True, batch=batch, expected_li23=symbol)

    def test_new_shared_final_cannot_be_attributed_as_an_older_cold_kernel(self):
        symbol = "v9_li_large_bs32_shared_final"
        events = fixture(["v9_li_large_hot", symbol, "v9_parallel_terminal_large"])
        self.assertEqual(packed_account(events, "scope", 1000, batch=32, fused=True, expected_li23=symbol)["calls"], 10)
        with self.assertRaisesRegex(ValueError, "Unattributed"):
            packed_account(events, "scope", 1000, batch=32, fused=True, expected_li23="v9_li_large_bs32_split_role")
        with self.assertRaisesRegex(ValueError, "requires BS32"):
            packed_account(events, "scope", 1000, batch=24, fused=True, expected_li23=symbol)

    def test_stage_p50_uses_the_ten_observed_spans(self):
        events = fixture(["v9_li_large_hot", "v9_li_large_threshold_cold", "v9_parallel_terminal_large"])
        tails = [e for e in events if e["name"].startswith("v9_parallel_terminal")]
        for i, tail in enumerate(tails):
            tail["dur"] = 30 if i < 3 else 5
        result = packed_account(events, "scope", 1000, fused=True, batch=32)
        self.assertEqual(result["p50_us"]["tail"], 5)
        self.assertEqual(result["mean_us"]["tail"], 12.5)
        self.assertEqual(result["graph_mean_us"], 100)

    def test_incomplete_and_invalid_traces_rejected(self):
        names = ["v9_li_kernel", "v9_li_kernel_1", "v9_li_kernel_2", "v9_parallel_terminal"]
        events = fixture(names)
        bad = dict(events[2], args={**events[2]["args"], "Task Id": -1})
        for altered in (events[:-1], events[:5] + events[6:], events[:2] + [bad] + events[3:],
                        events + [dict(events[-1], name="unexpected", cat="kernel")],
                        events + [dict(events[3], ts=str(int(events[-1]["ts"]) + 50))]):
            with self.assertRaises(ValueError):
                packed_account(altered, "scope", 1000, batch=16, fused=True)
        with self.assertRaisesRegex(ValueError, "event"):
            packed_account(events, "scope", 900, batch=16, fused=True)


if __name__ == "__main__":
    unittest.main()


def test_shared_final_timing_image_cannot_be_used_for_trace():
    import ctypes
    from types import SimpleNamespace

    import pytest
    from large_two_cold import attach

    entry = ctypes.CFUNCTYPE(ctypes.c_uint32)(lambda: 0)
    regular = SimpleNamespace(aclrtlaunch_v9_li_large_hot=entry, aclrtlaunch_v9_parallel_terminal_large=entry)
    class Forward:
        terminal = object()
        diagnostics_enabled = False

    state = SimpleNamespace(rotating=SimpleNamespace(library=regular), forward=Forward())
    for trace in (False, True):
        def launch(*args):
            return len(args)
        launch.window_policy = "shared-final"
        launch.trace_only = trace
        attach(state, launch)
        if not trace:
            from execution import StreamingForward
            with pytest.raises(ValueError, match="before enabling diagnostics"):
                StreamingForward.set_options(state.forward, diagnostics=True)
        state.forward.diagnostics_enabled = trace
        assert state.forward.terminal_function(1, 2, 3, 4) == 6
        state.forward.diagnostics_enabled = not trace
        with pytest.raises(ValueError, match="diagnostic flag"):
            state.forward.terminal_function(1, 2, 3, 4)
