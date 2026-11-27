# SPDX-License-Identifier: Apache-2.0
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import b24_g7_clocks as clocks


def write_cell(root, batch, card=7):
    path = root / f"g7/bs{batch}/fused/ARM_RESULT.json"
    path.parent.mkdir(parents=True)
    for name, value in (("machine_id", "machine-a"), ("native", clocks.NATIVE), ("physical_npu", "7")):
        (root / (name + ".txt")).write_text(value)
    rows = []
    for row in range(batch):
        queries = []
        for q in range(4):
            own = int(q == 0)
            stamp = [0] * 32
            order = [0, 1, 2, 3, 6, 11, 12, 7] + ([13] if own else []) + [14, 15, 4, 5, 8, 26, 16, 24, 17, 9, 10]
            for step, slot in enumerate(order):
                stamp[slot] = 100 + step + (80 if row == 1 else 0)
            stamp[20:23] = [own, own, 4]
            queries.append(stamp)
        rows.append(queries)
    windows = [[[20, 30, (epoch << 32) + r, 1] + [0] * 6 + [1] + [0] * 5
                for r in range(batch)] for epoch in range(4)]
    ident = dict(group=7, batch=batch, native=clocks.NATIVE, physical_npu=card, protocol=clocks.SINGLE,
                 donor_rows=list(range(16)) + [i * 16 // (batch-16) for i in range(batch-16)],
                 host=os.uname().nodename, sequence_sha256="b"*64, calls=10, warmup=4)
    record = dict(arm="fused", identity=ident, get_workers=32, checked_queries=40*batch, checked_rows=10*batch,
                  full_KV_Index_checked_rows=batch, ordered_Index_diagnostic_rows=batch,
                  separate_terminal_query_stamps=rows, separate_window_event_u64=windows,
                  separate_LI_diagnostic=dict(clock_mhz=2),
                  separate_terminal_teams=dict(last_row=1, last_q=1, observed_span_us=(rows[1][0][10]-100)/2))
    path.write_text(json.dumps(record))
    return path


class TestG7Tail(unittest.TestCase):
    def test_score_oracle_ignores_invalid_high_ids_and_accepts_only_cutoff_ties(self):
        from types import SimpleNamespace
        import numpy as np
        from large_batch_pair import captured_score_reference, check_scores
        from workloads import ReplayRound

        keys = np.zeros((1, 4, 2055), dtype=np.int32)
        keys[:, :, :2046] = 10
        keys[:, :, -1] = 100
        valid = np.ones(keys.shape, dtype=bool)
        valid[:, :, -1] = False
        inputs = SimpleNamespace(geometry=dict(batch=1, history=2055), logical_score_keys=keys, logical_score_valid=valid)
        inputs.validate_topk = lambda ids: ReplayRound.validate_topk(inputs, ids)
        reference = captured_score_reference(inputs)
        self.assertEqual(reference.shape, (10, 1, 4, 2048))
        other = reference[0].copy()
        missing = np.setdiff1d(np.arange(2054), other[0, 0])
        other[0, 0, -1] = missing[0]
        check_scores(inputs, other, reference[0])
        other[0, 0, 0] = missing[1]
        with self.assertRaises(ValueError):
            check_scores(inputs, other, reference[0])

    def test_per_cell_reuse_across_archives_and_foreign_card(self):
        with TemporaryDirectory() as temp, patch.object(clocks, "local_marker", return_value="machine-a"):
            root = Path(temp)
            write_cell(root / "part-a", 20)
            write_cell(root / "part-b", 32)
            write_cell(root / "wrong-card", 24, 3)
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
                clocks.main(root)
            self.assertEqual([s for s in out.getvalue().splitlines() if s.startswith("MISSING")],
                             ["MISSING 24", "MISSING 28"])
            write_cell(root / "part-c", 24)
            write_cell(root / "part-d", 28)
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
                clocks.main(root)
            self.assertIn("DIAGNOSTIC_READ cells=4/4", out.getvalue())
            self.assertEqual(out.getvalue().count("last_row=1 last_Q=1"), 4)
            self.assertIn("Q1 final_GET=1 start=40.00 WQE=44.00..44.50", out.getvalue())
            self.assertIn("Q2 final_GET=0 start=40.00 WQE=-", out.getvalue())
            self.assertIn("final_CQ_join=- Q2_prefetch_CQ_join=47.50..48.00", out.getvalue())

    def test_incompatible_input_or_truncated_clocks_cannot_suppress_measurement(self):
        with TemporaryDirectory() as temp, patch.object(clocks, "local_marker", return_value="machine-a"):
            root = Path(temp)
            path = write_cell(root, 28)
            with self.assertRaisesRegex(ValueError, "input differs"):
                clocks.timeline(root, 28, "c"*64)
            doc = json.loads(path.read_text())
            doc["separate_terminal_query_stamps"][1][0][10] = 0
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "terminal clocks"):
                clocks.timeline(root, 28, "b"*64)
