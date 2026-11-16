import json
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
routes = importlib.import_module("score_routes")
prepare = importlib.import_module("prepare_score_source")
runner = importlib.import_module("run_score_routes")


class ScoreRoutes(unittest.TestCase):
    def test_each_coordinate_owned_once_and_roundtrips_padding(self):
        for batch in (1, 2, 8, 16, 24, 32):
            for cube, vector in ((32, 64), (28, 56), (16, 24)):
                for mode, p, q in (("history", 7, 4), ("history", 32, 4), ("batch_query", 1, 2), ("batch_query", 1, 4)):
                    plan = routes.plan(batch, 8448, 32, cube, vector, mode, p, q)
                    count = np.zeros((batch, 4, 8448), np.uint8)
                    for t in plan["tasks"]:
                        count[t["owner"], t["query_begin"] : t["query_begin"] + q, t["begin"] : t["end"]] += 1
                    self.assertTrue(np.all(count == 1))
                    self.assertTrue(all(len(w) <= min(cube, vector // 2) for w in plan["waves"]))
                    data = np.arange(batch * 4 * 8448, dtype=np.uint32).reshape(batch, 4, 8448).astype(np.uint16)
                    rows = routes.gather_rows(data, plan)
                    np.testing.assert_array_equal(routes.scatter_rows(rows, plan), data)
                    if np.any(rows == 0x5A5A) and plan["row_stride"] != plan["tasks"][0]["end"]:
                        rows[0, 0, -1] = 0
                        with self.assertRaises(ValueError):
                            routes.scatter_rows(rows, plan)

    def test_cpp_and_python_adapters_for_actual_ranges(self):
        code = """#include "score_schedule.h"
#include <iostream>
int main(){for(unsigned b: {1U,8U,16U,24U,32U})for(unsigned p: {1U,7U,28U,32U})
for(unsigned q: {2U,4U}){if(p>1&&q==2)continue;auto s=kvip_score::MakePlan(b,8448,64,28,56,p,q);
std::cout<<b<<","<<p<<","<<q<<","<<s.capacity<<","<<s.rowStride<<"\\n";
for(auto t:s.tasks)std::cout<<t.owner<<","<<t.queryBegin<<","<<t.queryCount<<","<<t.begin<<","<<t.end<<"\\n";}}
"""
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "check.cpp"
            exe = Path(d) / "check"
            src.write_text(code)
            subprocess.run(
                ["clang++", "-std=c++17", "-I", str(ROOT / "csrc/operators/common/lifetch"), str(src), "-o", str(exe)],
                check=True,
                capture_output=True,
            )
            lines = iter(subprocess.check_output([str(exe)], text=True).splitlines())
            for b in (1, 8, 16, 24, 32):
                for p in (1, 7, 28, 32):
                    for q in (2, 4):
                        if p > 1 and q == 2:
                            continue
                        plan = routes.plan(b, 8448, 64, 28, 56, "history" if p > 1 else "batch_query", p, q)
                        self.assertEqual(
                            list(map(int, next(lines).split(","))), [b, p, q, plan["capacity"], plan["row_stride"]]
                        )
                        for t in plan["tasks"]:
                            self.assertEqual(list(map(int, next(lines).split(","))), list(t.values()))
            self.assertEqual(list(lines), [])

    def test_dot_relu_and_signed_owner_oracle(self):
        # ReLU must follow the complete D dot: (+1)+(-2) is negative.
        q = np.zeros((2, 4, 32, 128), np.uint8)
        k = np.zeros((2, 256, 128), np.uint8)
        q[:, :, 0, :2] = 0x38
        k[0, :, 0] = 0x38
        k[0, :, 1] = 0xC0
        k[1, :, 0] = 0x40
        w = np.zeros((2, 4, 32), np.uint16)
        w[:, :, 0] = 0x3F80
        w[1, :, 0] = 0xBF80
        actual = routes.score_oracle(q, k, w, np.ones(w.shape, np.float32), np.ones((2, 256), np.float32))
        self.assertTrue(np.all(actual[0] == 0x8000))
        self.assertTrue(np.all(actual[1] == np.uint16(~np.uint16(0xC000))))

    def test_bf16_midpoint_matches_original_cast_round(self):
        np.testing.assert_array_equal(
            routes.sortable_bf16(np.array([12.03125, -12.03125], np.float32)), np.array([0xC141, 0x3EBE], np.uint16)
        )

    def test_fixture_tasks_recompute_equal_keys_and_preserve_owner_query(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "fixture"
            routes.make_fixture(f, 2, 512, 32)
            q = np.fromfile(f / "query.bin", np.uint8).reshape(2, 4, 32, 128)
            k = np.fromfile(f / "key.bin", np.uint8).reshape(2, 512, 128)
            w = np.fromfile(f / "weight.bin", np.uint16).reshape(2, 4, 32)
            qs = np.fromfile(f / "qscale.bin", np.float32).reshape(2, 4, 32)
            ks = np.fromfile(f / "kscale.bin", np.float32).reshape(2, 512)
            oracle = np.fromfile(f / "oracle.bin", np.uint16).reshape(2, 4, 512)
            self.assertFalse(np.array_equal(oracle[0], oracle[1]))
            # Independently reconstruct whole Q4 result from Q2 batches and H halves.
            for b in range(2):
                for lo, hi in ((0, 256), (256, 512)):
                    out = routes.score_oracle(
                        q[b : b + 1], k[b : b + 1, lo:hi], w[b : b + 1], qs[b : b + 1], ks[b : b + 1, lo:hi]
                    )
                    np.testing.assert_array_equal(out[0], oracle[b, :, lo:hi])

    def test_recommendation_boundary_is_occupancy_not_timing(self):
        self.assertEqual(routes.plan(1)["producers"], 32)
        self.assertEqual(routes.plan(16)["producers"], 2)
        self.assertEqual(routes.plan(17)["producers"], 1)
        self.assertEqual(routes.plan(14, cube=28, vector=56)["producers"], 2)
        self.assertEqual(routes.plan(15, cube=28, vector=56)["producers"], 1)
        self.assertIsNone(routes.plan(32, cube=28, vector=56)["measured_latency_ms"])
        self.assertGreater(len(routes.plan(32, cube=28, vector=56)["waves"]), 1)
        for kw in (
            dict(batch=0),
            dict(batch=True),
            dict(batch=1, history=128),
            dict(batch=1, producers=0, mode="history"),
            dict(batch=1, query_group=2, mode="history"),
            dict(batch=1, vector=1),
            dict(batch=1, mode="unknown"),
        ):
            with self.assertRaises(ValueError):
                routes.plan(**kw)

    def test_measurement_selection_rejects_unpaired_or_unproven(self):
        a = dict(
            routes.boundary_matrix()[0],
            status="PASSED",
            oracle_passed=True,
            score_ms=3,
            input_sha="input",
            binary_sha="binary",
            device="device",
            run_mode="npu",
            config_sha="config",
            iterations=5,
        )
        h = {**a, "mode": "history", "producers": 4, "score_ms": 2}
        self.assertEqual(routes.measured_choices([a]), [])
        self.assertEqual(routes.measured_choices([{**a, "run_mode": "sim", "score_ms": None}]), [])
        self.assertEqual(routes.measured_choices([{**a, "run_mode": "sim"}, {**h, "run_mode": "sim"}]), [])
        self.assertEqual(routes.measured_choices([a, {**h, "binary_sha": "other"}]), [])
        self.assertEqual(routes.measured_choices([a, h])[0]["mode"], "history")
        self.assertEqual(routes.measured_choices([a, {**h, "oracle_passed": False}]), [])
        with self.assertRaises(ValueError):
            routes.measured_choices([{**h, "input_sha": None}])
        with self.assertRaises(ValueError):
            routes.measured_choices([{**h, "score_ms": float("nan")}])
        self.assertTrue(all(r["score_ms"] is None and r["status"] == "NOT_RUN" for r in routes.boundary_matrix()))

    def test_current_source_is_copied_without_mutation(self):
        source = ROOT / prepare.OP / "op_kernel/indexer/indexer.h"
        before = source.read_bytes()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "current"
            result = prepare.prepare(out)
            kernel = (out / prepare.OP / "op_kernel/indexer/indexer.h").read_text()
            self.assertIn("scoreOnly_", kernel)
            self.assertIn("scoreQueryGroup_", kernel)
            self.assertIn("splitCoreInfo.bN2Start = splitCoreInfo.bN2End = aiCoreIdx", kernel)
            self.assertEqual(result["version"], "current")
            self.assertEqual((out / prepare.OP / "op_kernel/indexer/indexer.h").read_bytes(), before)
        self.assertEqual(source.read_bytes(), before)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "removed-history"
            with self.assertRaises(ValueError):
                prepare.prepare(out, "historical")
            self.assertFalse(out.exists())

    def test_runner_rejects_empty_selection_before_lease_or_output(self):
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(
                output=Path(d) / "out",
                driver=Path(d) / "driver",
                iterations=1,
                batches=[1],
                history=8192,
                heads=32,
                cube=27,
                vector=56,
                version="current",
                producers=None,
                lease_file=Path(d) / "lease",
            )
            args.driver.write_text("placeholder")
            with self.assertRaises(ValueError):
                runner.run(args)
            self.assertFalse(args.output.exists())
            self.assertFalse(args.lease_file.exists())

    def test_runner_keeps_timeout_partial_output_and_command(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "driver").write_text("placeholder")
            (root / "config").write_text("env")
            args = SimpleNamespace(
                output=root / "out",
                driver=root / "driver",
                config=root / "config",
                iterations=1,
                batches=[1],
                history=512,
                heads=32,
                cube=32,
                vector=64,
                version="current",
                producers=None,
                lease_file=root / "lease",
                run_mode="npu",
                device=0,
                timeout=1,
            )
            with patch.object(
                runner.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(["driver"], 1, output=b"partial", stderr=b"error"),
            ):
                rows = runner.run(args)
            self.assertTrue(all(r["status"] == "FAILED" for r in rows))
            self.assertFalse(args.lease_file.exists())
            record = json.loads((args.output / "b1-p1-q4-command.json").read_text())
            self.assertEqual(record["status"], "TIMED_OUT")
            self.assertIsNone(record["exit_code"])
            self.assertTrue(record["argv"])
            self.assertEqual((args.output / "b1-p1-q4.log").read_text(), "partialerror")


if __name__ == "__main__":
    unittest.main()
