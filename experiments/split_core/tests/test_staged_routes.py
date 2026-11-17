from pathlib import Path
import importlib
import subprocess
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/split_core"))
staged_routes = importlib.import_module("staged_routes")
cases, owner_waves, stage_tasks = staged_routes.cases, staged_routes.owner_waves, staged_routes.stage_tasks


class StagedRoutes(unittest.TestCase):
    def test_all_default_cases_and_exact_cover(self):
        self.assertEqual(
            [(x["batch"], x["miss_percent"]) for x in cases()],
            [(b, m) for b in (1, 2, 4, 8, 16, 24, 32) for m in (3, 5, 10, 15, 20, 50)],
        )
        self.assertEqual(len(list(cases())), 42)
        for case in cases():
            b = case["batch"]
            for lo, hi in ((0, 16384), (16384, 131072)):
                p = stage_tasks(b, lo, hi)
                intervals = {}
                for t in p["tasks"]:
                    for q in range(t.query_begin, t.query_begin + p["query_group"]):
                        intervals.setdefault((t.owner, q), []).append((t.begin, t.end))
                self.assertEqual(set(intervals), {(x, q) for x in range(b) for q in range(4)})
                for ranges in intervals.values():
                    ranges.sort()
                    self.assertEqual(ranges[0][0], lo)
                    self.assertEqual(ranges[-1][1], hi)
                    self.assertTrue(all(a[1] == z[0] for a, z in zip(ranges, ranges[1:])))
                self.assertTrue(all(2 * len(w) <= 56 and len(w) <= 28 for w in p["waves"]))
            owners = [x for first, n in owner_waves(b) for x in range(first, first + n)]
            self.assertEqual(owners, list(range(b)))
            self.assertTrue(all(3 * n + 8 <= 56 for _, n in owner_waves(b)))

    def test_range_completion_first_eligible(self):
        for b in range(1, 33):
            for w in (1, 2, 4):
                for first, n in owner_waves(b, workers=w):
                    shards = [[x for x in range(first, first + n) if x % w == lane] for lane in range(w)]
                    self.assertEqual(sorted(x for s in shards for x in s), list(range(first, first + n)))
                    for lane, shard in enumerate(shards):
                        start = first + (lane + w - first % w) % w
                        self.assertEqual(shard, list(range(start, first + n, w)))

    def test_bsq_boundary_and_forced_routes(self):
        self.assertGreater(stage_tasks(1, 0, 16384)["producers"], 1)
        self.assertEqual(stage_tasks(14, 0, 16384)["query_group"], 4)
        self.assertEqual(stage_tasks(16, 0, 16384)["query_group"], 2)
        self.assertEqual(stage_tasks(32, 16384, 131072)["query_group"], 2)
        self.assertEqual(len(stage_tasks(32, 16384, 131072)["waves"]), 3)
        self.assertEqual(stage_tasks(1, 0, 16384, force="bsq")["query_group"], 2)
        for args in ((0, 0, 128), (1, 1, 128), (1, 128, 0)):
            with self.assertRaises(ValueError):
                stage_tasks(*args)

    def test_default42_real_fp8_and_owner_reference(self):
        from staged_routes import fixture
        from staged_oracle import check_case

        with tempfile.TemporaryDirectory() as d:
            folders = {}
            checked = []
            for case in cases():
                miss = case["miss_percent"]
                if miss not in folders:
                    folders[miss] = Path(d) / f"m{miss}"
                    fixture(folders[miss], miss)
                result = check_case(folders[miss], case)
                self.assertEqual(result["status"], "PASSED")
                self.assertEqual(result["unique_miss"], 2 * case["miss_per_query"])
                checked.append(case["case"])
            self.assertEqual(len(set(checked)), 42)

    def test_staged_index_update_b24_b32_cpu_abi(self):
        import torch
        from types import SimpleNamespace

        sys.path.insert(0, str(ROOT / "python"))
        from kvip.operators.lifetch.index_update import DirectIndexUpdate

        for b in (24, 32):
            rotating = SimpleNamespace(
                staged=True,
                runtime_shape=SimpleNamespace(batch=b),
                change_slots=torch.zeros((b, 10240), dtype=torch.int32),
                changes=torch.zeros((b, 10240), dtype=torch.int32),
                change_count=torch.zeros(b, dtype=torch.int32),
                rows=torch.zeros((b, 256), dtype=torch.int32),
                row_status=torch.zeros((b, 32), dtype=torch.int32),
                final_posts=torch.zeros((b, 4, 32), dtype=torch.int32),
                final_workers=4,
                hot_valid=torch.ones((b, 128), dtype=torch.int32),
                prefetch_status=torch.zeros((b * 2, 1216), dtype=torch.uint8),
                final_status=torch.zeros((b * 2, 1216), dtype=torch.uint8),
                owner=SimpleNamespace(abi_sizes={"status": 1216}),
            )
            inputs = dict(
                keys=torch.zeros((b, 256, 128), dtype=torch.int8),
                scales=torch.ones((b, 256)),
                addresses=torch.zeros((b, 256), dtype=torch.int64),
                logical_ids=torch.zeros((b, 256), dtype=torch.int64),
                row_owners=torch.arange(b, dtype=torch.int32),
                rotating=rotating,
            )
            update = DirectIndexUpdate(torch, **inputs)
            self.assertEqual(update.bindings.numel(), 28)
            self.assertEqual(update.bindings[14].item(), 4)
            self.assertEqual(update.bindings[7].item(), 256)
            rotating.staged = False
            with self.assertRaisesRegex(ValueError, "1..16"):
                DirectIndexUpdate(torch, **inputs)

    def test_existing_and_staged_python_binding(self):
        subprocess.run([sys.executable, str(ROOT / "experiments/split_core/tests/check_staged_binding.py")], check=True)

    def test_selector_validates_active_owners_after_startup(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import torch

        sys.path.insert(0, str(ROOT / "python"))
        from kvip.adapters.vllm.selector import VllmIndexSelector

        owners = torch.zeros(3, dtype=torch.int32)
        inputs = SimpleNamespace(
            batch=3,
            row_owners=owners,
            prepare=Mock(),
            query=torch.zeros(1),
            weights=torch.zeros(1),
            scale=torch.ones(1),
        )
        topk = torch.zeros((3, 4, 2048), dtype=torch.int32)
        args = [None] * 10
        args[0], args[2], args[3], args[9] = inputs.query, inputs.weights, inputs.scale, topk
        rotating = SimpleNamespace(staged=True, arguments=args, locations=topk)
        selector = VllmIndexSelector(
            torch,
            inputs=inputs,
            rotating=rotating,
            update=object(),
            logical_ids=torch.zeros((8, 8192), dtype=torch.int64),
            row_owners=owners,
            prefix_workspace=torch.zeros(1),
            forward_factory=lambda *a: object(),
        )
        self.assertTrue(selector.unique_active_owners)

        def plan(owner, count):
            return SimpleNamespace(owner=owner, begin=0, stop=count, decode=True)

        # Padded rows share owner zero at startup but never reserve a slot.
        selector.prepare([plan(7, 1), plan(0, 0), plan(0, 0)])
        selector.inputs.prepare.assert_called_once()
        with self.assertRaisesRegex(ValueError, "distinct stable owners"):
            selector.prepare([plan(7, 1), plan(7, 2)])

    def test_profile_reserves_shared_score_and_rebinds_events(self):
        import torch
        from unittest.mock import patch

        sys.path.insert(0, str(ROOT / "python"))
        from kvip.operators.lifetch.workspace import LIFetchWorkspace

        for batch, windows in ((1, 8), (3, 2)):
            workspace = LIFetchWorkspace(
                torch,
                batch=batch,
                history=8192,
                aics=4,
                windows=windows,
                device=torch.device("cpu"),
                pin_memory=False,
                index_heads=32,
            )
            workspace.allocate_rotation(torch, batch=batch, history=8192, resident=6144, aics=4, windows=windows)
            old_bytes = workspace.nbytes
            workspace.reserve_split_score(torch, resident=6144, aics=4, aivs=16, windows=windows, route="auto")
            self.assertEqual(tuple(workspace.scratch["events"].shape), (windows, 2, 128))
            self.assertGreater(workspace.nbytes, old_bytes)
            pointers = {key: value.data_ptr() for key, value in workspace.streaming_scratch.items()}
            with patch.object(torch, "empty", side_effect=AssertionError("Repeated score allocation")):
                workspace.reserve_split_score(torch, resident=6144, aics=4, aivs=16, windows=windows, route="auto")
            self.assertEqual(pointers, {key: value.data_ptr() for key, value in workspace.streaming_scratch.items()})

    def test_cpp_budget_shape_and_sidecar_layout(self):
        # Real host C++ validation: ABI sizes, original guard vs opt-in B32,
        # owner capacity and independent immutable physical budgets.
        code = """#include "staged_schedule.h"
#include "runtime_shape.h"
#include <cassert>
int main(){
 static_assert(sizeof(kvip_staged::ScoreWave)==40);
 static_assert(sizeof(kvip_staged::ScoreTask)==16);
 static_assert(sizeof(kvip_staged::Plan)==64);
 rotating::RuntimeShape s{};s.batch=32;s.history=131072;s.resident=16384;
 s.indexHeads=32;s.aicCount=28;s.tokenBytes=656;s.rounds=2;
 assert(rotating::PrepareShape(&s)!=0);
 assert(kvip_staged::PrepareShape(&s,56,2)==0);
 assert(s.batch==32 && s.serviceLanes==2 && s.finalWorkers==4);
 assert(kvip_staged::OwnerCapacity(56,2)==16);
 assert(kvip_staged::OwnerCapacity(8,2)==0);
}"""
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "test.cpp"
            source.write_text(code)
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-I" + str(ROOT / "csrc/operators/common/lifetch"),
                    "-I" + str(ROOT / "csrc/operators/lightning_indexer_with_fetch/include"),
                    str(source),
                    "-o",
                    d + "/test",
                ],
                check=True,
            )
            subprocess.run([d + "/test"], check=True)


if __name__ == "__main__":
    unittest.main()
