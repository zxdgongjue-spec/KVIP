"""Runner contract tests, requiring no CANN or Git history."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "tools/compare_kernels.py"
SPEC = importlib.util.spec_from_file_location("compare_kernels", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class MatrixContractTests(unittest.TestCase):
    def import_fixture(self, directory):
        existing = directory / "existing"
        snapshot = existing / "sources/legacy"
        snapshot.mkdir(parents=True)
        for filename in RUNNER.SOURCE_FILES:
            shutil.copyfile(RUNNER.ROOT / "variants/v0_base/op_kernel" / filename, snapshot / filename)
        old = {
            "name": "legacy",
            "revision": "test",
            "source_sha256": {name: RUNNER.sha256(snapshot / name) for name in RUNNER.SOURCE_FILES},
        }
        original = {"variants": [old], "cases": [{"name": "low", "batch": 1, "unique_misses": 300}]}
        RUNNER.write_json(existing / "manifest.json", original)
        variants = [{"name": "v0_base", "path": "variants/v0_base", "role": "baseline"}]
        output = directory / "output"
        output.mkdir()
        return existing, snapshot, original, variants, output

    def test_import_requires_matching_archived_headers(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing, snapshot, original, variants, output = self.import_fixture(Path(temporary))
            imported = RUNNER.import_manifest(existing, variants, [], output, None, None)
            self.assertEqual(imported["variants"][0]["name"], "v0_base")
            self.assertTrue(imported["variants"][0]["ranking_eligible"])
            header = snapshot / RUNNER.SOURCE_FILES[1]
            header.write_bytes(header.read_bytes() + b"\n// modified header\n")
            with self.assertRaisesRegex(ValueError, "do not match"):
                RUNNER.import_manifest(existing, variants, [], output, None, None)
            original["variants"][0]["source_sha256"][header.name] = RUNNER.sha256(header)
            RUNNER.write_json(existing / "manifest.json", original)
            imported = RUNNER.import_manifest(existing, variants, [], output, None, None)
            self.assertEqual(imported["variants"][0]["name"], "legacy")
            self.assertFalse(imported["variants"][0]["ranking_eligible"])

    def test_import_rejects_missing_source_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing, snapshot, _, variants, output = self.import_fixture(Path(temporary))
            (snapshot / RUNNER.SOURCE_FILES[2]).unlink()
            with self.assertRaisesRegex(ValueError, "requires all three"):
                RUNNER.import_manifest(existing, variants, [], output, None, None)

    def test_unrecognized_source_cannot_be_a_reference_or_winner(self):
        rows = [
            {
                "variant": "v0_base",
                "case": "low",
                "correctness": "pass",
                "median_total_cycles": 1,
                "ranking_eligible": False,
            },
            {"variant": "v2_low", "case": "low", "correctness": "pass", "median_total_cycles": 80},
        ]
        winners = RUNNER.add_comparisons(rows, [{"name": "low", "unique_misses": 300}])
        self.assertEqual(winners, {"low": "v2_low"})
        self.assertTrue(all(r["improvement_vs_reference_pct"] is None for r in rows))

    def test_unknown_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            RUNNER.select([{"name": "v0_base"}], ["typo"])

    def test_geometry_and_extended_boundary(self):
        self.assertEqual(RUNNER.core_geometry(1), (8, 8))
        self.assertEqual(RUNNER.core_geometry(8), (32, 4))
        self.assertEqual(RUNNER.core_geometry(33), (32, 1))
        self.assertEqual(RUNNER.canonical_reference({"unique_misses": 2048}), "v0_base")
        self.assertEqual(RUNNER.canonical_reference({"unique_misses": 2049}), "p0_extended_reference")

    def test_heterogeneous_case_preserves_per_request_geometry(self):
        case = RUNNER.normalize_case(
            {"name": "mixed", "batch": 3, "query_counts": [2, 3, 4], "miss_counts": [0, 300, 4096]}
        )
        self.assertEqual(case["query_counts"], [2, 3, 4])
        self.assertEqual(RUNNER.canonical_reference(case), "p0_extended_reference")
        with self.assertRaises(ValueError):
            RUNNER.normalize_case({"name": "bad", "batch": 2, "query_counts": [2]})

    def test_unimplemented_layout_mutations_are_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "dedicated contract oracle"):
            RUNNER.normalize_case({"name": "padding", "batch": 4, "mutation": "padding_last_row"})

    def test_failed_baseline_and_candidate_are_never_ranked(self):
        rows = [
            {"variant": "v0_base", "case": "low", "correctness": "fail", "median_total_cycles": 100},
            {"variant": "v2_low", "case": "low", "correctness": "pass", "median_total_cycles": 80},
            {"variant": "bad", "case": "low", "correctness": "fail", "median_total_cycles": 1},
        ]
        winners = RUNNER.add_comparisons(rows, [{"name": "low", "unique_misses": 300}])
        self.assertEqual(winners, {"low": "v2_low"})
        self.assertTrue(all(r["improvement_vs_reference_pct"] is None for r in rows))

    def test_reduction_uses_explicit_reference(self):
        rows = [
            {"variant": "v2_low", "case": "low", "correctness": "pass", "median_total_cycles": 80},
            {"variant": "v0_base", "case": "low", "correctness": "pass", "median_total_cycles": 100},
        ]
        RUNNER.add_comparisons(rows, [{"name": "low", "unique_misses": 300}])
        self.assertAlmostEqual(rows[0]["improvement_vs_reference_pct"], 20)
        self.assertEqual(RUNNER.add_comparisons(rows, [{"name": "low", "unique_misses": 300}], False), {})

    def test_input_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "input.bin"
            path.write_bytes(b"shared input")
            hashes = {path.name: RUNNER.sha256(path)}
            RUNNER.verify_input_hashes(directory, hashes)
            path.write_bytes(b"different input")
            with self.assertRaises(AssertionError):
                RUNNER.verify_input_hashes(directory, hashes)

    def test_active_outputs_must_all_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertEqual(RUNNER.sample_directories(directory, 1)[1], "last_sample_only_active_1")
            with self.assertRaises(AssertionError):
                RUNNER.sample_directories(directory, 3)
            for index in range(3):
                (directory / "samples" / str(index)).mkdir(parents=True)
            self.assertEqual(len(RUNNER.sample_directories(directory, 3)[0]), 3)

    def test_full_oracle_checks_each_active_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            spec = RUNNER.normalize_case({"name": "oracle_probe", "batch": 1, "unique_misses": 300})
            cases = RUNNER.prepare_cases(output, [spec], 0, 2, 1)
            RUNNER.write_json(
                output / "manifest.json",
                {
                    "variants": [{"name": "v0_base", "revision": "test"}],
                    "cases": cases,
                    "active": 2,
                    "warmup": 0,
                    "profile_enabled": 1,
                },
            )
            oracle = RUNNER.load_metrics().load_oracle()
            case = RUNNER.make_case(oracle, spec)
            expected = oracle.reference_update(case, layer_id=2, num_layers=4, qcap=1024)
            run = output / "runs/v0_base/oracle_probe"
            run.mkdir(parents=True)
            for path in (output / "cases/oracle_probe").iterdir():
                shutil.copyfile(path, run / path.name)
            outputs = {
                "destination_slots": expected.destination_slots,
                "miss_counts": expected.miss_counts,
                "indices": expected.indices,
                "offsets": expected.offsets,
                "output_indices": expected.output_indices,
                "resident_seq_lengths": expected.resident_seq_lengths,
                "token_to_slot_pool": expected.token_to_slot_pool,
                "slot_to_token_pool": expected.slot_to_token_pool,
                "eviction_cursors": expected.eviction_cursors,
            }
            for index in range(2):
                sample = run / "samples" / str(index)
                sample.mkdir(parents=True)
                for name, value in outputs.items():
                    np.ascontiguousarray(value).tofile(sample / f"actual_{name}.bin")
            profile = np.zeros((2, 1, 8, 32), dtype="<i8")
            profile[:, 0, 0, 0] = 100
            profile.tofile(run / "profiling.bin")
            np.zeros(2, dtype="<f8").tofile(run / "host_microseconds.bin")
            self.assertEqual(RUNNER.verify(output)["status"], "pass")
            np.zeros(2, dtype="<i4").tofile(run / "samples/0/actual_miss_counts.bin")
            result = RUNNER.verify(output)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["best_variant_by_case"], {})


if __name__ == "__main__":
    unittest.main()
