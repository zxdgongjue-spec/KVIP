"""Exercise dependency patch upgrades with real git apply and isolated fixtures."""

import difflib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import call, patch


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("kvip_dependency_build", SOURCE_ROOT / "scripts/build.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def diff(before, after, name="runtime.txt"):
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{name}",
            tofile=f"b/{name}",
        )
    )


class DependencyPatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "kvaf"
        self.source.mkdir()
        self.git("init", "-q")
        rows = [f"base {index}\n" for index in range(40)]
        self.stages = ["".join(rows)]
        # Match the real chain: the output-view patch replaces fused-context
        # hunks, while RA and background Put have independent source contexts.
        for index, value in (
            (0, "reuse RA"),
            (10, "fused"),
            (10, "output views"),
            (20, "background Put"),
            (30, "UBmem"),
        ):
            rows[index] = value + "\n"
            self.stages.append("".join(rows))
        (self.source / "runtime.txt").write_text(self.stages[0])
        (self.source / "unrelated.txt").write_text("original\n")
        self.git("add", ".")
        self.git(
            "-c", "user.name=Patch fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "test fixture"
        )
        revision = self.git("rev-parse", "HEAD").strip()
        (self.root / "scripts/patches").mkdir(parents=True)
        (self.root / "patches").mkdir()
        for index, name in enumerate(BUILD.PATCHES):
            data = diff(self.stages[index], self.stages[index + 1])
            (self.root / "scripts/patches" / f"{name}.patch").write_text(data)
        (self.root / "patches/kvaf-background-put.patch").write_text(diff(self.stages[3], self.stages[4]))
        (self.root / "patches/kvaf-ubmem.patch").write_text(diff(self.stages[4], self.stages[5]))
        (self.root / "patches/kvaf-urma-get-post32.patch").write_text(
            diff("original\n", "optional post32\n", "unrelated.txt")
        )
        (self.root / "patches/kvaf-urma-get-bulk.patch").write_text(
            diff("optional post32\n", "optional post32 bulk\n", "unrelated.txt")
        )
        self.tiled = self.stages[5].replace("UBmem\n", "UBmem tiled\n")
        (self.root / "patches/kvaf-ubmem-get-dispatch.patch").write_text(diff(self.stages[5], self.tiled))
        self.root_patch = patch.object(BUILD, "ROOT", self.root)
        self.revision_patch = patch.object(BUILD, "REVISION", revision)
        self.root_patch.start()
        self.revision_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.revision_patch.stop)

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.source), *args],
            text=True,
            stderr=subprocess.PIPE,
        )

    def text(self):
        return (self.source / "runtime.txt").read_text()

    def legacy(self):
        for name in BUILD.PATCHES:
            BUILD.apply_patch(self.source, name)
        self.assertEqual(self.text(), self.stages[3])

    def test_default_urma_retains_background_put_without_ubmem(self):
        BUILD.prepare_dependency(self.source)
        self.assertEqual(self.text(), self.stages[4])
        BUILD.prepare_dependency(self.source)
        self.assertEqual(self.text(), self.stages[4])

    def test_optional_post32_is_idempotent_and_default_path_does_not_apply_it(self):
        BUILD.prepare_dependency(self.source)
        self.assertEqual((self.source / "unrelated.txt").read_text(), "original\n")
        BUILD.prepare_dependency(self.source, urma_get_post32=True)
        self.assertEqual((self.source / "unrelated.txt").read_text(), "optional post32\n")
        before = self.git("diff")
        BUILD.prepare_dependency(self.source, urma_get_post32=True)
        BUILD.prepare_dependency(self.source)
        self.assertEqual(self.git("diff"), before)

    def test_post32_rejects_ubmem_before_mutation_and_preserves_conflicts(self):
        with self.assertRaisesRegex(ValueError, "requires --transport urma"):
            BUILD.prepare_dependency(self.source, "ubmem", urma_get_post32=True)
        self.assertEqual(self.text(), self.stages[0])
        BUILD.prepare_dependency(self.source)
        (self.source / "unrelated.txt").write_text("user edit\n")
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            BUILD.prepare_dependency(self.source, urma_get_post32=True)
        self.assertEqual((self.source / "unrelated.txt").read_text(), "user edit\n")

    def test_post32_prepare_only_preserves_manifest_and_does_not_build(self):
        (self.root / "build").mkdir()
        manifest = self.root / "build/dependencies.json"
        manifest.write_text("existing selected build")
        with (
            patch.object(
                sys, "argv", ["build.py", "--prepare-only", "--urma-get-post32", "--kvaf-root", str(self.source)]
            ),
            patch.object(BUILD, "build_urma_post32") as build,
        ):
            BUILD.main()
        build.assert_not_called()
        self.assertEqual(manifest.read_text(), "existing selected build")
        self.assertEqual((self.source / "unrelated.txt").read_text(), "optional post32\n")

    def test_post32_incremental_build_uses_explicit_nondefault_target(self):
        build_dir = self.source / "build-npu"
        build_dir.mkdir()
        with self.assertRaisesRegex(RuntimeError, "configured"):
            BUILD.build_urma_post32(build_dir, 2, {})
        (build_dir / "CMakeCache.txt").touch()
        (build_dir / "bin").mkdir()
        (build_dir / "bin/kvaf_execute_batch_get_post32.o").touch()
        with patch.object(BUILD, "call") as run:
            BUILD.build_urma_post32(build_dir, 2, {"PYTHON": "serving-python"})
        self.assertEqual(
            run.call_args_list,
            [
                call(["cmake", "-S", self.source, "-B", build_dir], env={"PYTHON": "serving-python"}),
                call(
                    [
                        "cmake",
                        "--build",
                        build_dir,
                        "--target",
                        "kvaf_execute_batch_get_post32_kernel",
                        "--parallel",
                        2,
                    ],
                    env={"PYTHON": "serving-python"},
                ),
            ],
        )

    def test_fresh_ubmem_and_repeat(self):
        BUILD.prepare_dependency(self.source, "ubmem")
        self.assertEqual(self.text(), self.stages[5])
        before = self.git("diff")
        BUILD.prepare_dependency(self.source, "ubmem")
        self.assertEqual(self.git("diff"), before)

    def test_tiled_upgrade_repeat_preserves_real_index_and_unrelated_work(self):
        BUILD.prepare_dependency(self.source, "ubmem")
        (self.source / "unrelated.txt").write_text("staged user work\n")
        self.git("add", "unrelated.txt")
        (self.source / "notes.txt").write_text("untracked user work\n")
        index = self.git("diff", "--cached")
        BUILD.prepare_dependency(self.source, "ubmem", ubmem_get_tiled=True)
        self.assertEqual(self.text(), self.tiled)
        self.assertFalse(BUILD.applies(self.source, BUILD.UBMEM_PATCH, True))
        self.assertTrue(BUILD.verify_applied_patch_stack(self.source, (BUILD.UBMEM_PATCH, BUILD.UBMEM_DISPATCH_PATCH)))
        before = self.git("diff")
        BUILD.prepare_dependency(self.source, "ubmem", ubmem_get_tiled=True)
        BUILD.prepare_dependency(self.source, "ubmem")
        self.assertEqual(self.git("diff"), before)
        self.assertEqual(self.git("diff", "--cached"), index)
        self.assertEqual((self.source / "notes.txt").read_text(), "untracked user work\n")

    def test_tiled_fresh_conflict_and_wrong_transport(self):
        with self.assertRaisesRegex(ValueError, "requires --transport ubmem"):
            BUILD.prepare_dependency(self.source, ubmem_get_tiled=True)
        BUILD.prepare_dependency(self.source, "ubmem", ubmem_get_tiled=True)
        self.assertEqual(self.text(), self.tiled)
        with self.assertRaisesRegex(RuntimeError, "separate URMA"):
            BUILD.prepare_dependency(self.source)
        (self.source / "runtime.txt").write_text(self.tiled.replace("UBmem tiled", "user change"))
        before = self.git("diff")
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            BUILD.prepare_dependency(self.source, "ubmem", ubmem_get_tiled=True)
        self.assertEqual(self.git("diff"), before)

    def test_tiled_kernel_only_does_not_run_other_builds_or_switch_manifest(self):
        (self.root / "build").mkdir()
        manifest = self.root / "build/dependencies.json"
        manifest.write_text("existing selected build")
        with (
            patch.object(
                sys,
                "argv",
                [
                    "build.py",
                    "--transport",
                    "ubmem",
                    "--ubmem-get-tiled",
                    "--get-kernel-only",
                    "--kvaf-root",
                    str(self.source),
                ],
            ),
            patch.object(BUILD, "build_ubmem_tiled") as build,
            patch.object(BUILD, "runtime_preflight") as preflight,
            patch.object(BUILD, "build_lidu") as lidu,
        ):
            BUILD.main()
        build.assert_called_once()
        self.assertEqual(build.call_args.args[:2], (self.source.resolve() / "build-ubmem-npu", 2))
        preflight.assert_not_called()
        lidu.assert_not_called()
        self.assertEqual(manifest.read_text(), "existing selected build")
        self.assertEqual(self.text(), self.tiled)

    def test_tiled_incremental_build_targets_only_the_optional_object(self):
        build_dir = self.source / "build-ubmem-npu"
        build_dir.mkdir()
        with self.assertRaisesRegex(RuntimeError, "configured"):
            BUILD.build_ubmem_tiled(build_dir, 2, {})
        (build_dir / "CMakeCache.txt").touch()
        (build_dir / "bin").mkdir()
        (build_dir / "bin/kvaf_ubmem_batch_get_tiled.o").touch()
        with patch.object(BUILD, "call") as run:
            BUILD.build_ubmem_tiled(build_dir, 2, {"PYTHON": "serving-python"})
        self.assertEqual(
            run.call_args_list,
            [
                call(["cmake", "-S", self.source, "-B", build_dir], env={"PYTHON": "serving-python"}),
                call(
                    ["cmake", "--build", build_dir, "--target", "kvaf_ubmem_batch_get_tiled_kernel", "--parallel", 2],
                    env={"PYTHON": "serving-python"},
                ),
            ],
        )

    def test_bulk_upgrade_repeat_and_wrong_transport(self):
        with self.assertRaisesRegex(ValueError, "requires --transport urma"):
            BUILD.prepare_dependency(self.source, "ubmem", urma_get_bulk=True)
        BUILD.prepare_dependency(self.source, urma_get_bulk=True)
        self.assertEqual((self.source / "unrelated.txt").read_text(), "optional post32 bulk\n")
        self.assertFalse(BUILD.applies(self.source, BUILD.URMA_POST32_PATCH, True))
        before, staged = self.git("diff"), self.git("diff", "--cached")
        BUILD.prepare_dependency(self.source, urma_get_bulk=True)
        BUILD.prepare_dependency(self.source, urma_get_post32=True)
        BUILD.prepare_dependency(self.source)
        self.assertEqual(self.git("diff"), before)
        self.assertEqual(self.git("diff", "--cached"), staged)
        with self.assertRaisesRegex(RuntimeError, "separate UBmem"):
            BUILD.prepare_dependency(self.source, "ubmem")

    def test_bulk_conflict_preserves_source(self):
        BUILD.prepare_dependency(self.source, urma_get_post32=True)
        (self.source / "unrelated.txt").write_text("user change\n")
        before = self.git("diff")
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            BUILD.prepare_dependency(self.source, urma_get_bulk=True)
        self.assertEqual(self.git("diff"), before)

    def test_bulk_kernel_only_keeps_environment_and_skips_other_builds(self):
        (self.root / "build").mkdir()
        manifest = self.root / "build/dependencies.json"
        manifest.write_text("existing selected build")
        with (
            patch.object(
                sys, "argv", ["build.py", "--urma-get-bulk", "--get-kernel-only", "--kvaf-root", str(self.source)]
            ),
            patch.object(BUILD, "build_urma_bulk") as build,
            patch.object(BUILD, "build_urma_post32") as baseline,
            patch.object(BUILD, "runtime_preflight") as preflight,
            patch.object(BUILD, "build_lidu") as lidu,
        ):
            BUILD.main()
        build.assert_called_once()
        self.assertEqual(build.call_args.args[:2], (self.source.resolve() / "build-npu", 2))
        baseline.assert_not_called()
        preflight.assert_not_called()
        lidu.assert_not_called()
        self.assertEqual(manifest.read_text(), "existing selected build")

    def test_bulk_incremental_build_targets_only_the_optional_object(self):
        build_dir = self.source / "build-npu"
        build_dir.mkdir()
        with self.assertRaisesRegex(RuntimeError, "configured"):
            BUILD.build_urma_bulk(build_dir, 2, {})
        (build_dir / "CMakeCache.txt").touch()
        (build_dir / "bin").mkdir()
        (build_dir / "bin/kvaf_execute_batch_get_bulk.o").touch()
        with patch.object(BUILD, "call") as run:
            BUILD.build_urma_bulk(build_dir, 2, {})
        self.assertEqual(
            run.call_args_list,
            [
                call(["cmake", "-S", self.source, "-B", build_dir], env={}),
                call(
                    ["cmake", "--build", build_dir, "--target", "kvaf_execute_batch_get_bulk_kernel", "--parallel", 2],
                    env={},
                ),
            ],
        )

    def test_legacy_three_patch_urma_upgrade(self):
        self.legacy()
        BUILD.prepare_dependency(self.source)
        self.assertEqual(self.text(), self.stages[4])

    def test_legacy_three_patch_ubmem_upgrade(self):
        self.legacy()
        BUILD.prepare_dependency(self.source, "ubmem")
        self.assertEqual(self.text(), self.stages[5])

    def test_common_background_put_checkout_upgrades_to_ubmem(self):
        BUILD.prepare_dependency(self.source)
        BUILD.prepare_dependency(self.source, "ubmem")
        self.assertEqual(self.text(), self.stages[5])

    def test_unrelated_tracked_and_untracked_edits_survive(self):
        (self.source / "unrelated.txt").write_text("user work\n")
        (self.source / "notes.txt").write_text("local notes\n")
        BUILD.prepare_dependency(self.source, "ubmem")
        BUILD.prepare_dependency(self.source, "ubmem")
        self.assertEqual((self.source / "unrelated.txt").read_text(), "user work\n")
        self.assertEqual((self.source / "notes.txt").read_text(), "local notes\n")

    def test_conflict_stops_without_overwriting_user_edit(self):
        BUILD.prepare_dependency(self.source)
        (self.source / "runtime.txt").write_text(self.stages[4].replace("base 30", "user change"))
        before = self.git("diff")
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            BUILD.prepare_dependency(self.source, "ubmem")
        self.assertEqual(self.git("diff"), before)

    def test_background_put_upgrade_conflict_preserves_legacy_checkout(self):
        self.legacy()
        (self.source / "runtime.txt").write_text(self.stages[3].replace("base 20", "user change"))
        before = self.git("diff")
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            BUILD.prepare_dependency(self.source)
        self.assertEqual(self.git("diff"), before)

    def test_urma_does_not_silently_reuse_local_ubmem_source(self):
        BUILD.prepare_dependency(self.source, "ubmem")
        before = self.git("diff")
        with self.assertRaisesRegex(RuntimeError, "separate URMA checkout"):
            BUILD.prepare_dependency(self.source)
        self.assertEqual(self.git("diff"), before)

    def test_wrong_revision_preserves_checkout(self):
        before = self.text()
        with patch.object(BUILD, "REVISION", "wrong-revision"):
            with self.assertRaisesRegex(RuntimeError, "Expected KVAF"):
                BUILD.prepare_dependency(self.source, "ubmem")
        self.assertEqual(self.text(), before)

    def test_invalid_transport_preserves_checkout(self):
        with self.assertRaisesRegex(ValueError, "transport"):
            BUILD.prepare_dependency(self.source, "invalid")
        self.assertEqual(self.text(), self.stages[0])

    def test_prepare_only_keeps_selected_runtime_manifest(self):
        (self.root / "build").mkdir()
        manifest = self.root / "build/dependencies.json"
        original = json.dumps({"transport": "urma", "kvaf_build_dir": "existing-build"})
        manifest.write_text(original)
        with patch.object(
            sys, "argv", ["build.py", "--prepare-only", "--transport", "ubmem", "--kvaf-root", str(self.source)]
        ):
            BUILD.main()
        self.assertEqual(manifest.read_text(), original)


class RuntimePreflightTests(unittest.TestCase):
    def test_old_python_fails_before_import(self):
        with patch.object(sys, "version_info", (3, 11)), patch.object(BUILD.importlib, "import_module") as load:
            with self.assertRaisesRegex(RuntimeError, "Python >= 3.12"):
                BUILD.runtime_preflight()
            load.assert_not_called()

    def test_selected_python_loads_both_runtime_modules(self):
        with patch.object(sys, "version_info", (3, 12)), patch.object(BUILD.importlib, "import_module") as load:
            BUILD.runtime_preflight()
            self.assertEqual(load.call_args_list, [call("torch"), call("torch_npu")])

    def test_missing_or_unloadable_runtime_has_interpreter_context(self):
        for failure in (ModuleNotFoundError("missing"), OSError("missing CANN library")):
            with (
                self.subTest(failure=type(failure).__name__),
                patch.object(sys, "version_info", (3, 12)),
                patch.object(BUILD.importlib, "import_module", side_effect=[object(), failure]),
            ):
                with self.assertRaisesRegex(RuntimeError, "Cannot load torch_npu") as caught:
                    BUILD.runtime_preflight()
                self.assertIn(sys.executable, str(caught.exception))

    def test_failed_preflight_precedes_dependency_or_build_changes(self):
        with (
            patch.object(sys, "argv", ["build.py"]),
            patch.object(BUILD, "runtime_preflight", side_effect=RuntimeError("preflight failed")),
            patch.object(BUILD, "prepare_dependency") as prepare,
        ):
            with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                BUILD.main()
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
