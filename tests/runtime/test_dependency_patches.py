"""Check the pinned dependency and runtime preflight."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import call, patch


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("kvip_dependency_build", SOURCE_ROOT / "scripts/build.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class DependencyPatchTests(unittest.TestCase):
    def test_wrong_revision_stops_before_patching(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(BUILD.subprocess, "check_output", return_value="old-sdk"):
                with self.assertRaisesRegex(RuntimeError, BUILD.REVISION):
                    BUILD.prepare_dependency(Path(directory), "urma")


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
