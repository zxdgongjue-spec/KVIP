"""Actual C++ publisher contracts, with modeled asynchronous copies.

These tests check outputs, UB bounds and explicit events. They do not model
NPU cache consistency, instruction timing, or real KV payload transport.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
KERNEL = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel"
SOURCE = KERNEL / "index_cache_update.cpp"


def extract_function(text, anchor):
    start = text.index(anchor)
    brace = text.index("{", start)
    end, depth = brace + 1, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


class OutputBulkTest(unittest.TestCase):
    def test_actual_publishers_preserve_outputs_and_scratch_dependencies(self):
        command_line_clang = Path("/Library/Developer/CommandLineTools/usr/bin/clang++")
        compiler = str(command_line_clang) if command_line_clang.is_file() else shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        source = SOURCE.read_text()
        # Exact source bodies, including both AICore wrappers; no translation.
        bodies = [
            extract_function(source, anchor)
            for anchor in (
                "__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishOutputs(",
                "__simt_callee__ inline int32_t LookupOutputLogical(",
                "__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void GatherOutputLogical(",
                "__aicore__ inline void PublishOutputsBulk(",
                "__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void GatherPackedOutputLogical(",
                "__aicore__ inline void PublishOutputsPacked(",
            )
        ]
        template = Path(__file__).with_name("support") / "output_bulk_native.cpp.in"
        program = template.read_text().replace("// @ACTUAL_PUBLISHERS@", "\n".join(bodies))
        self.assertNotIn("// @", program)
        with tempfile.TemporaryDirectory(prefix="kvip-output-bulk-") as temporary:
            directory = Path(temporary)
            cpp, binary = directory / "output_bulk.cpp", directory / "output_bulk"
            cpp.write_text(program)
            command = [
                compiler,
                "-std=c++17",
                "-O1",
                "-g",
                "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer",
                "-I" + str(KERNEL),
            ]
            sdk = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
            if Path(compiler).name == "clang++" and sdk.exists():
                command += ["-isysroot", str(sdk)]
            command += [str(cpp), "-o", str(binary)]
            build = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run(
                [str(binary)],
                text=True,
                capture_output=True,
                timeout=60,
                env={**os.environ, "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"},
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        rows = [line.split() for line in run.stdout.splitlines()]
        self.assertEqual(len(rows), 40)
        self.assertTrue(all(len(row) == 6 for row in rows))
        names = {row[0] for row in rows}
        self.assertTrue(
            {
                "nonzero_batch_packed_owners",
                "mixed_inactive_poison_unpack",
                "all_inactive_null_sources",
                "negative_tokens_and_slots",
                "all_negative_tokens",
                "last_destination_batch",
            }.issubset(names)
        )

    def test_cmake_rejects_invalid_optimization_values_before_cann(self):
        cmake = shutil.which("cmake")
        if cmake is None:
            self.skipTest("CMake unavailable")
        cases = (
            ("KVIP_OPT_LI_REGISTER_PREDICATE", "2", "must be 0 or 1"),
            ("KVIP_OPT_LI_REGISTER_PREDICATE", "ON", "must be 0 or 1"),
            ("KVIP_LI_SCALE_TILES", "16", "must be 0 or 8"),
            ("KVIP_OPT_OUTPUT_PINGPONG", "2", "must be 0 or 1"),
            ("KVIP_OPT_OUTPUT_BULK", "3", "must be 0, 1 or 2"),
            ("KVIP_OPT_OUTPUT_BULK", "-1", "must be 0, 1 or 2"),
        )
        for option, value, message in cases:
            with self.subTest(option=option, value=value), tempfile.TemporaryDirectory() as directory:
                configured = subprocess.run(
                    [cmake, "-S", str(KERNEL.parent), "-B", directory, f"-D{option}={value}"],
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(configured.returncode, 0)
                self.assertIn(message, configured.stdout + configured.stderr)

        for mode in (0, 2):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                configured = subprocess.run(
                    [
                        cmake,
                        "-S",
                        str(KERNEL.parent),
                        "-B",
                        directory,
                        "-DKVIP_OPT_OUTPUT_PINGPONG=1",
                        f"-DKVIP_OPT_OUTPUT_BULK={mode}",
                    ],
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(configured.returncode, 0)
                self.assertIn("requires KVIP_OPT_OUTPUT_BULK=1", configured.stdout + configured.stderr)


if __name__ == "__main__":
    unittest.main()
