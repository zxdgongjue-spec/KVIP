"""Exact publisher C++ bodies under ASan/UBSan and explicit event/range model."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
KERNEL = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel"


def function(text, anchor):
    start = text.index(anchor)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


class OutputPipelineTest(unittest.TestCase):
    def test_actual_cpp_and_synchronization_faults(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (KERNEL / "index_cache_update.cpp").read_text()
        bodies = "\n".join(
            function(source, anchor)
            for anchor in (
                "__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishOutputs(",
                "__simt_callee__ inline int32_t LookupOutputLogical(",
                "__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void GatherOutputLogical(",
                "template <bool Pipeline>\n__aicore__ inline void PublishOutputsPipelined(",
            )
        )
        directory = Path(__file__).parent / "support"
        prefix = (directory / "output_prefix.cpp.in").read_text().replace("// @ACTUAL_PUBLISHERS@", bodies)
        program = (directory / "output_pipeline.cpp.in").read_text().replace("@@PREFIX@@", prefix)
        reuse = "WaitFlag<HardEvent::MTE3_S>(done[bank]);"
        cases = {
            "actual": program,
            "missing_reuse_wait": program.replace(reuse, "/* missing reuse wait */", 1),
            "wrong_event_id": program.replace(reuse, "WaitFlag<HardEvent::MTE3_S>(done[bank] + 7);", 1),
            "missing_drain": program.replace("if (pending[bank]) " + reuse, "/* missing final drain */"),
            "constant_bank": program.replace("uint32_t bank = Pipeline ? q % 2 : 0;", "uint32_t bank = 0;"),
            "global_flush": program.replace("if(c.issue<=watermark)", "if(true)"),
        }
        with tempfile.TemporaryDirectory(prefix="kvip-pipeline-") as temporary:
            folder = Path(temporary)
            for name, cpp in cases.items():
                with self.subTest(name=name):
                    path, binary = folder / (name + ".cpp"), folder / name
                    path.write_text(cpp)
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
                    if sdk.exists():
                        command += ["-isysroot", str(sdk)]
                    run = subprocess.run(
                        command + [str(path), "-o", str(binary)], capture_output=True, text=True, timeout=60
                    )
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    run = subprocess.run(
                        [str(binary)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env={**os.environ, "UBSAN_OPTIONS": "halt_on_error=1"},
                    )
                    if name == "actual":
                        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                        self.assertEqual(len(run.stdout.splitlines()), 56)
                    else:
                        self.assertNotEqual(run.returncode, 0, name + " escaped")
                        self.assertTrue(
                            any(
                                s in run.stderr
                                for s in (
                                    "UB overwritten",
                                    "event ID",
                                    "before wait/drain",
                                    "overlap opportunities",
                                    "watermark",
                                )
                            ),
                            run.stderr,
                        )
