# SPDX-License-Identifier: Apache-2.0
import subprocess
from pathlib import Path


def test_touched_plan_device_composition(tmp_path):
    here = Path(__file__).resolve().parent
    kernel = here.parents[3] / "csrc/operators/lightning_indexer_with_fetch/op_kernel"
    source = (kernel / "high_index_touched_plan.h").read_text()
    vf_body = source.split("__aicore__ __attribute__((noinline)) inline bool TryTouchedPlan", 1)[0]
    (tmp_path / "touched_device.h").write_text(vf_body)
    binary = tmp_path / "plan"
    subprocess.run(["c++", "-std=c++17", "-O1", "-pthread", "-fsanitize=address,undefined",
                    "-I", str(tmp_path), str(here / "touched_plan_model.cpp"), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=75)
