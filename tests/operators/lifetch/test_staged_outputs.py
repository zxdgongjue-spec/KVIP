# SPDX-License-Identifier: Apache-2.0
"""Execute the production staged publisher with delayed DMA and UB reuse."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("batch,resident", ((1, 8192), (2, 16384), (16, 16384)))
@pytest.mark.parametrize("check_status", (False, True))
def test_staged_outputs_preserve_tnd_ownership_and_old_index(tmp_path, batch, resident, check_status):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the staged output model")
    root = Path(__file__).resolve().parents[3]
    binary = tmp_path / "staged-outputs"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-DMODEL_BATCH={batch}",
            f"-DMODEL_RESIDENT={resident}",
            f"-DMODEL_CHECK_STATUS={int(check_status)}",
            f"-I{root / 'csrc/operators/common/lifetch'}",
            str(root / "tests/operators/lifetch/staged_outputs_model.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
