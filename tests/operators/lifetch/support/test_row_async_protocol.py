# SPDX-License-Identifier: Apache-2.0
"""Run the native four-lane protocol against a FIFO and completion oracle."""

from pathlib import Path
import subprocess


def test_row_async_protocol(tmp_path):
    here = Path(__file__).resolve().parent
    kernel = here.parents[3] / 'csrc/operators/lightning_indexer_with_fetch/op_kernel'
    exe = tmp_path / 'protocol'
    subprocess.run(['c++', '-std=c++17', '-O1', '-pthread', '-fsanitize=address,undefined',
                    '-I', str(kernel), str(here / 'row_async_protocol_model.cpp'), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=30)
