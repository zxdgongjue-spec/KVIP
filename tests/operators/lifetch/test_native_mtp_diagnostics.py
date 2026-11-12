# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for opt-in Native LIDU count diagnostics."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from native_mtp import NativeMtp  # noqa: E402


def test_counts_invalid_detects_negative_and_bad_total():
    assert NativeMtp._counts_invalid([-1, -1, -1, -1, 0], 4)
    assert NativeMtp._counts_invalid([1, 2, 3, 4, 9], 4)
    assert not NativeMtp._counts_invalid([1, 2, 3, 4, 10], 4)


def test_status_dump_contains_signed_lidu_error_codes():
    # Bypass NPU __init__; only test the deterministic diagnostic formatting.
    native = object.__new__(NativeMtp)
    native.diagnose = True
    native.batch = 2
    native.hot = 2048
    native.source_tokens = 8192
    native.profiling = torch.zeros((2, 8, 32), dtype=torch.int64)
    native.profiling[0, 0, 9] = -101
    native.profiling[1, 0, 9] = -110
    native.fixture = {
        "actual_query": torch.tensor([4, 8], dtype=torch.int32),
        "actual_key": torch.tensor([8192, 8192], dtype=torch.int32),
    }
    native.budgets = torch.tensor([2048, 2048], dtype=torch.int32)
    native.entries = torch.tensor([0, 1], dtype=torch.int32)
    native.destinations = torch.empty((8, 1, 2048), dtype=torch.int32)
    msg = str(native._invalid_counts_error([-1, -1, 0]))
    assert "ROW_METADATA" in msg
    assert "DESTINATION_SLOT" in msg
    assert "actual_query_cumulative" in msg
