# SPDX-License-Identifier: Apache-2.0
"""CPU tests for P0-A2.2a K sequence and opt-in wiring."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from actual_k_transitions import transition_schedule  # noqa: E402


def test_long_short_long_mixed_restore():
    initial = (32768, 65536, 98304, 131072)
    schedule = transition_schedule(initial, 131072, 8192)
    assert [label for label, _ in schedule] == [
        "long_before_shrink", "short_after_long",
        "long_after_shrink", "restore_mixed_K"]
    assert schedule[0][1] == (131072,) * 4
    assert schedule[1][1] == (32768,) * 4
    assert schedule[2][1] == schedule[0][1]
    assert schedule[-1][1] == initial


def test_cannot_transition_if_all_k_are_full():
    with pytest.raises(ValueError, match="requires_short_request"):
        transition_schedule((131072,) * 4, 131072, 8192)


@pytest.mark.parametrize("values,history,resident", [
    ((4096, 8192), 8192, 8192),
    ((32769, 65536), 131072, 8192),
    ((32768, 131200), 131072, 8192),
])
def test_invalid_geometry_rejected(values, history, resident):
    with pytest.raises(ValueError, match="dynamic_actual_k_"):
        transition_schedule(values, history, resident)


def test_optin_flag_wired_to_replay():
    root = Path(__file__).resolve().parents[3]
    cli = (root / "tests/operators/lifetch/test_lifetch.py").read_text()
    runner = (root / "tests/operators/lifetch/support/lifetch.py").read_text()
    assert cli.count('add_argument("--actual-k-transition-check"') == 2
    assert "common.append('--actual-k-transition-check')" in cli
    assert 'getattr(args, "actual_k_transition_check", False)' in runner
    assert 'dynamic_k_transition_report' in runner
    assert "actual_key_buffer.copy_(device_k)" in runner
    assert "rotating.query_ends.copy_(device_k[:, None].expand(b, 4))" in runner
