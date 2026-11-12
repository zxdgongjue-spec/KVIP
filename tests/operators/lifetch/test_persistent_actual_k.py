# SPDX-License-Identifier: Apache-2.0
"""CPU tests for P0-A2.2b persistent Actual-K growth."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from persistent_actual_k import (  # noqa: E402
    persistent_growth_schedule,
    score_oracle_for_persistent_state,
)


def test_growth_schedule_is_monotonic_and_reaches_history():
    schedule = persistent_growth_schedule(
        (32768, 65536, 98304, 131072), 131072, 8192
    )
    assert [label for label, _ in schedule] == [
        "initial", "grow_1", "grow_2", "grow_3"
    ]
    assert [values for _, values in schedule] == [
        (32768, 65536, 98304, 131072),
        (65536, 98304, 131072, 131072),
        (98304, 131072, 131072, 131072),
        (131072, 131072, 131072, 131072),
    ]
    for (_, before), (_, after) in zip(schedule, schedule[1:]):
        assert all(new >= old for old, new in zip(before, after))


@pytest.mark.parametrize("initial,history,resident,error", [
    ((32768, 65536, 98304), 131072, 8192, "batch4"),
    ((32768, 65536, 98304, 131072), 131072, 8192, None),
    ((32768, 65536, 98304, 130944), 131072, 8192, "full_history_terminal"),
    ((4096, 65536, 98304, 131072), 131072, 8192, "invalid_geometry"),
    ((32768, 65536, 98304, 131073), 131072, 8192, "invalid_geometry"),
    ((131072, 131072, 131072, 131072), 131072, 8192, "multiple_initial"),
])
def test_growth_schedule_admission(initial, history, resident, error):
    if error is None:
        assert persistent_growth_schedule(initial, history, resident)
    else:
        with pytest.raises(ValueError, match=error):
            persistent_growth_schedule(initial, history, resident)


def test_state_oracle_composes_base_and_current_physical_mappings():
    history, resident = 4096, 2048
    scores = np.tile(np.arange(history, dtype=np.float32), (4, 1))
    base_order = np.arange(history - 1, -1, -1, dtype=np.int32)
    current = np.tile(np.arange(history, dtype=np.int32), (2, 1))
    current[0, 10], current[0, 100] = current[0, 100], current[0, 10]
    current[1, 2048], current[1, 3000] = current[1, 3000], current[1, 2048]

    reports, selected = score_oracle_for_persistent_state(
        scores, base_order, current, (2048, 3072), resident
    )
    assert selected.shape == (2, 4, 2048)
    assert [x["actual_key_length"] for x in reports] == [2048, 3072]
    for row, key in enumerate((2048, 3072)):
        current_order = base_order[current[row]]
        for query in range(4):
            prefix = scores[query, current_order[:key]]
            expected = np.sort(np.argsort(prefix)[-2048:])
            assert np.array_equal(selected[row, query], expected)


def test_state_oracle_rejects_non_permutation_after_update():
    history = 4096
    scores = np.tile(np.arange(history, dtype=np.float32), (4, 1))
    base_order = np.arange(history, dtype=np.int32)
    current = np.tile(np.arange(history, dtype=np.int32), (1, 1))
    current[0, 1] = current[0, 0]
    with pytest.raises(ValueError, match="state_map_not_permutation"):
        score_oracle_for_persistent_state(
            scores, base_order, current, (2048,), 2048
        )


def test_state_oracle_rejects_boundary_ties():
    history = 4096
    scores = np.zeros((4, history), dtype=np.float32)
    order = np.arange(history, dtype=np.int32)
    current = order.reshape(1, -1)
    with pytest.raises(ValueError, match="boundary_tie"):
        score_oracle_for_persistent_state(
            scores, order, current, (2048,), 2048
        )


def test_cli_wires_persistent_actual_k_mode():
    root = Path(__file__).resolve().parents[3]
    cli = (root / "tests/operators/lifetch/test_lifetch.py").read_text()
    assert cli.count('add_argument("--actual-k-persistent-check"') == 2
    assert "common.append('--actual-k-persistent-check')" in cli
    assert "persistent_growth_schedule" in cli
