# SPDX-License-Identifier: Apache-2.0
"""Host contracts for persistent automatic-route state; no NPU readback."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from kvip.operators.lifetch.auto_route import AutoRouteState
from test_threshold_binding import Device, Tensor, fixture


def state(f, **kwargs):
    return AutoRouteState(f.torch, Device(), 2, 3, **kwargs)


def test_auto_state_context_and_metadata_do_not_change_addresses(monkeypatch):
    f = fixture(monkeypatch)
    f.library.kvip_auto_route.return_value = 1
    f.library.kvip_adaptive_admission.return_value = 1
    auto = state(f)
    bound = f.construct(auto_route=auto)
    assert bound.auto_route is auto
    assert len(bound.context.values) == 344
    assert int.from_bytes(bound.context.values[336:344], "little") == auto.config.data_ptr()
    pointers = tuple(t.data_ptr() for t in (auto.config, auto.requests, auto.feedback, auto.pending))
    words = auto.config.values.view(np.uint64)
    assert tuple(words[:2]) == (1, 3)
    assert tuple(words[2:5]) == pointers[1:]
    assert tuple(words[5:13]) == (128, 192, 1, 0, 512, 64, 0, 0)

    auto.feedback.values[1, :8] = [7, 11, 90, 4, 520000, 656, 0, 0x4B565250]
    feedback = auto.feedback.values.copy()
    for step in (12, 13):
        rows = np.array([[1, 7, step, 4], [2, 9, step, 0]], dtype=np.int64)
        auto.set_requests(rows)
        auto.set_learning(True)
        bound.reset()
        np.testing.assert_array_equal(auto.requests.values, rows)
        np.testing.assert_array_equal(auto.feedback.values, feedback)
    assert pointers == tuple(t.data_ptr() for t in (auto.config, auto.requests, auto.feedback, auto.pending))
    with pytest.raises(RuntimeError, match="owns admission"):
        bound.set_admission_controls(None)


def test_auto_requires_native_capability_and_exclusive_controls(monkeypatch):
    f = fixture(monkeypatch)
    auto = state(f)
    with pytest.raises(ValueError, match="auto_route=True"):
        f.construct(auto_route=auto)
    with pytest.raises(ValueError, match="owns admission"):
        f.construct(auto_route=auto, admission_controls=np.zeros((2, 4), dtype=np.int32))
    with pytest.raises(ValueError, match="stable owners"):
        f.construct(auto_route=AutoRouteState(f.torch, Device(), 2, 4))


@pytest.mark.parametrize("bad", ("dtype", "shape", "duplicate", "owner", "query", "negative", "device"))
def test_request_validation_does_not_upload_invalid_metadata(monkeypatch, bad):
    f = fixture(monkeypatch)
    auto = state(f)
    rows = np.array([[1, 1, 1, 4], [2, 1, 1, 4]], dtype=np.int64)
    if bad == "dtype":
        rows = rows.astype(np.int32)
    elif bad == "shape":
        rows = rows[:1]
    elif bad == "duplicate":
        rows[1, 0] = 1
    elif bad == "owner":
        rows[1, 0] = 3
    elif bad == "query":
        rows[1, 3] = 5
    elif bad == "negative":
        rows[1, 1] = -1
    elif bad == "device":
        rows = Tensor(rows)
    before = auto.requests.values.copy()
    with pytest.raises(ValueError):
        auto.set_requests(rows)
    np.testing.assert_array_equal(auto.requests.values, before)


def test_updates_are_outside_capture_and_never_synchronize(monkeypatch):
    f = fixture(monkeypatch)
    f.torch.npu = SimpleNamespace(is_current_stream_capturing=lambda: False,
                                  synchronize=Mock(side_effect=AssertionError("no host wait")))
    auto = state(f)
    auto.set_learning(True)
    assert auto.config.values[8] == 1
    auto.set_learning(False)
    assert auto.config.values[8] == 0
    f.torch.npu.synchronize.assert_not_called()
    f.torch.npu.is_current_stream_capturing = lambda: True
    with pytest.raises(RuntimeError, match="outside capture"):
        auto.set_learning(True)
    with pytest.raises(RuntimeError, match="outside capture"):
        auto.set_requests(np.zeros((2, 4), dtype=np.int64))


@pytest.mark.parametrize("kwargs", [dict(low_budget=192), dict(high_budget=128),
                                   dict(max_step_gap=0), dict(full_controls=(0, 64, 0, 0))])
def test_policy_configuration_rejects_invalid_bounds(monkeypatch, kwargs):
    with pytest.raises(ValueError):
        state(fixture(monkeypatch), **kwargs)
