# SPDX-License-Identifier: Apache-2.0
"""Temporary KV must not change the ABC ring or the default ABI."""

from unittest.mock import Mock

import numpy as np
import pytest

from kvip.operators.lifetch.auto_route import AutoRouteState
from test_threshold_binding import Device, fixture


def setup(monkeypatch):
    f = fixture(monkeypatch)
    f.library.kvip_auto_route.return_value = 1
    f.library.kvip_adaptive_admission.return_value = 1
    f.library.kvip_low_miss_read_through = Mock(return_value=1)
    f.library.v9_context_bytes.return_value = 352
    auto = AutoRouteState(f.torch, Device(), 2, 3)
    original = f.metadata["output_stride"]
    config = np.array([0x52544831, original, original, 8192], dtype=np.uint64)
    return f, auto, config


def test_read_through_retains_original_abc_and_persistent_config(monkeypatch):
    f, auto, config = setup(monkeypatch)
    bound = f.construct(auto_route=auto, low_scratch_config=config,
                        output_stride=int(config[1]) + 8192)
    ptr = bound.low_scratch_config.data_ptr()
    before = bound.low_scratch_config.values.copy()
    assert len(bound.context.values) == 352
    assert int.from_bytes(bound.context.values[344:352], "little") == ptr
    config[:] = 0  # Caller buffer is not kept as a mutable alias.
    bound.reset()
    assert bound.low_scratch_config.data_ptr() == ptr
    np.testing.assert_array_equal(bound.low_scratch_config.values, before)
    no_scratch = f.construct(auto_route=auto)
    assert len(no_scratch.context.values) == 352
    assert int.from_bytes(no_scratch.context.values[344:352], "little") == 0


@pytest.mark.parametrize("fault", ["magic", "overlap", "capacity", "stride", "dtype", "shape", "abc", "auto", "feature"])
def test_invalid_scratch_rejected_before_launch(monkeypatch, fault):
    f, auto, config = setup(monkeypatch)
    stride = int(config[1]) + 8192
    if fault == "magic": config[0] = 0
    elif fault == "overlap": config[2] -= 128
    elif fault == "capacity": config[3] = 512
    elif fault == "stride": stride -= 128
    elif fault == "dtype": config = config.astype(np.int32)
    elif fault == "shape": config = config[:3]
    elif fault == "abc":
        config[1:3] -= 256
        stride -= 256
    elif fault == "auto": auto = None
    elif fault == "feature": f.library.kvip_low_miss_read_through.return_value = 0
    with pytest.raises(ValueError):
        f.construct(auto_route=auto, low_scratch_config=config, output_stride=stride)
    f.library.v9_init.assert_not_called()
    f.library.v9_li.assert_not_called()
