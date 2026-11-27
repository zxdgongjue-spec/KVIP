"""The sparse large-batch KV reference must preserve page aliases and token order."""
from types import SimpleNamespace

import numpy as np
import pytest

from execution import NativeMtp
from workloads import ReplayPagePlan


@pytest.mark.parametrize('width', [656, 2624])
def test_host_reference_matches_raw_ddr_bytes(width):
    pages = ReplayPagePlan.__new__(ReplayPagePlan)
    pages.identities = ('row0',)
    # Cross page ends, aliases and distant physical addresses. The last partial
    # host chunk includes duplicated tokens in a different order.
    pages.page_indices = np.array([[2053, 0, 2053, 131071, 1, 16777215]], dtype=np.int32)
    pages.page_count = 16777216
    history = pages.page_indices.shape[1] * 128
    pages.geometry = dict(history=history, token_bytes=width)
    ddr = pages.page_payload(pages.page_indices[0])
    rng = np.random.default_rng(481)
    ids = np.r_[[0, 127, 128, history - 1, 127], rng.integers(history, size=8193)]
    result = pages.token_payload(0, ids)
    assert result.flags.c_contiguous and result.dtype == np.uint8
    np.testing.assert_array_equal(result, ddr[ids])
    assert pages.token_payload(0, []).shape == (0, width)
    for bad in (-1, history):
        with pytest.raises(ValueError, match='outside captured history'):
            pages.token_payload(0, [bad])


def test_large_serial_reference_matches_materialized_payload():
    torch = pytest.importorskip('torch')
    pages = ReplayPagePlan.__new__(ReplayPagePlan)
    pages.identities = ('row0', 'row1')
    pages.geometry = dict(history=384, token_bytes=656)
    pages.page_indices = np.array([[2, 7, 2], [7, 2, 7]])
    dense = torch.from_numpy(np.stack([pages.token_payload(row, np.arange(384)) for row in range(2)]))
    native = NativeMtp.__new__(NativeMtp)
    native.torch = torch
    native.owner = SimpleNamespace(session=SimpleNamespace(replay_pages=pages))
    ids = torch.tensor([383, 127, 128, 0, 256, 127])
    for row in range(2):
        assert torch.equal(native.cpu_tokens(row, ids), dense[row, ids])
    native.owner.session = SimpleNamespace(cpu_payload=dense)
    assert torch.equal(native.cpu_tokens(1, ids), dense[1, ids])
