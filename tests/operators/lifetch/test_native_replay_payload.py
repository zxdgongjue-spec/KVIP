"""The sparse large-batch KV reference must preserve page aliases and token order."""
from types import SimpleNamespace

import numpy as np
import pytest

from execution import NativeMtp
from workloads import ReplayPagePlan


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
