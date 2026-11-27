# SPDX-License-Identifier: Apache-2.0
"""Shared LIFetch benchmark transport shutdown must preserve close semantics."""

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'python'), str(Path(__file__).resolve().parent / 'support')]
from execution import BenchPool  # noqa: E402


PUT_CLOSE_ERROR = 'Put transport teardown failed; resources retained'


@pytest.mark.parametrize('errors, attempts, recovered', [
    ([RuntimeError(PUT_CLOSE_ERROR), None], 2, True),
    ([RuntimeError(PUT_CLOSE_ERROR), RuntimeError(PUT_CLOSE_ERROR)], 2, False),
    ([RuntimeError('different runtime failure')], 1, False),
])
def test_bench_pool_retries_only_retained_put_shutdown(errors, attempts, recovered):
    pool = BenchPool(SimpleNamespace())
    pool.torch = SimpleNamespace(npu=Mock())
    runtime = pool.runtime = Mock()
    pool.tables = {1: ('table', 'view')}
    runtime.shutdown.side_effect = errors

    if recovered:
        pool.close()
        assert pool.runtime is None
    else:
        with pytest.raises(RuntimeError) as failure:
            pool.close()
        assert failure.value is errors[-1]
        assert pool.runtime is runtime

    pool.torch.npu.synchronize.assert_called_once_with()
    runtime.release_access_table.assert_called_once_with('table')
    assert runtime.shutdown.call_count == attempts
    assert pool.shutdown_retries == attempts - 1
