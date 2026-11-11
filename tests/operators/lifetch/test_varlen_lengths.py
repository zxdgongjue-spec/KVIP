# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for publishing variable Q/K lengths into the Native LI ABI."""

import pytest
import torch

from kvip.operators.lifetch.workspace import LIFetchWorkspace


def make_workspace():
    return LIFetchWorkspace(
        torch,
        batch=4,
        history=256,
        aics=32,
        windows=1,
        device=torch.device("cpu"),
        pin_memory=False,
        index_heads=32,
    )


def test_bind_positions_updates_native_actual_lengths_and_clears_padding():
    workspace = make_workspace()
    inputs = workspace.inputs
    query_ptr = workspace.query_lengths.data_ptr()
    key_ptr = workspace.key_lengths.data_ptr()

    inputs.prepare(
        [4, 2, 1, 0],
        query_starts=[0, 4, 6, 7],
        query_owners=[0, 1, 2, 3],
    )
    positions = torch.tensor(
        [1000, 1001, 1002, 1003, 1000, 1001, 1000],
        dtype=torch.int64,
    )

    query_ends = inputs.bind_positions(positions)

    assert query_ends.tolist() == [
        [1001, 1002, 1003, 1004],
        [1001, 1002, 0, 0],
        [1001, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert workspace.query_lengths.tolist() == [4, 2, 1, 0]
    assert workspace.key_lengths.tolist() == [1004, 1002, 1001, 0]
    assert workspace.query_lengths.data_ptr() == query_ptr
    assert workspace.key_lengths.data_ptr() == key_ptr

    # A second forward must overwrite every row rather than retain prior lengths.
    inputs.prepare(
        [1, 0, 3, 0],
        query_starts=[0, 1, 1, 4],
        query_owners=[0, 1, 2, 3],
    )
    second_positions = torch.tensor([20, 30, 31, 32], dtype=torch.int64)
    second_ends = inputs.bind_positions(second_positions)

    assert second_ends.tolist() == [
        [21, 0, 0, 0],
        [0, 0, 0, 0],
        [31, 32, 33, 0],
        [0, 0, 0, 0],
    ]
    assert workspace.query_lengths.tolist() == [1, 0, 3, 0]
    assert workspace.key_lengths.tolist() == [21, 0, 33, 0]
    assert workspace.query_lengths.data_ptr() == query_ptr
    assert workspace.key_lengths.data_ptr() == key_ptr


def test_bind_actual_lengths_rejects_invalid_native_abi_buffers():
    workspace = make_workspace()
    invalid_query_lengths = torch.empty((4,), dtype=torch.int64)

    with pytest.raises(ValueError, match="contiguous int32"):
        workspace.inputs.bind_actual_lengths(
            invalid_query_lengths,
            workspace.key_lengths,
        )
