"""Exercise real Torch buffer uploads without claiming device-kernel coverage."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from replay_device import FusedReplayState
from replay_inputs import ReplayPagePlan, assemble_round
from test_replay_inputs import reference


def cpu_state(tmp_path):
    inputs = assemble_round([reference(tmp_path)])
    pages = ReplayPagePlan([inputs])
    state = FusedReplayState.__new__(FusedReplayState)
    state.torch, state.device, state.pages, state.initial = torch, torch.device('cpu'), pages, inputs
    state.batch, state.history, state.resident, state.stride, state.token_bytes = 1, 4096, 2048, 2176, 656
    state.logical_addresses = np.arange(4096, dtype=np.int64)[None] * 656 + 0x100000
    state.storage = SimpleNamespace(
        logical_ids=torch.arange(4096).repeat(2, 1), keys=torch.zeros((2, 4096, 128), dtype=torch.uint8),
        scales=torch.ones((2, 4096)), hot_valid=torch.ones((2, 2048), dtype=torch.int32),
        addresses=torch.zeros((2, 4096), dtype=torch.int64))
    state.workspace = SimpleNamespace(inputs=SimpleNamespace(
        query=torch.zeros((4, 2, 128), dtype=torch.uint8),
        weights=torch.zeros((4, 2), dtype=torch.bfloat16), scale=torch.zeros((4, 2)),
        query_ends=torch.zeros(4, dtype=torch.int32)), tail_bounds=torch.zeros((1, 2), dtype=torch.int32))
    state.output = torch.zeros((2, 2176, 656), dtype=torch.uint8)
    state.rotating = SimpleNamespace(lru_slots=torch.empty((1, 2048), dtype=torch.int32),
                                    reserved_hot=torch.empty((1, 2048), dtype=torch.int32))
    state.policies = [SimpleNamespace(slots=np.arange(2048, dtype=np.int32)[None])]
    return state, inputs


def test_prepare_imports_bits_and_retains_current_state(tmp_path):
    state, inputs = cpu_state(tmp_path)
    state.storage.logical_ids[1, [0, 3000]] = state.storage.logical_ids[1, [3000, 0]]
    state.storage.hot_valid[1, 0] = 0
    inputs.weights[:] = 0xBF80  # -1 BF16, not float value 49024.
    state.prepare(inputs)
    assert torch.all(state.workspace.inputs.weights == -1)
    assert state.storage.scales[1, 0] == 1095
    assert state.storage.logical_ids[1, 0] == 3000 and state.storage.hot_valid[1, 0] == 0
    assert state.storage.addresses[1, 0] == 0x100000 + 3000 * 656
    assert state.rotating.lru_slots[0, -1] == 2047
    ids = np.arange(3968, 4096)
    np.testing.assert_array_equal(state.output[1, 2048:].numpy(), state.pages.token_payload(0, ids))


def test_completion_error_stops_before_reading_outputs(tmp_path):
    state, inputs = cpu_state(tmp_path)
    state.update = SimpleNamespace(status=torch.tensor([7]))
    with pytest.raises(RuntimeError, match='completion failed'):
        state.validate(inputs)


def valid_index_state(tmp_path):
    state, inputs = cpu_state(tmp_path)
    state.prepare(inputs)
    layout = state.storage.logical_ids[1].numpy()
    state.upload(state.output[1, :state.resident], state.pages.token_payload(0, layout[:state.resident]))
    return state, inputs


def test_all_index_fields_and_hot_bytes_checked_before_next_prepare(tmp_path):
    state, inputs = valid_index_state(tmp_path)
    state.validate_index(inputs, state.output[1:].numpy())
    # A non-first key column outside the selected TopK must still be checked.
    state.storage.keys[1, 3000, 127] ^= 1
    with pytest.raises(RuntimeError, match='Index keys mismatch'):
        state.validate_index(inputs, state.output[1:].numpy())


@pytest.mark.parametrize('field', ['scales', 'addresses', 'logical_ids', 'hot_valid', 'output'])
def test_persistent_state_corruption_is_fatal(tmp_path, field):
    state, inputs = valid_index_state(tmp_path)
    if field == 'output':
        state.output[1, 1, 655] ^= 1
    elif field == 'logical_ids':
        state.storage.logical_ids[1, 1] = 0  # Duplicate identity, not a permutation.
    elif field == 'hot_valid':
        state.storage.hot_valid[1, 1] = 2
    else:
        getattr(state.storage, field)[1, 3000] += 1
    with pytest.raises(RuntimeError, match='Replay'):
        state.validate_index(inputs, state.output[1:].numpy())


def test_urma_replay_allocates_complete_scratch_before_binding(monkeypatch, tmp_path):
    from kvip.adapters import kvaf
    from kvip.operators.lifetch import binding, workspace
    from kvip.operators.lifetch.geometry import final_aivs, li_geometry

    real_workspace = workspace.LIFetchWorkspace
    monkeypatch.setattr(workspace, 'LIFetchWorkspace',
                        lambda torch, **kwargs: real_workspace(torch, pin_memory=False, **kwargs))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(current_stream=lambda device: None), raising=False)
    lease = SimpleNamespace(sizes={'status': 1536}, close=lambda: closed.append(True))
    closed = []
    monkeypatch.setattr(kvaf, 'KvafGetLease', lambda *args: lease)
    monkeypatch.setattr(kvaf, 'KvafQueueView', lambda *args: None)
    shape = li_geometry(2, 'auto', 8, 24)
    expected = workspace.rotation_spec(2, 12288, 4096, final_aivs(2, 'auto', 24),
                                      shape['intermediate_fetches'],
                                      shape['service_aivs_per_window'], 1536)

    class BindingReached(Exception):
        pass

    def inspect_binding(*args, scratch, **kwargs):
        assert kwargs['output_stride'] - kwargs['resident'] >= 384
        assert kwargs['tail_bounds'].shape == (2, 2)
        assert set(scratch) == set(expected)
        for name, (size, dtype) in expected.items():
            assert tuple(scratch[name].shape) == size
            assert scratch[name].dtype == getattr(torch, dtype)
            assert torch.count_nonzero(scratch[name]) == 0
        raise BindingReached

    monkeypatch.setattr(binding, 'LIFetchBinding', inspect_binding)
    from correctness import stateful_inputs
    initial = next(iter(stateful_inputs(tmp_path)))
    with pytest.raises(BindingReached):
        FusedReplayState(torch, SimpleNamespace(transport='urma'),
                         SimpleNamespace(addresses=None, valid=None), None, initial, None, None,
                         device=torch.device('cpu'), aics=24, windows=8,
                         prefetch_tokens=0, soc='Ascend950PR_9599')
    assert closed == [True]
