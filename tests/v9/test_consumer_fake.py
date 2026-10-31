import inspect

import pytest

from kvip.testing.transport import FakeNpuRuntime, TimingContext


def make_fake(*, zero_task=False):
    torch = pytest.importorskip("torch")
    runtime = FakeNpuRuntime(
        torch=torch,
        batch_get_token_bytes=656,
        batch_get_tokens_per_block=128,
        zero_task=zero_task,
    )
    source = (
        torch.arange(128 * 656, dtype=torch.int32).to(torch.uint8).reshape(128, 656)
    )
    runtime.register_block(4096, source)
    output = torch.full((8, 656), 0xA5, dtype=torch.uint8)
    args = [
        torch.tensor([4096], dtype=torch.int64),
        torch.ones(1, dtype=torch.uint8),
        torch.tensor([0, 0, -999], dtype=torch.int32),
        torch.tensor([656, 2 * 656, -999], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([4, 6, -999], dtype=torch.int32),
        output,
        None,
        False,
    ]
    return runtime, source, output, args


def test_exact_nine_arguments_and_global_prefix_with_hit_tail_protection():
    assert (
        len(inspect.signature(FakeNpuRuntime.batch_get).parameters) == 10
    )  # self + nine
    runtime, source, output, args = make_fake()
    before = output.clone()
    runtime.batch_get(*args)
    torch = runtime.torch
    assert torch.equal(output[4], source[1]) and torch.equal(output[6], source[2])
    assert torch.equal(output[[0, 1, 2, 3, 5, 7]], before[[0, 1, 2, 3, 5, 7]])


@pytest.mark.parametrize("field,value", [(2, -1), (3, 1), (5, 8)])
def test_invalid_descriptor_is_rejected_before_any_writes(field, value):
    runtime, _, output, args = make_fake()
    before = output.clone()
    args[field][1] = value
    with pytest.raises(ValueError):
        runtime.batch_get(*args)
    assert runtime.torch.equal(output, before) and runtime.calls == 0


def test_zero_task_and_capture_are_not_correctness_or_npu_providers():
    runtime, _, output, args = make_fake(zero_task=True)
    before = output.clone()
    runtime.batch_get(*args)
    assert runtime.torch.equal(output, before)
    runtime.timing = TimingContext(capturing=True)
    with pytest.raises(RuntimeError, match="capture"):
        runtime.batch_get(*args)


def test_zero_count_ignores_descriptor_contents_and_preserves_every_output_row():
    runtime, _, output, args = make_fake()
    args[4].zero_()
    for field in (2, 3, 5):
        args[field].fill_(-999)
    before = output.clone()
    runtime.batch_get(*args)
    assert runtime.torch.equal(output, before) and runtime.calls == 1


def test_full_capacity_multiple_directories_and_last_block_row():
    runtime, source, output, args = make_fake()
    torch = runtime.torch
    second = source.bitwise_xor(0xFF)
    runtime.register_block(8192, second)
    args[0] = torch.tensor([4096, 8192], dtype=torch.int64)
    args[1] = torch.ones(2, dtype=torch.uint8)
    args[2] = torch.tensor([1, 0, 1], dtype=torch.int32)
    args[3] = torch.tensor([127 * 656, 0, 656], dtype=torch.int32)
    args[4].fill_(3)
    args[5] = torch.tensor([7, 0, 3], dtype=torch.int32)
    args[8] = True
    before = output.clone()
    runtime.batch_get(*args)
    assert torch.equal(output[7], second[127])
    assert torch.equal(output[0], source[0])
    assert torch.equal(output[3], second[1])
    assert torch.equal(output[[1, 2, 4, 5, 6]], before[[1, 2, 4, 5, 6]])


@pytest.mark.parametrize(
    "field,value",
    [
        (4, -1),
        (4, 4),
        (2, 1),
        (1, 0),
        (0, 8192),
        (3, -656),
        (3, 128 * 656),
        (5, -1),
    ],
)
def test_invalid_counts_directory_and_boundaries_never_partially_copy(field, value):
    runtime, _, output, args = make_fake()
    before = output.clone()
    args[field][-1 if field in (0, 1, 4) else 1] = value
    with pytest.raises(ValueError):
        runtime.batch_get(*args)
    assert runtime.torch.equal(output, before) and runtime.calls == 0


@pytest.mark.parametrize("field", range(7))
def test_wrong_tensor_dtype_fails_before_writing(field):
    runtime, _, output, args = make_fake()
    before = output.clone()
    args[field] = args[field].to(runtime.torch.float32)
    with pytest.raises(ValueError, match="dtype"):
        runtime.batch_get(*args)
    assert runtime.torch.equal(output, before) and runtime.calls == 0


@pytest.mark.parametrize("field", range(7))
def test_noncontiguous_tensor_fails_before_writing(field):
    runtime, _, output, args = make_fake()
    torch = runtime.torch
    original = args[field]
    # A singleton strided view is considered contiguous by Torch. Make all
    # tested vectors length >= 2 so this tests the actual storage invariant.
    shape = (max(2, original.numel()),) if original.ndim == 1 else original.shape
    storage = torch.zeros((*shape, 2), dtype=original.dtype)
    args[field] = storage[..., 0]
    assert not args[field].is_contiguous()
    before = output.clone()
    with pytest.raises(ValueError, match="contiguous"):
        runtime.batch_get(*args)
    assert torch.equal(output, before) and runtime.calls == 0


@pytest.mark.parametrize(
    "field,shape",
    [
        (0, (1, 1)),
        (1, (1, 1)),
        (2, (1, 3)),
        (3, (1, 3)),
        (4, (1, 1)),
        (5, (1, 3)),
        (6, (8 * 656,)),
    ],
)
def test_invalid_tensor_ranks_fail_before_writing(field, shape):
    runtime, _, output, args = make_fake()
    before = output.clone()
    args[field] = args[field].reshape(shape)
    with pytest.raises(ValueError):
        runtime.batch_get(*args)
    assert runtime.torch.equal(output, before) and runtime.calls == 0


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_synchronization_flag_requires_an_actual_bool(value):
    runtime, _, output, args = make_fake()
    before = output.clone()
    args[8] = value
    with pytest.raises(ValueError, match="must be bool"):
        runtime.batch_get(*args)
    assert runtime.torch.equal(output, before) and runtime.calls == 0


def test_short_descriptor_array_limits_count_even_when_other_arrays_have_room():
    runtime, _, output, args = make_fake()
    args[3] = args[3][:1]
    before = output.clone()
    with pytest.raises(ValueError, match="capacity"):
        runtime.batch_get(*args)
    assert runtime.torch.equal(output, before) and runtime.calls == 0


@pytest.mark.parametrize("field", range(7))
def test_non_cpu_tensors_are_rejected_without_device_readback(field):
    runtime, _, output, args = make_fake()
    args[field] = runtime.torch.empty_like(args[field], device="meta")
    before = output.clone()
    with pytest.raises(ValueError, match="CPU"):
        runtime.batch_get(*args)
    assert runtime.torch.equal(output, before) and runtime.calls == 0


@pytest.mark.parametrize("invalid", ["directory_size", "count_size", "output_width"])
def test_invalid_array_sizes_fail_before_writing(invalid):
    runtime, _, output, args = make_fake()
    if invalid == "directory_size":
        args[1] = args[1].repeat(2)
    elif invalid == "count_size":
        args[4] = args[4].repeat(2)
    else:
        args[6] = output.reshape(4, 1312)
    before = output.clone()
    with pytest.raises(ValueError):
        runtime.batch_get(*args)
    assert runtime.torch.equal(output, before) and runtime.calls == 0


def test_only_explicitly_registered_complete_cpu_blocks_can_be_sources():
    runtime, source, _, _ = make_fake()
    for address, block in [(0, source), (4096, source), (8192, source[:-1])]:
        with pytest.raises(ValueError):
            runtime.register_block(address, block)
    assert set(runtime.blocks) == {4096}
