import inspect

import pytest
from kvop.testing.transport import FakeNpuRuntime, TimingContext


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
