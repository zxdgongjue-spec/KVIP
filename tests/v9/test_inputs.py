"""CPU Torch checks for the public 32/64-head preparation boundary."""

import pytest

from kvip.v9 import pack_query_heads


def inputs(heads, tokens=4):
    torch = pytest.importorskip("torch")
    query = (
        torch.arange(tokens * heads * 128).reshape(tokens, heads, 128) % 15 - 7
    ).to(torch.float8_e4m3fn)
    weights = (torch.arange(tokens * heads).reshape(tokens, heads) % 7).to(
        torch.bfloat16
    )
    scales = torch.full((tokens, heads), 0.5)
    return torch, query, weights, scales


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("flattened", [False, True])
def test_pack_preserves_original_heads_and_neutral_padding(heads, flattened):
    torch, query, weights, scales = inputs(heads)
    saved = [value.clone() for value in (query, weights, scales)]
    supplied_query = query.reshape(-1, 128) if flattened else query
    packed = pack_query_heads(torch, supplied_query, weights, scales, 4)
    assert [tuple(x.shape) for x in packed] == [(4, 64, 128), (4, 64), (4, 64)]
    assert [x.dtype for x in packed] == [
        torch.float8_e4m3fn,
        torch.bfloat16,
        torch.float32,
    ]
    assert all(x.is_contiguous() for x in packed)
    assert torch.equal(
        packed[0][:, :heads].view(torch.uint8), saved[0].view(torch.uint8)
    )
    assert torch.equal(packed[1][:, :heads], saved[1])
    assert torch.equal(packed[2][:, :heads], saved[2])
    if heads == 32:
        assert torch.count_nonzero(packed[0][:, heads:].view(torch.uint8)) == 0
        assert torch.count_nonzero(packed[1][:, heads:]) == 0
        assert torch.equal(packed[2][:, heads:], torch.ones((4, 32)))
    else:
        assert [x.data_ptr() for x in packed] == [
            x.data_ptr() for x in (query, weights, scales)
        ]
    assert torch.equal(query.view(torch.uint8), saved[0].view(torch.uint8))
    assert torch.equal(weights, saved[1]) and torch.equal(scales, saved[2])


def test_padding_preserves_the_weighted_relu_score():
    torch, query, weights, scales = inputs(32)
    key = (torch.arange(128) % 3 - 1).float()
    packed_query, packed_weights, packed_scales = pack_query_heads(
        torch, query, weights, scales, 4
    )
    expected = (torch.relu(query.float() @ key) * weights.float() * scales).sum(1)
    actual = (
        torch.relu(packed_query.float() @ key) * packed_weights.float() * packed_scales
    ).sum(1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_noncontiguous_inputs_produce_contiguous_preserved_heads():
    torch, query, weights, scales = inputs(32)
    query = query.transpose(1, 2).contiguous().transpose(1, 2)
    weights = weights.t().contiguous().t()
    scales = scales.t().contiguous().t()
    assert not query.is_contiguous() and not weights.is_contiguous()
    packed = pack_query_heads(torch, query, weights, scales, 4)
    assert all(value.is_contiguous() for value in packed)
    assert torch.equal(packed[0][:, :32].float(), query.float())
    assert torch.equal(packed[1][:, :32], weights)
    assert torch.equal(packed[2][:, :32], scales)


@pytest.mark.parametrize(
    "invalid", ["heads", "tokens", "dimension", "rank", "dtype", "weights", "scales"]
)
def test_invalid_inputs_are_rejected(invalid):
    torch, query, weights, scales = inputs(32)
    if invalid == "heads":
        query, weights, scales = query[:, :16], weights[:, :16], scales[:, :16]
    elif invalid == "tokens":
        query = query[:3]
    elif invalid == "dimension":
        query = query[:, :, :127]
    elif invalid == "rank":
        query = query.unsqueeze(0)
    elif invalid == "dtype":
        query = query.float()
    elif invalid == "weights":
        weights = weights[:, :-1]
    elif invalid == "scales":
        scales = scales[:, :-1]
    with pytest.raises(ValueError):
        pack_query_heads(torch, query, weights, scales, 4)
