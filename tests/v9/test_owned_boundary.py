"""Real installed ownership API at prefill/decode/native boundaries, on CPU."""

import pytest
from contract_fixture import make_fixture, to_device
from kvop.v9 import OwnedIndexCache


def test_owned_handoff_growth_rewrite_and_native_restore():
    torch = pytest.importorskip("torch")
    g, arrays, _ = make_fixture(candidate=16385)
    t = to_device(torch, arrays, g, torch.device("cpu"))
    length = 16380
    table = t["index_table"][0]
    tokens = torch.arange(16385)
    physical = table[tokens // 128].long() * 128 + tokens % 128
    original_key = t["key"].view(torch.uint8).reshape(-1, 128)[physical].clone()
    original_scale = t["key_scale"].reshape(-1)[physical].clone()
    cache = OwnedIndexCache(
        torch,
        g,
        t["key"],
        t["key_scale"],
        table,
        pool=int(arrays["pool_rows"][0]),
        length=length,
        budget=12288,
        forward=t["forward"],
        reverse=t["reverse"],
    )
    _, keys, scales = cache.logical()
    assert torch.equal(keys, original_key[:length])
    assert torch.equal(scales, original_scale[:length])
    cache.grow(16385)
    _, keys, scales = cache.logical()
    assert torch.equal(keys, original_key)
    assert torch.equal(scales, original_scale)
    hot = int(cache.ids[0])
    cold = int(cache.ids[g.resident_capacity])
    changed = torch.tensor([hot, cold])
    updated_keys = torch.full((2, 128), 42, dtype=torch.uint8)
    updated_scales = torch.tensor([2.0, 3.0])
    cache.write_logical(changed, updated_keys, updated_scales)
    original_key[changed] = updated_keys
    original_scale[changed] = updated_scales
    cache.restore_logical()
    assert torch.equal(
        t["key"].view(torch.uint8).reshape(-1, 128)[physical], original_key
    )
    assert torch.equal(t["key_scale"].reshape(-1)[physical], original_scale)


def test_page_replacement_preserves_cold_hot_and_tail_after_checked_completion():
    torch = pytest.importorskip("torch")
    g, arrays, _ = make_fixture(candidate=16385)
    t = to_device(torch, arrays, g, torch.device("cpu"))
    table = t["index_table"][0]
    length = 16385
    tail_end = min(length + 64, table.numel() * 128)
    tokens = torch.arange(tail_end)
    physical = table[tokens // 128].long() * 128 + tokens % 128
    keys = t["key"].view(torch.uint8).reshape(-1, 128)[physical].clone()
    scales = t["key_scale"].reshape(-1)[physical].clone()
    cache = OwnedIndexCache(
        torch,
        g,
        t["key"],
        t["key_scale"],
        table,
        pool=int(arrays["pool_rows"][0]),
        length=length,
        budget=12288,
        forward=t["forward"],
        reverse=t["reverse"],
    )
    replacement = table.roll(1)
    with pytest.raises(ValueError, match="completion"):
        cache.rebind_pages(replacement, completion_verified=False, exclusive_pages=True)
    cache.rebind_pages(
        replacement,
        completion_verified=True,
        exclusive_pages=True,
        logical_length=tail_end,
    )
    cache.restore_logical()
    destinations = replacement[tokens // 128].long() * 128 + tokens % 128
    assert torch.equal(t["key"].view(torch.uint8).reshape(-1, 128)[destinations], keys)
    assert torch.equal(t["key_scale"].reshape(-1)[destinations], scales)
