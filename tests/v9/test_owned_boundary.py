"""Real installed ownership API at prefill/decode/native boundaries, on CPU."""

import pytest
from contract_fixture import make_fixture, to_device

from kvip.v9 import OwnedIndexCache


def owned_fixture(length=16380):
    torch = pytest.importorskip("torch")
    g, arrays, _ = make_fixture(candidate=16385)
    t = to_device(torch, arrays, g, torch.device("cpu"))
    table = t["index_table"][0]
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
    return torch, cache, table


@pytest.mark.parametrize("growth", [0, 5])
def test_grow_cannot_replace_pages_or_change_payload_and_h0(growth):
    torch, cache, table = owned_fixture()
    _, keys, scales = cache.logical()
    before_key, before_scale = cache.key.clone(), cache.scale.clone()
    cache.header[20] = 123
    with pytest.raises(ValueError, match="rebind_pages"):
        cache.grow(cache.length + growth, table.roll(1))
    assert cache.header[20] == 123 and cache.length == 16380
    assert torch.equal(cache.table, table)
    assert torch.equal(cache.key, before_key) and torch.equal(cache.scale, before_scale)
    _, actual_keys, actual_scales = cache.logical()
    assert torch.equal(keys, actual_keys) and torch.equal(scales, actual_scales)


def test_host_metadata_allows_append_and_rejects_stale_or_invalid_pages():
    torch, cache, table = owned_fixture()
    cache.page_ids = tuple(table[:128].tolist())
    cache.table = table[:128].clone()
    cache.grow(16385, table, page_ids=tuple(table.tolist()))
    assert cache.length == 16385 and cache.page_ids == tuple(table.tolist())
    before = cache.key.clone()
    for pages in (
        tuple(table.roll(1).tolist()),
        (True,) * len(table),
        (0,) * len(table),
    ):
        with pytest.raises(ValueError):
            cache.grow(cache.length, table, page_ids=pages)
        assert torch.equal(cache.key, before)
    with pytest.raises(ValueError, match="requires a refreshed"):
        cache.grow(cache.length, page_ids=cache.page_ids)


def test_stable_page_metadata_reuses_the_private_table_without_a_copy():
    _, cache, table = owned_fixture()
    private = cache.table
    cache.grow(cache.length, table, page_ids=cache.page_ids)
    assert cache.table is private
    cache.grow(cache.length + 1, table, page_ids=cache.page_ids)
    assert cache.table is private


@pytest.mark.parametrize("invalid", ["duplicate", "negative", "overflow", "short"])
def test_invalid_rebind_leaves_all_owned_state_unchanged(invalid):
    torch, cache, table = owned_fixture()
    replacement = table.clone()
    if invalid == "duplicate":
        replacement[0] = replacement[1]
    elif invalid == "negative":
        replacement[0] = -1
    elif invalid == "overflow":
        replacement[0] = cache.g.index_blocks
    else:
        replacement = replacement[:1]
    before = tuple(
        x.clone() for x in (cache.key, cache.scale, cache.buffer, cache.table)
    )
    with pytest.raises(ValueError):
        cache.rebind_pages(replacement, completion_verified=True, exclusive_pages=True)
    assert all(
        torch.equal(a, b)
        for a, b in zip(before, (cache.key, cache.scale, cache.buffer, cache.table))
    )


def test_restored_owner_cannot_be_reused():
    _, cache, _ = owned_fixture()
    cache.restore_logical()
    for operation in (
        cache.logical,
        lambda: cache.grow(cache.length),
        cache.restore_logical,
    ):
        with pytest.raises(RuntimeError, match="retired"):
            operation()


@pytest.mark.parametrize("operation", ["rebind", "restore", "write"])
def test_partial_owned_write_failure_poisoned_owner_must_be_retained(operation):
    from types import SimpleNamespace

    torch, cache, table = owned_fixture()
    scale = cache.scale.reshape(-1)

    class FailingScale:
        def __getitem__(self, index):
            return scale[index]

        def index_copy_(self, *args):
            raise RuntimeError("injected copy failure")

    cache.scale = SimpleNamespace(reshape=lambda _: FailingScale())
    cold = int(cache.ids[cache.g.resident_capacity])
    operations = {
        "rebind": lambda: cache.rebind_pages(
            table.roll(1), completion_verified=True, exclusive_pages=True
        ),
        "restore": cache.restore_logical,
        "write": lambda: cache.write_logical(
            torch.tensor([cold]),
            torch.zeros((1, 128), dtype=torch.uint8),
            torch.ones(1),
        ),
    }
    with pytest.raises(RuntimeError, match="copy failure"):
        operations[operation]()
    assert cache.poisoned and cache.header[20] == 0
    with pytest.raises(RuntimeError, match="incomplete page transition"):
        cache.logical()


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


def cross_page_growth_fixture():
    """One hot row makes append compaction overlap the incoming canonical tail."""
    torch = pytest.importorskip("torch")
    g, arrays, _ = make_fixture(candidate=257, budget=2048, tail=4)
    tensors = to_device(torch, arrays, g, torch.device("cpu"))
    table = tensors["index_table"][0]
    tokens = torch.arange(257)
    physical = table[tokens // 128].long() * 128 + tokens % 128
    expected_keys = (tokens[:, None] % 251).to(torch.uint8).expand(-1, 128).contiguous()
    expected_scales = tokens.float() + 1
    tensors["key"].view(torch.uint8).reshape(-1, 128)[physical] = expected_keys
    tensors["key_scale"].reshape(-1)[physical] = expected_scales
    tensors["forward"].fill_(-1)
    tensors["reverse"].fill_(-1)
    pool = int(arrays["pool_rows"][0])
    tensors["forward"][pool, 0] = 0
    tensors["reverse"][pool, 0] = 0
    cache = OwnedIndexCache(
        torch,
        g,
        tensors["key"],
        tensors["key_scale"],
        table[:1],
        pool=pool,
        length=123,
        budget=2048,
        forward=tensors["forward"],
        reverse=tensors["reverse"],
        page_ids=table[:1].tolist(),
    )
    return torch, cache, table[:2], expected_keys, expected_scales


def test_cross_page_append_preserves_hot_cold_and_remaining_logical_tail():
    torch, cache, table, keys, scales = cross_page_growth_fixture()
    cache.header[20] = 123
    cache.grow(129, table, page_ids=table.tolist())
    _, actual_keys, actual_scales = cache.logical()
    assert torch.equal(actual_keys, keys[:129])
    assert torch.equal(actual_scales, scales[:129])
    assert cache.page_ids == tuple(table.tolist()) and cache.length == 129
    assert cache.header[20] == 0
    tail = torch.arange(129, 133)
    physical = table[tail // 128].long() * 128 + tail % 128
    assert torch.equal(
        cache.key.view(torch.uint8).reshape(-1, 128)[physical], keys[tail]
    )
    assert torch.equal(cache.scale.reshape(-1)[physical], scales[tail])


def test_partial_cross_page_append_failure_poisoned_owner_refuses_retry():
    torch, cache, table, keys, scales = cross_page_growth_fixture()
    original_scale = cache.scale
    flat_scale = original_scale.reshape(-1)
    cache.header[20] = 123

    class FailingScale:
        def reshape(self, _):
            return self

        def __getitem__(self, index):
            return flat_scale[index]

        def index_copy_(self, *args):
            raise RuntimeError("injected append scale failure")

    cache.scale = FailingScale()
    with pytest.raises(RuntimeError, match="append scale failure"):
        cache.grow(129, table, page_ids=table.tolist())
    cache.scale = original_scale
    assert cache.poisoned and cache.header[20] == 0
    # The first key copy overwrote canonical tail source rows. Retrying would
    # silently corrupt tokens 123..127; every subsequent owner operation must fail.
    for operation in (
        cache.logical,
        lambda: cache.grow(129, table, page_ids=table.tolist()),
        cache.restore_logical,
        lambda: cache.write_logical(torch.tensor([0]), keys[:1], scales[:1]),
    ):
        with pytest.raises(RuntimeError, match="incomplete page transition"):
            operation()
