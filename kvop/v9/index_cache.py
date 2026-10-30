"""Owned device allocations; token layout is maintained exclusively by kernels."""


def cache_bytes(geometry, *, owned=False):
    # Header, packed keys/scales, IDs, dirty flags and token -> bank position.
    c, s = geometry.candidate_capacity, geometry.resident_capacity
    return 128 + (s if owned else c + s) * 132 + (c + s) * 12 + c * 4


def allocate_cache(torch, geometry, device, *, owned=False):
    buffer = torch.zeros(cache_bytes(geometry, owned=owned), dtype=torch.uint8, device=device)
    if owned:
        buffer[:128].view(torch.int32)[18] = 1
    return buffer


def bind_caches(torch, tensors, geometry, buffers):
    if len(buffers) != geometry.batch:
        raise ValueError("one persistent index cache per request/layer is required")
    if any(x.numel() not in (cache_bytes(geometry), cache_bytes(geometry, owned=True)) for x in buffers):
        raise ValueError("index cache capacity mismatch")
    device = tensors["query"].device
    if any(
        x.dtype != torch.uint8 or x.device != device or not x.is_contiguous() or x.data_ptr() % 128 for x in buffers
    ):
        raise ValueError("index caches must be contiguous aligned uint8 buffers on the query device")
    if len({x.data_ptr() for x in buffers}) != len(buffers):
        raise ValueError("requests cannot share a writable index cache")
    tensors["index_cache"] = torch.tensor(
        [x.data_ptr() for x in buffers], dtype=torch.int64, device=tensors["query"].device
    )
