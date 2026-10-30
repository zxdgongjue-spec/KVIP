"""Independent layout/byte checks; never used by the production launch path."""

import numpy as np


class ColdOrder:
    def __init__(self):
        self.hot = []
        self.cold = []
        self.length = 0

    def reconcile(self, hot, length):
        desired = sorted(hot, key=hot.get)
        members = set(desired)

        def keep_positions(old, wanted):
            target = set(wanted)
            result = [t if t in target else None for t in old[: len(wanted)]]
            result.extend([None] * (len(wanted) - len(result)))
            retained = set(result) - {None}
            incoming = iter(t for t in wanted if t not in retained)
            return [next(incoming) if t is None else t for t in result]

        self.hot = keep_positions(self.hot, desired)
        self.cold = keep_positions(self.cold, [t for t in range(length) if t not in members])
        self.length = length
        assert len(set(self.cold)) == len(self.cold)
        assert set(self.cold).isdisjoint(members)
        assert set(self.cold) | members == set(range(length))
        return np.asarray(self.cold, np.int32)


def inspect_cache(raw, g):
    raw = np.asarray(raw, np.uint8)
    c, s = g.candidate_capacity, g.resident_capacity
    header = raw[:128].view(np.int32)
    rows = s if header[18] == 1 else s + c
    keys = raw[128 : 128 + rows * 128].reshape(rows, 128)
    scales = raw[128 + rows * 128 : 128 + rows * 132].view(np.uint32)
    begin = 128 + rows * 132
    ids = raw[begin : begin + (s + c) * 4].view(np.int32)
    begin += (s + c) * 8
    positions = raw[begin : begin + c * 4].view(np.int32)
    return header, keys, scales, ids, positions


def verify_cache(raw, g, arrays, b, cold=None, hot_order=None, backing=None):
    from contract_reference import frozen_hot

    header, keys, scales, ids, positions = inspect_cache(raw, g)
    pool, length, budget = (int(arrays[n][b]) for n in ("pool_rows", "candidate_lens", "budgets"))
    hot = frozen_hot(arrays["forward"][pool], arrays["reverse"][pool], arrays["victims"][b], length, budget)
    hotids = np.asarray(sorted(hot, key=hot.get), np.int32)
    nh, nc = header[2:4]
    assert header[0] == 1 and header[1] == length
    assert tuple(header[4:7]) == (pool, g.layer_id, budget)
    assert nh == len(hotids) and nc == length - nh
    assert set(ids[:nh]) == set(hotids)
    if hot_order is not None:
        assert np.array_equal(ids[:nh], hot_order)
    hotids = ids[:nh]
    coldids = ids[g.resident_capacity : g.resident_capacity + nc]
    assert set(coldids) == set(range(length)) - set(hot)
    if cold is not None:
        assert np.array_equal(coldids, cold)
    for offset, n, bankids in ((0, nh, hotids), (g.resident_capacity, nc, coldids)):
        pages = arrays["index_table"][b, bankids // 128]
        expected = arrays["key"][pages, bankids % 128, 0, :]
        expected_scale = arrays["key_scale"][pages, bankids % 128, 0].view(np.uint32)
        actual_key, actual_scale = keys[offset : offset + n], scales[offset : offset + n]
        if offset and header[18] == 1:
            assert backing is not None, "owned cache verification requires the actual paged cold payload"
            physical = arrays["index_table"][b, np.arange(n) // 128] * 128 + np.arange(n) % 128
            actual_key = backing["key"].reshape(-1, 128)[physical]
            actual_scale = backing["key_scale"].reshape(-1).view(np.uint32)[physical]
        assert np.array_equal(actual_key, expected), "cached FP8 key mismatch"
        assert np.array_equal(actual_scale, expected_scale), "cached key scale mismatch"
        expected_pos = -np.arange(n) - 2 if offset == 0 else np.arange(n)
        assert np.array_equal(positions[bankids], expected_pos), "token position mismatch"


def cache_metrics(raw, g):
    """Read tail copy jobs after successful CQ; init copies are not included."""
    header, _, _, _, _ = inspect_cache(raw, g)
    rows = g.resident_capacity if header[18] == 1 else g.resident_capacity + g.candidate_capacity
    start = 128 + rows * 132 + (g.resident_capacity + g.candidate_capacity) * 8 + g.candidate_capacity * 4
    jobs = np.asarray(raw, np.uint8)[start:].view(np.int32)[: header[7]]
    hot = jobs < g.resident_capacity
    copied = (
        int(header[7])
        if header[17] in (3, 4)
        else int(np.count_nonzero((hot & (jobs < header[2])) | (~hot & (jobs - g.resident_capacity < header[3]))))
    )
    return {
        "persistent_bytes": len(raw),
        "storage": "owned_hot_and_paged_cold" if header[18] == 1 else "derived_contiguous_banks",
        "tail_key_refresh_rows": copied,
        "tail_key_and_scale_write_bytes": copied * 132,
        "scope": "tail maintenance only; excludes init and metadata traffic",
    }
