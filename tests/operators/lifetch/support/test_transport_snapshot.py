"""Semantic transport comparison retained independently of the old launcher."""

def test_snapshot_compares_kv_and_reports_prefetch_differences(tmp_path):
    import numpy as np
    import pytest
    from diagnostics import save_transport_snapshot as save, compare_transport_snapshots as compare_snapshot

    ids = np.tile(np.arange(2048, dtype=np.int32), (1, 4, 1))
    slots = ids.copy()
    output = np.arange(4096 * 4, dtype=np.uint8).reshape(1, 4096, 4)
    mapping = np.full((1, 4096), -1, dtype=np.int32)
    mapping[0, 8] = 4096
    lru = np.arange(2048, dtype=np.int32)[None, :]
    rows = np.zeros((1, 256), dtype=np.uint32)
    rows[0, 16] = 1
    rows[0, 214] = 1
    addresses = np.arange(4096, dtype=np.int64)[None, :]
    a, b = tmp_path / 'a.npz', tmp_path / 'b.npz'
    save(a, ids, slots, output, mapping, lru, rows, addresses)
    mapping[0, 9] = 4097
    rows[0, 17] = 1
    rows[0, 215] = 2
    lru[0, 1] = 17
    save(b, ids, slots, output, mapping, lru, rows, addresses)
    result = compare_snapshot(a, b, 4096)
    assert 'logical_topk=identical selected_kv_bytes=identical' in result[0]
    assert 'lru_candidate_order_different_rows=1/1 differing_positions=1' in result
    assert any('ubmem_only=1' in line for line in result)
    output[0, 0, 0] ^= 1
    save(b, ids, slots, output, mapping, lru, rows, addresses)
    with pytest.raises(AssertionError, match='KV bytes differ'):
        compare_snapshot(a, b, 4096)
