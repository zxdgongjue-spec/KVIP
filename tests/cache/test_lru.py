import numpy as np
from kvip.cache.lru import SlotLRU


def test_candidates_and_feedback():
    lru = SlotLRU()
    lru.resize(16384, 16384)
    assert len(np.unique(lru.slots)) == 2048
    before = lru.slots.copy()
    lru.commit(np.array([0, 1, 2, 0], dtype=np.int32))
    assert not np.isin([0, 1, 2], lru.slots).any()
    assert len(np.unique(lru.slots)) == 2048
    assert not np.array_equal(before, lru.slots)
    lru.resize(16384, 100)
    assert (lru.slots[0, 100:] == -1).all()


def test_extent_order_matches_latest_slot_access():
    rng = np.random.default_rng(17)
    for extent in (4, 8, 16):
        lru = SlotLRU(policy="extent_r1", extent=extent)
        lru.resize(4096, 4096)
        for _ in range(12):
            feedback = rng.integers(0, 4100, 2048, dtype=np.int32)
            lru.commit(feedback)
            chain = []
            slot = lru.head[0]
            while slot >= 0:
                chain.append(slot)
                slot = lru.next[0, slot]
            rank = np.argsort(chain).reshape(-1, extent).max(axis=1)
            blocks = np.argsort(rank)[: 2048 // extent]
            expected = (blocks[:, None] * extent + np.arange(extent)).ravel()
            np.testing.assert_array_equal(lru.slots[0], expected)
            bits = np.unpackbits(lru.bitmap.view(np.uint8), bitorder="little")
            np.testing.assert_array_equal(np.flatnonzero(bits), np.sort(expected))
            assert lru.unique[0] == len(np.unique(feedback[feedback < 4096]))


def test_extent_resize_preserves_slot_fallback():
    extent = SlotLRU(policy="extent_r1", extent=8)
    baseline = SlotLRU(policy="slot_lru", extent=1)
    for committed in (21, 24, 65, 72, 9, 16, 0, 8):
        for lru in (extent, baseline):
            lru.resize(72, committed)
            lru.commit(np.array([0, 0, 1, 70], dtype=np.int32))
        assert np.all(extent.slots[0, committed:] == -1)
        np.testing.assert_array_equal(np.sort(extent.slots[0, :committed]), np.arange(committed))
        if committed % 8:
            np.testing.assert_array_equal(extent.slots, baseline.slots)
            np.testing.assert_array_equal(extent.bitmap, baseline.bitmap)


def test_extent_configuration_from_environment(monkeypatch):
    monkeypatch.setenv("KVIP_HOT_EVICTION_POLICY", "extent_r1")
    monkeypatch.setenv("KVIP_HOT_EVICTION_EXTENT", "4")
    assert SlotLRU().extent == 4

    monkeypatch.setenv("KVIP_HOT_EVICTION_POLICY", "slot_lru")
    assert SlotLRU().extent == 1
