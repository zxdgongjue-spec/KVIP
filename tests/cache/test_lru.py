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
