import numpy as np

from streaming_fetch_diagnostics import summarize_mismatches


def test_reports_actual_payload_and_both_writers():
    payload = np.arange(64, dtype=np.uint8).reshape(8, 8)
    output = payload.copy()
    output[2] = payload[5]
    selected = np.full((4, 2048), 1, dtype=np.int32)
    ids = selected.copy()
    slots = selected.copy()
    selected[0, 3] = ids[0, 3] = slots[0, 3] = 2
    destinations = np.array([2], dtype=np.int32)
    addresses = np.array([123], dtype=np.int64)
    posts = np.zeros((4, 32), dtype=np.int32)
    posts[0, 0] = 1
    lru = np.array([2, 4], dtype=np.int32)
    mapping = np.full(8, -1, dtype=np.int32)
    mapping[5] = 8
    report = summarize_mismatches(0, selected, ids, slots, output, payload,
                                  destinations, addresses, posts, lru, mapping, 8, 8, np.array([0, 123, 0, 0, 0, 0, 0, 0], dtype=np.uint64))
    assert report['mismatch_positions'] == 1
    item = report['examples'][0]
    assert item['actual_matching_tokens'] == [5]
    assert item['prefetch_writers'] == [5]
    assert item['final_writers'] == [{'q': 1, 'descriptor': 0, 'remote_address': '0x7b', 'source_tokens': [1]}]
