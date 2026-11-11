"""Reference invariants for per-query descriptor publication and ordering."""
from transport_reference import Descriptor, publish


def staged(records_by_q, reserved, used):
    posted, chunks, joined = [], [], False
    for query, records in enumerate(records_by_q):
        packed, posts = publish(records, reserved, used, batch=0, owner=0,
                                output_stride=len(reserved), workers=1)
        post = posts[0]
        if post.count > 2048:
            raise ValueError("query exceeds its descriptor region")
        chunks.append((query * 2048, post.count, post.nonconflicting))
        safe, conflict = packed[:post.nonconflicting], packed[post.nonconflicting:]
        if safe:
            posted.append(('get', safe))
        if conflict and not joined:
            posted.append(('join_prefetch', []))
            joined = True
        if conflict:
            posted.append(('get', conflict))
    if not joined:
        posted.append(('join_prefetch', []))
    return chunks, posted


def test_query_submission_waits_only_before_first_conflict():
    reserved = [-1, 0, -1, 3, -1, 1, -1, -1]
    q = [[Descriptor(10, 0, 100), Descriptor(11, 1, 101)],
         [], [Descriptor(12, 2, 102), Descriptor(13, 5, 103)],
         [Descriptor(14, 4, 104)]]
    chunks, posts = staged(q, reserved, used=2)
    assert chunks == [(0, 2, 1), (2048, 0, 0), (4096, 2, 1), (6144, 1, 1)]
    assert [x[0] for x in posts] == ['get', 'join_prefetch', 'get', 'get', 'get', 'get']
    assert [row[0] for kind, group in posts if kind == 'get' for row in group] == [10, 11, 12, 13, 14]


def test_zero_miss_and_no_conflict_still_join_prefetch():
    chunks, posts = staged([[], [], [], []], [-1] * 8, used=0)
    assert chunks == [(q * 2048, 0, 0) for q in range(4)]
    assert posts == [('join_prefetch', [])]
    chunks, posts = staged([[Descriptor(1, 3, 12)], [], [], []], [-1] * 8, used=0)
    assert chunks[0] == (0, 1, 1)
    assert [x[0] for x in posts] == ['get', 'join_prefetch']
