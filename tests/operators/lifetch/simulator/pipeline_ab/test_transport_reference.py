import pytest
from transport_reference import Descriptor, publish


def test_stable_partition_and_worker_ranges():
    reserved = [0, 1, 2, -1, 4, -1, 6, -1]
    records = [Descriptor(100 + i, slot, 1000 + i) for i, slot in enumerate([3, 1, 4, 0, 5, 2, 7])]
    packed, posts = publish(records, reserved, used=3, batch=2, owner=4, output_stride=8, workers=3)
    assert [(p.descriptor, p.count, p.nonconflicting) for p in posts] == [(16384, 2, 1), (16386, 2, 1), (16388, 3, 2)]
    assert [x[0] for x in packed] == [100, 101, 102, 103, 104, 106, 105]
    assert all(dest == 32 + next(x.slot for x in records if x.token == tok) for tok, _, dest in packed)
    assert [addr for _, addr, _ in packed] == [1000, 1001, 1002, 1003, 1004, 1006, 1005]


def test_no_prefetch_and_empty_workers():
    packed, posts = publish([Descriptor(10, 1, 123)], [-1, 0], used=0, batch=0, owner=0, output_stride=2, workers=4)
    assert packed == [(10, 123, 1)]
    assert [(p.count, p.nonconflicting, p.deferred) for p in posts] == [(0, 0, False)] * 3 + [(1, 1, True)]


def test_duplicate_descriptors_rejected():
    for records in ([Descriptor(10, 0, 1), Descriptor(10, 1, 2)],
                    [Descriptor(10, 0, 1), Descriptor(11, 0, 2)]):
        with pytest.raises(ValueError):
            publish(records, [-1, -1], 0, 0, 0, 2, 1)


def test_all_conflicts_wait_until_prefetch_completes():
    packed, posts = publish([Descriptor(9, 0, 1), Descriptor(10, 1, 2)], [0, 1], 2, 1, 2, 2, 1)
    assert posts[0].descriptor == 8192
    assert posts[0].count == 2 and posts[0].nonconflicting == 0
    assert [x[2] for x in packed] == [4, 5]
