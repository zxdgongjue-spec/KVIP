"""Check the profiled-only streaming timing layout without a device."""
import numpy as np
from streaming_forward import StreamingForward


class FakeTensor:
    def __init__(self, value):
        self.value = value
    def cpu(self):
        return self
    def numpy(self):
        return self.value


def test_decode_two_requests_and_all_query_stages():
    batch = 2
    data = np.zeros(batch * 128, dtype=np.uint64)
    rows = data[:batch * 64].reshape(batch, 64)
    li = data[batch * 64:].reshape(batch, 4, 16)
    for b in range(batch):
        p, u = rows[b, :32], rows[b, 32:]
        p[1:4] = [10, 30, 35]
        p[8] = 200
        for q in range(4):
            p[10 + q * 5:15 + q * 5] = np.arange(10, 15) + q * 10
            li[b, q, :8] = np.arange(100, 108) + q * 10
            li[b, q, 8:10] = [1, 1]
        u[10], u[24] = 10, 100
        for phase in range(5):
            u[12 + phase * 2:14 + phase * 2] = [20 + phase * 10, 25 + phase * 10]
        u[22:24] = [70, 78]
    instance = StreamingForward.__new__(StreamingForward)
    instance.batch = batch
    instance.times = FakeTensor(data)
    decoded = instance.diagnostic_cycles()
    assert len(decoded) == 2
    for row in decoded:
        assert row['input_and_bitmap'] == 20 and row['initialize'] == 5
        assert row['metadata'] == 8 and row['update'] == [5] * 5
        assert row['update_wait'] == [10, 5, 5, 5, 5]
        assert len(row['li']) == len(row['producer']) == 4
        assert all(q['overlap'] and q['topk_only'] for q in row['li'])
        assert all(q['resolve'] == q['output'] == q['pack'] == q['publish'] == 1 for q in row['producer'])
