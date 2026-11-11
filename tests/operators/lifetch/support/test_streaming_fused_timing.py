"""Check diagnostic timestamp layout and separate queue counters."""
import numpy as np

from streaming_forward import StreamingForward


class HostTensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_fused_fetch_timing_decodes_per_query_and_per_request():
    pipeline = StreamingForward.__new__(StreamingForward)
    pipeline.batch = 1
    pipeline.fused_final = True
    pipeline.q_workers = None
    words = np.zeros(224, dtype=np.uint64)
    p, u, li, f = words[:32], words[32:64], words[64:128].reshape(4, 16), words[128:]
    p[1:4] = [10, 20, 30]
    p[8] = 100
    for q in range(4):
        p[10+q*5:15+q*5] = np.arange(10, 15) + q*10
        li[q, :8] = np.arange(1, 9)
        li[q, 10:12] = [6, 7]
        f[q*10:q*10+10] = np.arange(100, 110) + q*20
        f[64+q*6:70+q*6] = [1, 2, 3, 4, 5, 6]
    u[10:25] = np.arange(20, 35)
    f[48:55] = [1, 2, 3, 4, 5, 6, 7]
    flags = np.zeros((1, 7, 32), dtype=np.int32)
    flags[0, :5, 1] = [100, 1, 2, 3, 4]
    pipeline.times, pipeline.flags = HostTensor(words), HostTensor(flags)
    data = pipeline.diagnostic_cycles()[0]
    assert data['prefetch_changes'] == 100 and data['q_new_misses'] == [1, 2, 3, 4]
    assert data['fetch']['queries'][1]['safe_post'] == 1
    assert data['fetch']['queries'][1]['safe_cq_wait'] == 2
    assert data['fetch']['queries'][1]['conflict_groups'] == 6
    assert data['fetch']['write_status'] == 1


def test_query_worker_timing_and_assignment_are_distinct():
    pipeline = StreamingForward.__new__(StreamingForward)
    pipeline.batch = 1
    pipeline.fused_final = True
    pipeline.q_workers = 2
    words = np.zeros(256, dtype=np.uint64)
    p, u, li, f = words[:32], words[32:64], words[64:128].reshape(4, 16), words[128:]
    p[1:4] = [10, 20, 30]
    p[8] = 100
    for q in range(4):
        p[10+q*5:15+q*5] = np.arange(10, 15) + q*10
        li[q, :8] = np.arange(1, 9)
        li[q, 10:12] = [6, 7]
        f[q*24:q*24+10] = np.arange(100, 110) + q*20
        f[q*24+12:q*24+18] = [1, 2, 3, 4, 5, 6]
        f[q*24+20] = 1
    u[10:25] = np.arange(20, 35)
    f[96:100] = [50, 60, 70, 80]
    flags = np.zeros((1, 12, 32), dtype=np.int32)
    flags[0, :5, 1] = [100, 1, 2, 3, 4]
    pipeline.times, pipeline.flags = HostTensor(words), HostTensor(flags)
    data = pipeline.diagnostic_cycles()[0]
    assert data['q_new_misses'] == [1, 2, 3, 4]
    assert data['fetch']['queries'][2]['safe_mode'] == 1
    assert data['fetch']['queries'][2]['safe_cq_wait'] == 2
    assert data['fetch']['queries'][2]['conflict_groups'] == 6
    assert data['fetch']['finalize'] == {'wait': 10, 'residency': 10, 'status': 10}
