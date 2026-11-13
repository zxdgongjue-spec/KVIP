# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from kvip.diagnostics import topk_probe


def fixture():
    raw = np.zeros(128, dtype=np.int64)
    words = raw[64:128].view(np.uint64).reshape(4, 16)[:, 12:16]
    words[:, 0] = topk_probe.MAGIC | (6 << 48) | (1 << 56)
    words[:, 1] = 130048 | (2048 << 32)
    words[:, 2] = 8192 | (8 << 32)
    words[:, 3] = 130048 | (12345 << 32) | (8 << 48)
    return raw, words


def test_real_counter_protocol_does_not_treat_counts_as_cycles(capsys):
    raw, words = fixture()
    words[1] = [topk_probe.MAGIC | (4 << 32) | (2 << 48), 0, 0, 130048]
    result = topk_probe.decode(raw, 1)
    assert result['activated_queries'] == 3
    assert result['statuses'] == {'success': 3, 'guard_fallback': 1}
    assert result['guard_reasons'] == {'topk_only': 1}
    assert result['totals'] == dict(visited=390144, survivors=6144, radix_inputs=24576, radix_calls=24)
    assert result['included_in_timing'] is False
    topk_probe.print_summary(result)
    output = capsys.readouterr().out
    assert 'TOPK_PROBE activated=3/4' in output
    assert 'timing=excluded' in output and '_us=' not in output


@pytest.mark.parametrize('mutation', ['absent', 'unfinished', 'excess', 'unknown', 'partial_success'])
def test_rejects_missing_or_inconsistent_diagnostic(mutation):
    raw, words = fixture()
    if mutation == 'absent':
        words[0, 0] = 0
    elif mutation == 'unfinished':
        words[0, 0] = topk_probe.MAGIC | (8 << 48)
    elif mutation == 'excess':
        words[0, 1] = 130048 | (130049 << 32)
    elif mutation == 'unknown':
        words[0, 0] = int(words[0, 0]) | (1 << 57)
    else:
        words[0, 3] = int(words[0, 3]) + 1
    with pytest.raises(ValueError):
        topk_probe.decode(raw, 1)


def test_independent_probe_has_no_event_timing_and_runs_oracle(monkeypatch):
    from types import SimpleNamespace
    from kvip.diagnostics import replay_measure
    state = dict(profile=False, replays=0, verifies=0, graph_closed=False)

    class Forward:
        batch = 1
        times = SimpleNamespace(zero_=lambda: None)

        def __call__(self):
            pass

        def set_options(self, diagnostics=False):
            state['profile'] = diagnostics

    class Graph:
        def capture_begin(self, stream):
            pass

        def capture_end(self):
            pass

        def replay(self, stream):
            state['replays'] += 1

        def reset(self):
            state['graph_closed'] = True

    def verify():
        state['verifies'] += 1
        return {}

    def event(**kwargs):
        raise AssertionError('Probe must never create timing events')

    monkeypatch.setattr(replay_measure, 'cpu_array', lambda *args: fixture()[0])
    result = replay_measure.measure(
        SimpleNamespace(npu=SimpleNamespace(Event=event)),
        SimpleNamespace(AclGraph=Graph), SimpleNamespace(synchronize=lambda: None),
        SimpleNamespace(reset=lambda: None, verify=verify, forward=Forward(), lease=None),
        SimpleNamespace(mode='fused', probe_only=True), {},
    )
    assert list(result) == ['topk_probe']
    assert result['topk_probe']['activated_queries'] == 4
    assert state == dict(profile=False, replays=2, verifies=3, graph_closed=True)
