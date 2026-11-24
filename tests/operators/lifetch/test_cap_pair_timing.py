# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
import json
from types import SimpleNamespace
from unittest.mock import Mock

import cap_pair_timing
import performance
from reporting import CorrectnessRecorder


def test_one_state_is_restored_before_each_graph_event(tmp_path, monkeypatch):
    state = Mock()
    state.lease.capture_scope.side_effect = nullcontext
    device = Mock()
    device.stream.return_value = device.graph.return_value = nullcontext()
    native_graph = device.NPUGraph.return_value
    profile_calls = []
    trace = []
    state.reset.side_effect = lambda: trace.append("reset")
    state.validate.side_effect = lambda _: trace.append("check")
    state.forward.side_effect = lambda: trace.append("capture")
    native_graph.replay.side_effect = lambda: trace.append("replay")
    state.forward.set_diagnostics.side_effect = lambda on: trace.append("diag-on" if on else "diag-off")

    def profile(_, graph, *, calls, folder, stream):
        assert calls == 1 and trace[-1] == "reset"
        trace.append("event-start")
        graph.replay()
        graph.completed()
        trace.append("event-end")
        profile_calls.append(folder.name)
        return {"samples": [{"us": 45.0}], "accounting": {"event_us": 47.0 + len(profile_calls)}}

    monkeypatch.setattr(performance, "profile_graph_batch", profile)
    ex = SimpleNamespace(CorrectnessRecorder=CorrectnessRecorder)
    config = SimpleNamespace(mode="diagnose", iterations=20, warmup=4)
    inputs = SimpleNamespace(source_round=1)

    class Sequence:
        rounds = [[1]]

        def __iter__(self):
            yield inputs

    cap_pair_timing.install(ex, config)
    assert config.iterations == 1
    result = ex.measure_replay_batch(SimpleNamespace(npu=device), [state], Sequence(), config, tmp_path, fused=True)
    assert result["correctness"] == "PASS"
    assert profile_calls == [f"call-{i:02d}" for i in range(20)]
    assert trace == (["diag-off", "reset", "capture"] + ["reset", "replay", "check"] * 4 +
                     ["reset", "event-start", "replay", "event-end", "check"] * 20 +
                     ["diag-on"] + ["reset", "replay", "check"] * 20 + ["diag-off"])
    archived = json.loads((tmp_path / "diagnose/round-1/full_attribution.json").read_text())
    assert [r["accounting"]["invocation"] for r in archived["samples"]] == list(range(20))
    assert result["all_rounds"]["mean_us"] == result["all_rounds"]["p50_us"] == 57.5
    native_graph.reset.assert_called_once()
