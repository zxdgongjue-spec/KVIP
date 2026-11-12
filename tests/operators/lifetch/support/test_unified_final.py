"""Host model of the common final scheduler and UBmem descriptor boundary."""
import subprocess
from pathlib import Path

from device_trace import account_ubmem

ROOT = Path(__file__).resolve().parents[4]


def test_ubmem_trace_distinguishes_inline_and_separate_fetch():
    events = [{"name": "replay", "ph": "X", "ts": "1000.0", "dur": 100.0}]
    names = ("init", "li", "resolve", "fetch", "complete")
    events += [{"name": f"kvip_ubmem_656_{name}_kernel_0", "ph": "X", "ts": str(1001 + 10 * i),
                "dur": 5.0} for i, name in enumerate(names)]
    events += [{"name": f"kvip_ubmem_direct_{name}_kernel_0", "ph": "X", "ts": str(1052 + 8 * i),
                "dur": 4.0} for i, name in enumerate(("update_owned", "update_outputs", "apply_owned"))]
    result = account_ubmem(events, "replay", 90)
    assert result["fetch_separate"] and result["stages_us"]["resolve"] == 5
    assert result["stages_us"]["fetch"] == 5 and result["stages_us"]["update"] == 12
    old = account_ubmem([event for event in events if "_fetch_kernel" not in event["name"]], "replay", 90)
    assert not old["fetch_separate"] and old["stages_us"]["fetch"] == 0
    assert "error" in account_ubmem([event for event in events if "_resolve_kernel" not in event["name"]], "replay", 90)




def test_ubmem_trace_recognizes_fused_pipeline_without_phantom_fetch():
    events = [{"name": "replay", "ph": "X", "ts": 1000, "dur": 100}]
    for name, start, duration in (
        ("kvip_ubmem_656_init_kernel", 1001, 5),
        ("kvip_ubmem_656_li_kernel", 1007, 30),
        ("v9_streaming_resolve_update_656", 1038, 40),
    ):
        events.append({"name": name, "ph": "X", "cat": "kernel", "ts": start, "dur": duration})
    result = account_ubmem(events, "replay", 80)
    assert "error" not in result
    assert result["stages_us"]["resolve_update_fetch"] == 40
    assert result["stages_us"]["fetch"] == 0 and not result["fetch_separate"]
    assert result["kernel_span_us"] == 77
    assert "error" in account_ubmem(events[:-1], "replay", 80)


def test_ubmem_profile_finishes_schedule_and_keeps_incomplete_trace(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    import numpy as np
    from device_trace import collect_ubmem

    class Profile:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.steps = 0

        def __enter__(self):
            return self

        def step(self):
            self.steps += 1
            if self.steps == 2:
                self.kwargs['on_trace_ready'](self)

        def export_chrome_trace(self, path):
            Path(path).write_text(json.dumps({'traceEvents': []}))

        def __exit__(self, *args):
            assert self.steps == 2

    profiles = []

    def profile(**kwargs):
        value = Profile(**kwargs)
        profiles.append(value)
        return value

    def schedule(**kwargs):
        assert kwargs == dict(wait=0, warmup=0, active=2, repeat=1)
        return object()

    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(profiler=SimpleNamespace(
        profile=profile, schedule=schedule, ProfilerActivity=SimpleNamespace(CPU=0, NPU=1))))
    monkeypatch.setitem(sys.modules, 'torch.profiler', SimpleNamespace(record_function=lambda name: nullcontext()))
    event = lambda **kwargs: SimpleNamespace(record=lambda stream: None, synchronize=lambda: None,
                                             elapsed_time=lambda end: 0.4)
    stream = SimpleNamespace(synchronize=lambda: None)
    torch = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda device: stream, Event=event,
                                                synchronize=lambda: None))
    rotating = SimpleNamespace(device=0, rows=SimpleNamespace(cpu=lambda: SimpleNamespace(
        numpy=lambda: np.zeros((1, 256), dtype=np.uint32))), library=object())
    checked = []
    collect_ubmem(torch, rotating, SimpleNamespace(replay=lambda stream: None), lambda: None,
                  lambda: checked.append(True), 2, tmp_path)
    assert len(checked) == 2 and profiles[0].steps == 2
    saved = json.loads((tmp_path / 'attribution.json').read_text())
    assert all('error' in sample['accounting'] for sample in saved)
    assert capsys.readouterr().out.count('UBMEM_TRACE_ERROR') == 2


def test_ubmem_trace_counts_sequence_phase_and_dispatch_suffixes():
    events = [dict(name='replay', ph='X', ts=1000, dur=100)]
    for name, start, duration in (
        ('kvip_ubmem_656_init_kernel_0', 1001, 3),
        ('kvip_ubmem_656_li_kernel_0', 1005, 10),
        ('kvip_ubmem_656_li_kernel_1_0', 1016, 5),
        ('kvip_ubmem_656_li_kernel_2_0', 1022, 20),
        ('kvip_ubmem_656_li_kernel_3_0', 1043, 7),
        ('v9_streaming_resolve_update_656_0', 1051, 30),
    ):
        events.append(dict(name=name, ph='X', ts=start, dur=duration, cat='kernel'))
    result = account_ubmem(events, 'replay', 85)
    assert result['stages_us']['li'] == 42
    assert result['inter_kernel_gap_us'] == 5
