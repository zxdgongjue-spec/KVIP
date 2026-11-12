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


def test_ubmem_final_uses_published_compact_descriptors(tmp_path):
    common = ROOT / "csrc/operators/common/lifetch/final_phase_loop.h"
    phase = ROOT / "csrc/operators/lightning_indexer_with_fetch_ubmem/op_kernel/final_fetch_phase.h"
    loop = common.read_text().replace("#include LIFETCH_FINAL_PHASE", phase.read_text())
    loop = loop.replace("LIFETCH_FINAL_JOIN", "(void)0")
    source = r'''
#include <cstdint>
#include <cassert>
#define __gm__
constexpr uint32_t kSlots=8192;
enum class TPosition { VECCALC };
struct Scratch { template<class T> T* Get() { return nullptr; } };
struct TPipe { template<class T> void InitBuffer(T&,int) {} void Destroy() {} };
template<TPosition> struct TBuf : Scratch {};
namespace ubmem_device { void CleanCacheLines(uint8_t*,uint64_t) {} }
struct Region { uint64_t peerBase=100, mappedBase=200, bytes=300, outputBase=400, outputBytes=500; };
struct Shape { uint32_t resident=16384,finalWorkers=4,tokenBytes=656; };
struct Context { uint64_t fallbackAddresses,destinations,rowOwners,outputStride; };
namespace kvaf_device {
struct UbmemBatchGetDeviceArgs { uint64_t* addresses; int32_t* destinations;
  uint64_t peer,mapped,bytes,output,out_bytes; uint32_t begin,count,owner,resident,worker,workers,width; };
struct UbmemBatchGetDeviceResult { uint32_t scheduled,completed,error; };
int calls=0;
UbmemBatchGetDeviceResult KvafBatchGetDevice(UbmemBatchGetDeviceArgs args,uint8_t*) {
  ++calls; assert(args.begin==8192 && args.count==37 && args.owner==2*16384);
  assert(args.addresses[args.begin]==12345 && args.destinations[args.begin]==17);
  assert(args.worker==3 && args.workers==4 && args.width==656);
  return {9,9,0};
}
}
int main() {
  uint64_t addresses[8200]={}; int32_t destinations[8200]={}; uint32_t owners[2]={0,2};
  addresses[8192]=12345; destinations[8192]=17;
  Context c{(uint64_t)addresses,(uint64_t)destinations,(uint64_t)owners,16384};
  auto ctx=&c; Region region_value; auto region=&region_value; Shape shape;
  uint32_t info[2]={0,37},b=1,w=3,error=0; kvaf_device::UbmemBatchGetDeviceResult result{};
  LOOP
  assert(kvaf_device::calls==1 && result.scheduled==9 && result.completed==9);
}
'''.replace("LOOP", loop)
    cpp = tmp_path / "final.cpp"
    cpp.write_text(source)
    exe = tmp_path / "final"
    subprocess.run(["c++", "-std=c++17", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


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
