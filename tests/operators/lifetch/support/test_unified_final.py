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


def test_full_trace_accounts_all_li_kernels_and_combined_fetch_update():
    from device_trace import account

    events = [dict(name="replay", ph="X", ts="1000", dur=100)]
    kernels = [
        ("v9_init_kernel_0", 1, 2),
        ("v9_li_kernel_0", 4, 10),
        ("v9_li_kernel_1_0", 14, 2),
        ("v9_li_kernel_2", 16, 8),
        ("v9_li_kernel_3", 24, 2),
        ("v9_resolve_kernel_0", 26, 4),
        ("kvop_direct_update_outputs_early_kernel_0", 30, 3),
        ("kvop_direct_apply_owned_kernel_0", 33, 5),
        ("v9_fetch_update_kernel_0", 38, 7),
    ]
    events.extend(dict(name=name, ph="X", cat="kernel", ts=str(1000 + start), dur=duration)
                  for name, start, duration in kernels)
    result = account(events, "replay", 50, full=True)
    assert result["stages_us"] == dict(init=2, li=22, resolve=4, publish=3, apply=5, fetch=7)
    assert result["inter_kernel_gap_us"] == 1
    assert result["boundary_residual_us"] == 6
    assert sum(result["stages_us"].values()) + result["inter_kernel_gap_us"] + result["boundary_residual_us"] == 50
    missing = account(events[:-1], "replay", 50, full=True)
    assert "error" in missing and missing["found"]["fetch"] == 0


def test_trace_accounts_early_fetch_before_update():
    from device_trace import account

    events = [dict(name="replay", ph="X", ts="0", dur=100)]
    for name, start, duration in (
        ("v9_init_kernel_0", 0, 2), ("v9_li_kernel_0", 2, 10),
        ("v9_resolve_kernel_0", 12, 4), ("v9_early_fetch_kernel_0", 16, 5),
        ("kvop_direct_update_outputs_early_kernel_0", 21, 3),
        ("kvop_direct_apply_owned_kernel_0", 24, 5), ("v9_fetch_update_kernel_0", 29, 2),
    ):
        events.append(dict(name=name, ph="X", cat="kernel", ts=str(start), dur=duration))
    result = account(events, "replay", 31, full=True)
    assert result["stages_us"]["early_fetch"] == 5
    assert result["stages_us"]["fetch"] == 2
    assert sum(result["stages_us"].values()) + result["inter_kernel_gap_us"] + result["boundary_residual_us"] == 31


def test_split_fetch_timing_decodes_64_bit_counters():
    import numpy as np
    from device_trace import decode_final_detail

    owners = np.zeros((1, 16), dtype=np.uint64)
    owners[0, 4:12] = [100, 130, 200, 300, 310, 400, 460, 500]
    owners[0, 12:15] = [11, 7, 3]
    statuses = np.zeros((1, 32), dtype=np.uint32)
    statuses[0, 8:16].view(np.uint64)[:] = [2**32 + 1, 7, 11, 13]
    actual = decode_final_detail(owners, statuses)
    assert actual == [dict(ticks=[100, 130, 200, 300, 310, 400, 460, 500],
                           safe=11, conflict=7, pending=3, kvaf_ticks=[2**32 + 1, 7, 11, 13])]


def test_worker_comparison_summary_checks_all_geometries(tmp_path):
    import importlib.util
    import json

    source = Path(__file__).resolve().parents[1] / 'compare_resolver_workers.py'
    spec = importlib.util.spec_from_file_location('compare_resolver_workers', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for workers in (1, 2, 4):
        for batch in (1, 2, 4, 8, 16):
            directory = tmp_path / f'w{workers}' / f'case-{batch}'
            directory.mkdir(parents=True)
            report = {
                'status': 'pass', 'final_workers': workers,
                'resource_geometry': {'logical_aic_tasks': 2 * batch, 'cores_per_batch': 2,
                                      'intermediate_fetches': min(8, 2 * batch)},
                'full_trace': {'samples': [{'accounting': {'stages_us': {'resolve': value}}}
                                           for value in (10, 12, 11)]},
                'timing': {'prototype_with_update': {'p50_us': 400, 'p99_us': 450}},
                'split_samples': {'prototype_with_update': [{'final_reads': 5}, {'final_reads': 7}]},
                'sfa': {'max_abs': 0.001},
            }
            (directory / 'report.json').write_text(json.dumps(report))
    summary = module.summarize(tmp_path)
    assert '1 | 1 | 2 | 11.000 | 400.000 | 450.000 | 5..7' in summary
    assert '16 | 4 | 8 | 11.000 | 400.000 | 450.000 | 5..7' in summary
    assert len([line for line in summary.splitlines() if ' | ' in line]) == 16
