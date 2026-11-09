"""Host model of the common final scheduler and UBmem descriptor boundary."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_ubmem_final_uses_published_compact_descriptors(tmp_path):
    common = ROOT / "csrc/operators/common/lifetch/final_phase_loop.h"
    phase = ROOT / "csrc/operators/lightning_indexer_with_fetch_ubmem/op_kernel/resolver/final_fetch_phase.h"
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
