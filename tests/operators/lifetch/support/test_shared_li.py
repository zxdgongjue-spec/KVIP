"""Host checks for the exact shared LI launch and shard schedule."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_shared_li_dispatch_and_shard_boundaries(tmp_path):
    common = ROOT / 'csrc/operators/common/lifetch'
    source = r'''
#include <cstdint>
#include <cassert>
#include <vector>
namespace rotating {
struct RuntimeShape {uint32_t splitQ=1,indexHeads=32,rounds=8,resident=16384,history=131072,liCores=8;};
#include "sequence_schedule.h"
}
std::vector<int> calls;
int failure=-1;
uint32_t run(int phase,uint32_t cores,void*,void*) {calls.push_back(phase);return phase==failure?99:0;}
uint32_t launch(void* bindings,void* stream,const rotating::RuntimeShape* shape) {
#define LI_LAUNCH_0(c,s,b) run(0,c,s,b)
#define LI_LAUNCH_1(c,s,b) run(1,c,s,b)
#define LI_LAUNCH_2(c,s,b) run(2,c,s,b)
#define LI_LAUNCH_3(c,s,b) run(3,c,s,b)
#include "li_launch.h"
}
int main() {
 rotating::RuntimeShape shape;auto ptr=reinterpret_cast<void*>(1);
 for(auto rounds:{2U,8U}) {
  shape.rounds=rounds;assert(rotating::UseSequenceSplit(shape));
  calls.clear();assert(!launch(ptr,ptr,&shape));assert((calls==std::vector<int>{0,1,2,3}));
  auto half=rotating::SequenceHalfWindow(shape);
  for(uint32_t token=0;token<shape.history;++token) {
   auto part=rotating::SequenceSliceAt(token,shape.resident,half);
   assert(part.end>token && part.end<=shape.history);
   if(token<shape.resident)assert(part.shard==0 && part.end==shape.resident);
   else assert(part.shard==((token-shape.resident)/half)%2);
  }
 }
 // Uneven windows retain the shared general-geometry path, not UBmem-specific logic.
 shape.rounds=4;calls.clear();assert(!launch(ptr,ptr,&shape));assert(calls.size()==1);
 shape.rounds=8;
 for(int phase=0;phase<4;++phase) {failure=phase;calls.clear();assert(launch(ptr,ptr,&shape)==99);assert(calls.size()==phase+1);}
 assert(launch(nullptr,ptr,&shape)==100000);
}
'''
    path = tmp_path / 'shared.cpp'
    path.write_text(source)
    binary = tmp_path / 'shared'
    subprocess.run(['c++', '-std=c++17', '-I', str(common), str(path), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_saved_transport_comparison_uses_full_trace(tmp_path, capsys):
    import json
    from compare_transport_tail import summarize
    for transport, filename, stage in (
        ('urma', 'full_attribution.json', 'resolve_update'),
        ('ubmem', 'attribution.json', 'resolve_update_fetch'),
    ):
        root = tmp_path / transport
        case = root / 'case0-b1-prototype-graph'
        case.mkdir(parents=True)
        samples = [dict(round=i, accounting=dict(stages_us={stage: value}))
                   for i, value in enumerate((60, 80, 70))]
        (case / filename).write_text(json.dumps(dict(samples=samples) if transport == 'urma' else samples))
        assert summarize(root, transport) == {1: 70}
    assert 'missing per-core diagnostics' in capsys.readouterr().out
