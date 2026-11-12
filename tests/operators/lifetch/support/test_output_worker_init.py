"""Exercise the actual publisher initialization with delayed Vector writes."""
from pathlib import Path
import shutil
import subprocess


def test_context_snapshot_survives_delayed_vector_clear(tmp_path):
    root = Path(__file__).resolve().parents[4]
    source = (root / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/output_worker.h').read_text()
    # Compile the actual scalar initialization between scratch aliases and bindings.
    body = source.split('auto slots=local[kFinalSlotsOffset]', 1)[1].split(';', 1)[1].split(' auto u=', 1)[0]
    program = r'''
#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>
std::vector<std::function<void()>> pending;
enum class HardEvent {V_MTE2};
constexpr int EVENT_ID0=0, PIPE_V=0;
void Duplicate(uint32_t* p,uint32_t value,uint32_t count){
 pending.emplace_back([=]{for(uint32_t i=0;i<count;++i)p[i]=value;});
}
template<int> void PipeBarrier() {} // Does not join the independent MTE2 engine.
template<HardEvent> void SetFlag(int) {}
template<HardEvent> void WaitFlag(int) {for(auto& op:pending)op();pending.clear();}
int main(){
 uint32_t memory[2560];for(auto& x:memory)x=0xfeed1234U;
 auto ctl=memory;
 INITIALIZE
 // MTE2 is allowed to complete before an unjoined Vector clear.
 for(unsigned i=128;i<208;++i)memory[i]=0x12340000U+i;
 for(auto& op:pending)op();
 for(unsigned i=128;i<208;++i)assert(memory[i]==0x12340000U+i);
 for(unsigned i=0;i<64;++i)assert(memory[i]==0);
 assert(memory[64]==0xfeed1234U);
}
'''.replace('INITIALIZE', body)
    cpp = tmp_path / 'publisher.cpp'
    cpp.write_text(program)
    exe = tmp_path / 'publisher'
    subprocess.run([shutil.which('clang++') or 'c++', '-std=c++17', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=10)
