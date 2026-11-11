"""Exercise the actual descriptor-range cache synchronization with host stubs."""
import subprocess
from pathlib import Path


def test_descriptor_sync_uses_only_published_range(tmp_path):
    root = Path(__file__).resolve().parents[4]
    source = (root / 'csrc/operators/lightning_indexer_with_fetch_streaming_fetch/op_kernel/streaming.cpp').read_text()
    start = source.index('__aicore__ inline void SyncFetchDescriptors(')
    end = source.index('\n}', start) + 2
    function = source[start:end]
    code = r'''
#include <cstdint>
#include <vector>
#include <cassert>
#define __aicore__
#define __gm__
struct Context { uint64_t sourceBlocks, sourceOffsets, destinations, fallbackAddresses; };
namespace kvaf_device {
std::vector<std::pair<uint64_t,uint64_t>> calls;
void CleanCacheLines(uint8_t* p,uint64_t n) { calls.emplace_back(uint64_t(p),n); }
}
''' + function + r'''
int main() {
 Context c{0x10000,0x20000,0x30000,0x40000};
 SyncFetchDescriptors(&c, 8195, 23);
 assert(kvaf_device::calls.size()==4);
 assert((kvaf_device::calls[0]==std::pair<uint64_t,uint64_t>(0x10000+8195*4,23*4)));
 assert((kvaf_device::calls[1]==std::pair<uint64_t,uint64_t>(0x20000+8195*4,23*4)));
 assert((kvaf_device::calls[2]==std::pair<uint64_t,uint64_t>(0x30000+8195*4,23*4)));
 assert((kvaf_device::calls[3]==std::pair<uint64_t,uint64_t>(0x40000+8195*8,23*8)));
 kvaf_device::calls.clear(); SyncFetchDescriptors(&c,8195,0); assert(kvaf_device::calls.empty());
 c.fallbackAddresses=0; SyncFetchDescriptors(&c,2,1); assert(kvaf_device::calls.size()==3);
}
'''
    cpp = tmp_path / 'sync.cpp'
    cpp.write_text(code)
    executable = tmp_path / 'sync'
    subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
