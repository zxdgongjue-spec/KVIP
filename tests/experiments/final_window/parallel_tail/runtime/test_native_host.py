"""Verify the combined real-call entry never launches a tail after prefix failure."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'csrc/operators/common/lifetch').is_dir())


class NativeHostTest(unittest.TestCase):
    def test_native_call_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy2(HERE / 'abi.h', root / 'terminal_runtime_abi.h')
            source = (HERE / 'host.cpp').read_text() + r'''
#include <cassert>
static int prefixCalls=0,tailCalls=0;
static uint32_t prefixError=0,tailError=0;
static void* token=reinterpret_cast<void*>(1);
uint32_t prefix(void* bindings,void* stream,const rotating::RuntimeShape* shape){
 assert(bindings==token && stream==token && shape->batch==16);
 ++prefixCalls;return prefixError;
}
extern "C" uint32_t aclrtlaunch_v9_parallel_terminal(uint32_t blocks,void* stream,void* binding){
 assert(blocks==64 && stream==token && binding==token);
 ++tailCalls;return tailError;
}
int main(){
 rotating::RuntimeShape s{};s.batch=16;s.aicCount=32;s.finalWorkers=4;s.indexHeads=32;
 assert(kvip_parallel_tail_forward(prefix,token,token,token,&s)==0);
 assert(prefixCalls==1 && tailCalls==1);
 prefixError=17;
 assert(kvip_parallel_tail_forward(prefix,token,token,token,&s)==17);
 assert(prefixCalls==2 && tailCalls==1);
 prefixError=0;tailError=23;
 assert(kvip_parallel_tail_forward(prefix,token,token,token,&s)==23);
 assert(prefixCalls==3 && tailCalls==2);
 s.aicCount=16;
 assert(kvip_parallel_tail_forward(prefix,token,token,token,&s)==100000);
 assert(prefixCalls==3 && tailCalls==2);
 assert(kvip_parallel_tail_forward(nullptr,token,token,token,&s)==100000);
 assert(kvip_parallel_tail_abi_version()==1 && kvip_parallel_tail_binding_bytes()==128);
}
'''
            (root / 'probe.cpp').write_text(source)
            subprocess.run(['c++', '-std=c++17', str(root / 'probe.cpp'), '-o', str(root / 'probe'),
                            '-I' + str(ROOT / 'csrc/operators/lightning_indexer_with_fetch/include')], check=True)
            subprocess.run([str(root / 'probe')], check=True)


if __name__ == '__main__':
    unittest.main()
