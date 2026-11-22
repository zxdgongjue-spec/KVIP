"""Compile the isolated host entry and verify exact prefix dispatch/error flow."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'csrc/operators/common/lifetch').is_dir())


class PrefixDispatchTest(unittest.TestCase):
    def test_launch_order_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host = root / 'operators/lightning_indexer_with_fetch/op_host'
            host.mkdir(parents=True)
            shutil.copytree(ROOT / 'csrc/operators/common', root / 'operators/common')
            header = (HERE / 'launch_prefix.h').read_text()
            source = r'''
#include <cassert>
#include <vector>
#include "runtime_shape.h"
namespace rotating {
#include "sequence_schedule.h"
}
static std::vector<int> calls;
static int failure=-1;
static void* expected=reinterpret_cast<void*>(1);
uint32_t invoke(int id,uint32_t n,void* stream,void* bindings) {
 assert(n==32 && stream==expected && bindings==expected);
 calls.push_back(id);return failure==id?17:0;
}
uint32_t aclrtlaunch_v9_li_kernel(uint32_t n,void*s,void*b){return invoke(0,n,s,b);}
uint32_t aclrtlaunch_v9_li_kernel_1(uint32_t n,void*s,void*b){return invoke(1,n,s,b);}
uint32_t aclrtlaunch_v9_li_kernel_2(uint32_t n,void*s,void*b){return invoke(2,n,s,b);}
''' + header + r'''
int main(){
 rotating::RuntimeShape s{};s.batch=16;s.history=131328;s.resident=16384;
 s.indexHeads=32;s.aicCount=32;s.splitQ=1;s.liCores=32;s.rounds=4;s.finalWorkers=4;
 assert(kvip_parallel_tail_prefix(expected,expected,&s)==0);
 assert((calls==std::vector<int>{0,1,2}));
 for(int i=0;i<3;++i){calls.clear();failure=i;
  assert(kvip_parallel_tail_prefix(expected,expected,&s)==17);
  assert(calls.size()==unsigned(i+1));}
 failure=-1;calls.clear();s.splitQ=0;
 assert(kvip_parallel_tail_prefix(expected,expected,&s)==100000 && calls.empty());
 s.splitQ=1;s.aicCount=16;
 assert(kvip_parallel_tail_prefix(expected,expected,&s)==100000 && calls.empty());
 assert(kvip_parallel_tail_prefix(nullptr,expected,&s)==100000 && calls.empty());
}
'''
            path = host / 'probe.cpp'
            path.write_text(source)
            binary = root / 'probe'
            subprocess.run(['c++', '-std=c++17', str(path), '-o', str(binary),
                            '-I' + str(ROOT / 'csrc/operators/lightning_indexer_with_fetch/include'),
                            '-I' + str(ROOT / 'csrc/operators/common/lifetch')], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
