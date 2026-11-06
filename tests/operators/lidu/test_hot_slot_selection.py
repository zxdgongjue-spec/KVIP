"""Execute the shared LIDU hot bitmap selector, including wrapped cursors."""

import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class HotSlotSelectionTest(unittest.TestCase):
    def test_bitmap_selection(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler required")
        root = Path(__file__).resolve().parents[3]
        source = r"""
#include <cstdint>
#include <iostream>
#include <vector>
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define __ubuf__
#define LAUNCH_BOUND(x)
namespace AscendC { namespace Simt {
unsigned tid; unsigned GetThreadIdx(){return tid;} unsigned GetThreadNum(){return 512;}
}}
#include "hot_slot_selection.h"
int main(){ unsigned budget,begin,n;
while(std::cin>>budget>>begin>>n){
 std::vector<uint32_t> bits((budget+31)/32);for(auto& v:bits)std::cin>>v;
 unsigned first=(budget-1)/32-begin/32+1,second=begin?(begin-1)/32+1:0;
 std::vector<int32_t> count(first+second),prefix(first+second),out(n+8,-1);
 for(unsigned t=0;t<512;++t){AscendC::Simt::tid=t;
 kvip_hot_slots::CountAvailableBitmapWordsSimt(bits.data(),count.data(),budget,begin,first,second);}
 unsigned selected=0;for(unsigned i=0;i<count.size();++i){prefix[i]=selected;selected+=count[i];}
 for(unsigned t=0;t<512;++t){AscendC::Simt::tid=t;
 kvip_hot_slots::WriteAvailableBitmapSlotsBody(bits.data(),count.data(),prefix.data(),out.data(),budget,begin,first,second,n,4);}
 for(auto v:out)std::cout<<v<<' ';std::cout<<'\n';
}}
"""
        rng = random.Random(21)
        cases = []
        for budget in (1, 31, 32, 33, 8192, 16384, 32768):
            for begin in (0, budget // 2, budget - 1):
                protected = {i for i in range(budget) if rng.random() < 0.65}
                free = [i for i in list(range(begin, budget)) + list(range(begin)) if i not in protected]
                n = min(len(free), 4096)
                words = [sum(1 << (i % 32) for i in protected if i // 32 == w) for w in range((budget + 31) // 32)]
                cases.append((budget, begin, n, words, [-1] * 4 + free[:n] + [-1] * 4))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "test.cpp").write_text(source)
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    str(p / "test.cpp"),
                    "-I" + str(root / "csrc/operators/lidu/op_kernel"),
                    "-o",
                    str(p / "test"),
                ],
                check=True,
            )
            data = "".join(" ".join(map(str, [b, c, n, *w])) + "\n" for b, c, n, w, _ in cases)
            r = subprocess.run([str(p / "test")], input=data, text=True, capture_output=True, check=True)
            self.assertEqual(len(r.stdout.splitlines()), len(cases))
            for case, line in zip(cases, r.stdout.splitlines()):
                self.assertEqual(list(map(int, line.split())), case[-1])


if __name__ == "__main__":
    unittest.main()
