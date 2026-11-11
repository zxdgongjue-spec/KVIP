// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <cassert>
#include <cstdint>
#include "threshold_trace.h"

int main() {
    using namespace kvip_score_trace;
    for(uint32_t history : {8192U,33024U,131328U}) {
        alignas(128) std::array<uint32_t,4096> output{};
        const uint32_t words=QueryWords(history),capacity=Capacity(history);
        assert(words%32==0 && words>=capacity*4);
        for(uint32_t q=0;q<8;++q) {
            uint32_t resident=history<16384?4096:16384;
            auto line=Store(output.data(),8,history,q,resident,resident,0x8000,1);
            assert(reinterpret_cast<uintptr_t>(line)%128==0);
            assert(line==output.data()+q*words);
            assert(output[q*words+2]==0x8000);
            if(history>16384) {
                line=Store(output.data(),8,history,q,history-3,history,0xbf80,3);
                uint32_t slot=(history+16383)/16384-1;
                assert(line>=output.data()+q*words && line<output.data()+(q+1)*words);
                assert(reinterpret_cast<uintptr_t>(line)%128==0);
                assert(output[q*words+slot*4]==history-3);
                assert(output[q*words+slot*4+1]==history);
                assert(output[q*words+slot*4+2]==0xbf80);
            }
            for(uint32_t i=(capacity-1)*4;i<words;++i)assert(output[q*words+i]==0);
        }
        auto saved=output;
        assert(!Store(nullptr,8,history,0,4096,4096,0x8000,1));
        assert(!Store(output.data(),8,history,8,4096,4096,0x8000,1));
        assert(!Store(output.data(),8,history,0,history+1,4096,0x8000,1));
        assert(!Store(output.data(),8,history,0,4096,4096,65536,1));
        assert(!Store(output.data(),8,history,0,4096,4096,0x8000,4));
        assert(saved==output);
    }
}
