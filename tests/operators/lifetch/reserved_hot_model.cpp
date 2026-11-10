// SPDX-License-Identifier: Apache-2.0
// Execute the shared init helper with explicit DMA/event ordering, not device timing.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>
#define __CCE_AICORE__ 1
#define V9_COMPACT_RESERVED 1
#define __aicore__
#define __simt_vf__
#define __gm__
#define LAUNCH_BOUND(n)
struct {uint32_t x;} threadIdx;
inline uint32_t asc_ldcg(const uint32_t* p) {return *p;}
inline void asc_stcg(uint32_t* p,uint32_t value) {*p=value;}
inline void asc_threadfence() {}
namespace AscendC {
constexpr uint32_t EVENT_ID0=0;
enum class HardEvent {V_MTE3,MTE3_V,V_S};
inline bool flags[3]{};
inline std::vector<std::function<void()>> pending;
template<HardEvent event> void SetFlag(uint32_t) {assert(!flags[unsigned(event)]);flags[unsigned(event)]=true;}
template<HardEvent event> void WaitFlag(uint32_t) {
    assert(flags[unsigned(event)]);flags[unsigned(event)]=false;
    if constexpr(event==HardEvent::MTE3_V) {for(auto& op:pending)op();pending.clear();}
}
template<class T> struct LocalTensor {T* data;};
template<class T> struct GlobalTensor {
    T* data=nullptr;
    void SetGlobalBuffer(T* value) {data=value;}
    GlobalTensor operator[](uint32_t offset)const {return {data+offset};}
};
template<class T> void Duplicate(LocalTensor<T> output,T value,uint32_t count) {
    assert(pending.empty());std::fill_n(output.data,count,value);
}
template<class T> void DataCopy(GlobalTensor<T> output,LocalTensor<T> input,uint32_t count) {
    assert(count%8==0);
    pending.push_back([=] {std::copy_n(input.data,count,output.data);});
}
namespace Simt {
struct Dim3 {uint32_t x,y,z;};
template<auto function,class... Args> void VF_CALL(Dim3 grid,Args... args) {
    assert(pending.empty());
    for(threadIdx.x=0;threadIdx.x<grid.x;++threadIdx.x)function(args...);
}
}
}
#include "reserved_hot.h"

int main() {
    constexpr uint32_t batches=4,count=2048;
    std::array<uint32_t,4096> local{};
    std::vector<uint32_t> candidates(batches*count),ends(batches*4),expected;
    for(uint32_t resident:{8192U,16384U,131072U}) {
        std::vector<uint32_t> output(batches*resident+32,91);
        // Full -> shrink/reorder -> empty -> owner/row reuse. Inactive rows
        // must retain their bytes; reactivated rows must lose all old entries.
        for(uint32_t round=0;round<4;++round) {
            uint32_t active=round==0?4:round==1?1:round==2?0:3;
            std::fill(ends.begin(),ends.end(),0);
            expected=output;
            for(uint32_t b=0;b<batches;++b) {
                for(uint32_t i=0;i<count;++i)
                    candidates[b*count+i]=(resident-1024+i+round*173+b*991)%resident;
                if(b<active) {
                    ends[b*4+round%4]=130000;
                    std::fill_n(expected.begin()+b*resident,resident,UINT32_MAX);
                    for(uint32_t i=0;i<count;++i)expected[b*resident+candidates[b*count+i]]=i;
                }
                rotating::RebuildReservedHot(uint64_t(candidates.data()),uint64_t(output.data()),
                    uint64_t(ends.data()),b,resident,count,AscendC::LocalTensor<uint32_t>{local.data()});
                assert(AscendC::pending.empty());
                for(bool flag:AscendC::flags)assert(!flag);
            }
            assert(output==expected);
        }
    }
}
