// SPDX-License-Identifier: Apache-2.0
// Drive the actual publisher with all 512 lane offsets. The prefix sum is
// supplied by an independent stable partition; the oracle checks every field.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>
#define __gm__
#define __ubuf__
#define __aicore__
#define __simt_vf__
#define LAUNCH_BOUND(n)
constexpr unsigned kTop=2048,kSlots=kTop*4,warpSize=32;
struct { unsigned x; } threadIdx;
void asc_syncthreads() {}
void asc_threadfence() {}
template<class T>T asc_ldcg(const T* p){return *p;}
template<class T>void asc_stcg(T* p,T v){*p=v;}
void asc_atomic_or(uint32_t* p,uint32_t v){*p|=v;}
void asc_atomic_add(uint32_t* p,uint32_t v){*p+=v;}
struct Context {uint64_t sourceBlocks{},sourceOffsets{},destinations{},fallbackAddresses{},
    reservedHot{},rows{},count{}; uint32_t owner=1;};
using StagedContextPointer=Context*;
struct Row {uint64_t addresses;};
unsigned OwnerRow(Context* c,unsigned){return c->owner;}
unsigned OutputStride(unsigned n,Context*){return n+512;}
std::array<unsigned,513> prefix;
unsigned StagedPrefix(unsigned count,unsigned* ctl) {
    assert(count==prefix[threadIdx.x+1]-prefix[threadIdx.x]);
    ctl[544+512/warpSize-1]=prefix.back();
    return prefix[threadIdx.x];
}
void StoreFallbackSource(unsigned,unsigned,Context* ctx,unsigned,unsigned pos,unsigned,uint64_t address,int* blocks,int* offsets){
    reinterpret_cast<uint64_t*>(ctx->fallbackAddresses)[pos]=address;blocks[pos]=pos;offsets[pos]=0;
}
#include "pack_row_events.h"

int main(){
    constexpr unsigned resident=2048,history=8192,bytes=656,batch=1,q=2,rawBegin=7;
    constexpr uint64_t remote=0x200000000ULL;
    constexpr unsigned base=batch*kSlots+q*kTop,used=512;
    const unsigned capacity=2*kSlots;
    std::vector<uint32_t> source(rawBegin+kTop),slots(rawBegin+kTop),scratch(kTop),ctl(600);
    std::vector<int32_t> block(capacity),offset(capacity),destination(capacity),reserved(2*resident);
    std::vector<uint64_t> fallback(capacity),addresses(history);
    uint32_t submitted=0;
    Row table[2]{{0},{uint64_t(addresses.data())}};
    Context ctx{uint64_t(block.data()),uint64_t(offset.data()),uint64_t(destination.data()),
        uint64_t(fallback.data()),uint64_t(reserved.data()),uint64_t(table),uint64_t(&submitted)};
    std::vector<unsigned> stagedTokens(kTop);

    for (unsigned n:{1U,32U,63U,64U,65U,511U,512U,513U,1200U,2048U}) for (unsigned fault=0;fault<3;++fault) {
        std::fill(fallback.begin(),fallback.end(),0);
        std::fill(block.begin(),block.end(),-1);
        std::fill(ctl.begin(),ctl.end(),0);submitted=0;
        for(unsigned i=0;i<history;++i)addresses[i]=remote+uint64_t(i)*bytes;
        for(unsigned i=0;i<resident;++i)reserved[batch*resident+i]=i%4 ? -1 : int(i%used);
        for(unsigned i=0;i<n;++i){slots[rawBegin+i]=i;source[rawBegin+i]=(history-1)-i;}
        const unsigned test=n/2;
        if(fault==1)slots[rawBegin+test]=resident;
        if(fault==2)source[rawBegin+test]=history;
        const unsigned per=(n+511)/512;
        std::vector<unsigned> order;
        std::array<unsigned,512> local{};prefix.fill(0);
        for(unsigned phase=0;phase<2;++phase)for(unsigned i=0;i<n;++i){
            unsigned s=slots[rawBegin+i];if(s>=resident)continue;
            bool conflict=reserved[batch*resident+s]>=0;
            if(!phase && !conflict)++local[i/per];
            if(unsigned(conflict)==phase)order.push_back(i);
        }
        std::partial_sum(local.begin(),local.end(),prefix.begin()+1);
        for(unsigned lane=512;lane--;) {
            threadIdx.x=lane;
            PackRowEvents(&ctx,history,resident,bytes,batch,q,used,rawBegin,n,source.data(),slots.data(),
                stagedTokens.data(),scratch.data(),ctl.data());
        }
        assert(bool(ctl[1])==bool(fault));
        assert(submitted==(fault?0:n));
        if(fault)continue;
        assert(ctl[16]==n && ctl[17]==prefix.back());
        for(unsigned pos=0;pos<capacity;++pos){
            if(pos<base || pos>=base+n){assert(block[pos]==-1 && !fallback[pos]);continue;}
            unsigned i=order[pos-base];
            assert(block[pos]==int(pos) && offset[pos]==0 && destination[pos]==int(resident+512+i));
            assert(fallback[pos]==addresses[source[rawBegin+i]]);
        }
    }
}
