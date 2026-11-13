#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>

#define __aicore__
#define __gm__
#define __simt_callee__
namespace copy_model {
inline bool events[7][8]{};
inline unsigned worker=0;
inline std::map<uintptr_t,unsigned> owners;
inline void Write(void* output,size_t bytes) {
    auto begin=reinterpret_cast<uintptr_t>(output);
    for(uintptr_t sector=begin/32;sector<=(begin+bytes-1)/32;++sector) {
        auto result=owners.emplace(sector,worker);
        assert(result.second || result.first->second==worker);
    }
}
inline void Drained() {for(auto& direction:events)for(bool credit:direction)assert(!credit);}
}
namespace AscendC {
using event_t=unsigned;
constexpr event_t EVENT_ID0=0;
enum class HardEvent {S_MTE2,MTE2_S,S_MTE3,MTE3_S,MTE3_MTE2,MTE2_MTE3};
template<HardEvent Direction> void SetFlag(event_t event) {
    assert(event<8);auto& credit=copy_model::events[static_cast<unsigned>(Direction)][event];assert(!credit);credit=true;
}
template<HardEvent Direction> void WaitFlag(event_t event) {
    assert(event<8);auto& credit=copy_model::events[static_cast<unsigned>(Direction)][event];assert(credit);credit=false;
}
template<class T> struct LocalTensor {
    T* data;size_t bytes;
    LocalTensor operator[](size_t index)const {assert(index*sizeof(T)<=bytes);return {data+index,bytes-index*sizeof(T)};}
    T GetValue(size_t index)const {assert((index+1)*sizeof(T)<=bytes);T result;std::memcpy(&result,data+index,sizeof(T));return result;}
    void SetValue(size_t index,T value)const {assert((index+1)*sizeof(T)<=bytes);std::memcpy(data+index,&value,sizeof(T));}
    template<class U> LocalTensor<U> ReinterpretCast()const {return {reinterpret_cast<U*>(data),bytes};}
};
template<class T> struct GlobalTensor {
    T* data;
    void SetGlobalBuffer(T* pointer) {data=pointer;}
};
enum class PaddingMode {Normal,Compact};
struct DataCopyExtParams {uint16_t blockCount;uint32_t blockLen,srcStride,dstStride,reserved;};
template<class T> struct DataCopyPadExtParams {bool enabled;uint8_t left,right;T value;};
template<class T,PaddingMode Mode>
void DataCopyPad(LocalTensor<T> destination,GlobalTensor<T> source,DataCopyExtParams copy,DataCopyPadExtParams<T>) {
    size_t localStride=(Mode==PaddingMode::Compact?copy.blockLen:(copy.blockLen+31)/32*32)+copy.dstStride*32;
    auto* out=reinterpret_cast<uint8_t*>(destination.data);
    auto* in=reinterpret_cast<uint8_t*>(source.data);
    for(unsigned block=0;block<copy.blockCount;++block) {
        assert(block*localStride+copy.blockLen<=destination.bytes);
        std::memcpy(out+block*localStride,in+block*(copy.blockLen+copy.srcStride),copy.blockLen);
    }
}
template<class T,PaddingMode>
void DataCopyPad(GlobalTensor<T> destination,LocalTensor<T> source,DataCopyExtParams copy) {
    size_t localStride=(copy.blockLen+31)/32*32+copy.srcStride*32;
    auto* out=reinterpret_cast<uint8_t*>(destination.data);
    auto* in=reinterpret_cast<uint8_t*>(source.data);
    for(unsigned block=0;block<copy.blockCount;++block) {
        assert(block*localStride+copy.blockLen<=source.bytes);
        copy_model::Write(out+block*(copy.blockLen+copy.dstStride),copy.blockLen);
        std::memcpy(out+block*(copy.blockLen+copy.dstStride),in+block*localStride,copy.blockLen);
    }
}
}
using namespace AscendC;

#include "context.h"
#include "ubmem_copy.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using namespace rotating;
uint64_t Address(const void* value) {return reinterpret_cast<uintptr_t>(value);}

void CheckOutputLayout(uint32_t resident,uint32_t tailTokens) {
    // This runs the production ring code. It validates byte extents, sector
    // ownership and event credits, not hardware DMA overlap or visibility.
    constexpr uint32_t sourceRows=128;
    const uint32_t stride=resident+tailTokens;
    const size_t outputBytes=uint64_t(SIM_BATCH+1)*stride*kTokenBytes;
    const size_t storageBytes=(outputBytes+128+63)/64*64;
    std::unique_ptr<uint8_t,decltype(&std::free)> allocation(
        static_cast<uint8_t*>(std::aligned_alloc(64,storageBytes)),&std::free);
    assert(allocation);
    auto* output=allocation.get()+64;
    std::vector<uint8_t> source(sourceRows*kTokenBytes+3);
    for(size_t i=0;i<source.size();++i)source[i]=uint8_t((i*23+i/11)%251);
    UbmemRegion region{kUbmemMagic,kUbmemVersion,0x100000,Address(source.data()),source.size(),7,
        Address(output),outputBytes,0,kUbmemHostLocal};
    assert(UbmemRegionValid(&region,resident));
    // Packed batch size does not describe the registered stable-owner extent.
    region.outputBytes=uint64_t(resident)*kTokenBytes;
    assert(UbmemRegionValid(&region,resident));
    --region.outputBytes;
    assert(!UbmemRegionValid(&region,resident));
    region.outputBytes=outputBytes;
    Context ctx{};ctx.ubmemRegion=Address(&region);ctx.shape.resident=resident;
    std::vector<uint8_t> scratch(32000),overflow(12288);
    LocalTensor<uint8_t> local{scratch.data(),scratch.size()},extra{overflow.data(),overflow.size()};
    std::vector<uint8_t> expected(outputBytes);
    auto reset=[&] {std::memset(allocation.get(),0xa9,storageBytes);std::fill(expected.begin(),expected.end(),0xa9);copy_model::owners.clear();};
    auto check=[&] {
        assert(std::memcmp(output,expected.data(),outputBytes)==0);
        for(size_t i=0;i<64;++i)assert(allocation.get()[i]==0xa9 && output[outputBytes+i]==0xa9);
        copy_model::Drained();
    };
    reset();
    // Window: one AIV owns each batch, with irregular direct-hot destinations.
    for(uint32_t b=0;b<SIM_BATCH;++b) {
        copy_model::worker=b;
        UbmemWindowCopyPipeline pipe(&ctx,0,local,extra);
        uint64_t base=Address(output)+uint64_t(b+1)*stride*kTokenBytes;
        for(uint32_t i=0;i<39;++i) {
            uint32_t slot=(i*19)%61;
            uint64_t offset=3+uint64_t((i*7)%sourceRows)*kTokenBytes;
            assert(pipe.AddAt(region.peerBase+offset,base,slot));
            std::memcpy(expected.data()+(uint64_t(b+1)*stride+slot)*kTokenBytes,source.data()+offset,kTokenBytes);
        }
        assert(!pipe.AddAt(region.peerBase+region.bytes-1,base,71));
        // ABC and the next owner may be inside the registration, but Fetch
        // must still reject them as destinations for this owner's hot rows.
        assert(!pipe.AddAt(region.peerBase,base,resident));
        assert(!pipe.AddAt(region.peerBase,base,resident+kSlots-1));
        assert(!pipe.AddAt(region.peerBase,region.outputBase-1,0));
        assert(!pipe.AddAt(region.peerBase,region.outputBase+region.outputBytes,0));
        assert(pipe.Finish()==39);
        copy_model::Drained();
    }
    check();
    reset();
    // Window grouped fast paths, including the exact 6/4/1 remainder paths.
    for(uint32_t b=0;b<SIM_BATCH;++b) {
        copy_model::worker=b;
        UbmemWindowCopyPipeline pipe(&ctx,0,local,extra);
        uint64_t base=Address(output)+uint64_t(b+1)*stride*kTokenBytes;
        assert(pipe.BeginRunAt(base,resident-32,31));
        uint32_t copied=0;
        while(copied<31) {
            auto peer=[&](uint32_t j){return region.peerBase+3+uint64_t(copied+j)*kTokenBytes;};
            uint32_t n=1;
            if(31-copied>=8 && pipe.TryAddNext8(peer(0),peer(1),peer(2),peer(3),peer(4),peer(5),peer(6),peer(7)))n=8;
            else if(31-copied>=6 && pipe.TryAddNext6(peer(0),peer(1),peer(2),peer(3),peer(4),peer(5)))n=6;
            else if(31-copied>=4 && pipe.TryAddNext4(peer(0),peer(1),peer(2),peer(3)))n=4;
            else assert(pipe.AddNext(peer(0)));
            copied+=n;
        }
        assert(pipe.Finish()==31);
        assert(!pipe.BeginRunAt(base,resident,1));
        assert(!pipe.BeginRunAt(base,resident-1,2));
        assert(!pipe.BeginRunAt(base,0,0));
        assert(!pipe.BeginRunAt(region.outputBase+region.outputBytes,0,1));
        std::memcpy(expected.data()+(uint64_t(b+1)*stride+resident-32)*kTokenBytes,source.data()+3,31*kTokenBytes);
        copy_model::Drained();
    }
    check();
    assert(UbmemWindowCopyPipeline::kScratchBytes<=32000);
    assert(4096+UbmemWindowCopyPipeline::kOverflowBytes<=16384);
    std::cout<<"UBmem fused copy model PASS B="<<SIM_BATCH<<" token="<<kTokenBytes
             <<" tail="<<tailTokens
             <<" window_ub="<<UbmemWindowCopyPipeline::kScratchBytes
             <<" overflow="<<UbmemWindowCopyPipeline::kOverflowBytes
             <<"\n";
}

int main() {
    for(uint32_t tail:{0U,384U,640U})CheckOutputLayout(SIM_RESIDENT,tail);
    if(SIM_BATCH<=2)CheckOutputLayout(SIM_RESIDENT==8192?2048:8192,0);
}
