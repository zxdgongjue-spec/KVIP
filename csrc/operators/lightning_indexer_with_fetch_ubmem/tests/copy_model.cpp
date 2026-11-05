#include "copy_model.h"
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

int main() {
    // This runs the production ring code. It validates byte extents, sector
    // ownership and event credits, not hardware DMA overlap or visibility.
    constexpr uint32_t sourceRows=128,stride=SIM_RESIDENT+kSlots+384;
    const size_t outputBytes=uint64_t(SIM_BATCH)*stride*kTokenBytes;
    const size_t storageBytes=(outputBytes+128+63)/64*64;
    std::unique_ptr<uint8_t,decltype(&std::free)> allocation(
        static_cast<uint8_t*>(std::aligned_alloc(64,storageBytes)),&std::free);
    assert(allocation);
    auto* output=allocation.get()+64;
    std::vector<uint8_t> source(sourceRows*kTokenBytes+3);
    for(size_t i=0;i<source.size();++i)source[i]=uint8_t((i*23+i/11)%251);
    UbmemRegion region{kUbmemMagic,kUbmemVersion,0x100000,Address(source.data()),source.size(),7,
        Address(output),outputBytes,0,kUbmemHostLocal};
    Context ctx{};ctx.ubmemRegion=Address(&region);
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
        uint64_t base=Address(output)+uint64_t(b)*stride*kTokenBytes;
        for(uint32_t i=0;i<39;++i) {
            uint32_t slot=(i*19)%61;
            uint64_t offset=3+uint64_t((i*7)%sourceRows)*kTokenBytes;
            assert(pipe.AddAt(region.peerBase+offset,base,slot));
            std::memcpy(expected.data()+(uint64_t(b)*stride+slot)*kTokenBytes,source.data()+offset,kTokenBytes);
        }
        assert(!pipe.AddAt(region.peerBase+region.bytes-1,base,71));
        assert(pipe.Finish()==39);
        copy_model::Drained();
    }
    check();
    reset();
    // Window grouped fast paths, including the exact 6/4/1 remainder paths.
    for(uint32_t b=0;b<SIM_BATCH;++b) {
        copy_model::worker=b;
        UbmemWindowCopyPipeline pipe(&ctx,0,local,extra);
        uint64_t base=Address(output)+uint64_t(b)*stride*kTokenBytes;
        assert(pipe.BeginRunAt(base,SIM_RESIDENT,31));
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
        std::memcpy(expected.data()+(uint64_t(b)*stride+SIM_RESIDENT)*kTokenBytes,source.data()+3,31*kTokenBytes);
        copy_model::Drained();
    }
    check();
    reset();
    // Replicated final workers select actual destination sectors after an
    // arbitrary LRU mapping. Consecutive logical ranks are not safe owners.
    std::vector<uint32_t> slots;
    for(uint32_t i=0;i<61;++i)slots.push_back((i*17)%61);
    for(uint32_t b=0;b<SIM_BATCH;++b) {
        for(uint32_t worker=0;worker<kFinalWorkers;++worker) {
            copy_model::worker=b*kFinalWorkers+worker;
            uint64_t base=Address(output)+uint64_t(b)*stride*kTokenBytes;
            UbmemFinalCopyPipeline pipe(&ctx,base,local);
            uint32_t accepted=0;
            for(uint32_t i=0;i<slots.size();++i)if(UbmemDestinationOwner(slots[i],kFinalWorkers)==worker) {
                uint64_t offset=3+uint64_t((i*5)%sourceRows)*kTokenBytes;
                assert(pipe.Add(region.peerBase+offset,slots[i]));++accepted;
                std::memcpy(expected.data()+(uint64_t(b)*stride+slots[i])*kTokenBytes,source.data()+offset,kTokenBytes);
            }
            assert(pipe.Finish()==accepted);copy_model::Drained();
        }
    }
    check();
    assert(UbmemWindowCopyPipeline::kScratchBytes<=32000);
    assert(4096+UbmemWindowCopyPipeline::kOverflowBytes<=16384);
    assert(UbmemFinalCopyPipeline::kScratchBytes<=23808);
    std::cout<<"UBmem fused copy model PASS B="<<SIM_BATCH<<" token="<<kTokenBytes
             <<" window_ub="<<UbmemWindowCopyPipeline::kScratchBytes
             <<" overflow="<<UbmemWindowCopyPipeline::kOverflowBytes
             <<" final_ub="<<UbmemFinalCopyPipeline::kScratchBytes<<"\n";
}
