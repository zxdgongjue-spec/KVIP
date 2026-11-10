#include "copy_model.h"
#include "context.h"
#include "ubmem_copy.h"
#include "../op_kernel/resolver/final_scratch.h"
namespace ubmem_device {
inline void CleanCacheLines(uint8_t*,size_t) {}
}
#include "../op_kernel/resolver/final_worker_submit.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using namespace rotating;
uint64_t Address(const void* value) {return reinterpret_cast<uintptr_t>(value);}

void CheckOutputLayout(uint32_t tailTokens) {
    // This runs the production ring code. It validates byte extents, sector
    // ownership and event credits, not hardware DMA overlap or visibility.
    constexpr uint32_t sourceRows=128;
    const uint32_t stride=SIM_RESIDENT+tailTokens;
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
    assert(UbmemRegionValid(&region));
    // Packed batch size does not describe the registered stable-owner extent.
    region.outputBytes=uint64_t(SIM_RESIDENT)*kTokenBytes;
    assert(UbmemRegionValid(&region));
    --region.outputBytes;
    assert(!UbmemRegionValid(&region));
    region.outputBytes=outputBytes;
    Context ctx{};ctx.ubmemRegion=Address(&region);
    std::vector<uint8_t> scratch(std::max(32000U,UbmemFinalCopyPipeline::kScratchBytes)),overflow(12288);
    LocalTensor<uint8_t> local{scratch.data(),32000},extra{overflow.data(),overflow.size()};
    LocalTensor<uint8_t> finalLocal{scratch.data(),scratch.size()};
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
        assert(!pipe.AddAt(region.peerBase,base,SIM_RESIDENT));
        assert(!pipe.AddAt(region.peerBase,base,SIM_RESIDENT+kSlots-1));
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
        assert(pipe.BeginRunAt(base,SIM_RESIDENT-32,31));
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
        assert(!pipe.BeginRunAt(base,SIM_RESIDENT,1));
        assert(!pipe.BeginRunAt(base,SIM_RESIDENT-1,2));
        assert(!pipe.BeginRunAt(base,0,0));
        assert(!pipe.BeginRunAt(region.outputBase+region.outputBytes,0,1));
        std::memcpy(expected.data()+(uint64_t(b+1)*stride+SIM_RESIDENT-32)*kTokenBytes,source.data()+3,31*kTokenBytes);
        copy_model::Drained();
    }
    check();
    reset();
    // Replicated final workers select actual destination sectors after an
    // arbitrary LRU mapping. Consecutive logical ranks are not safe owners.
    std::vector<uint32_t> slots;
    for(uint32_t i=0;i<61;++i)slots.push_back((i*17)%61);
    slots.push_back(SIM_RESIDENT-2);slots.push_back(SIM_RESIDENT-1);
    for(uint32_t b=0;b<SIM_BATCH;++b) {
        for(uint32_t worker=0;worker<kFinalWorkers;++worker) {
            copy_model::worker=b*kFinalWorkers+worker;
            uint64_t base=Address(output)+uint64_t(b+1)*stride*kTokenBytes;
            UbmemFinalCopyPipeline pipe(&ctx,base,finalLocal);
            uint32_t accepted=0;
            for(uint32_t i=0;i<slots.size();++i)if(UbmemDestinationOwner(slots[i],kFinalWorkers)==worker) {
                uint64_t offset=3+uint64_t((i*5)%sourceRows)*kTokenBytes;
                assert(pipe.Add(region.peerBase+offset,slots[i]));++accepted;
                std::memcpy(expected.data()+(uint64_t(b+1)*stride+slots[i])*kTokenBytes,source.data()+offset,kTokenBytes);
            }
            assert(!pipe.Add(region.peerBase,SIM_RESIDENT));
            assert(pipe.Finish()==accepted);copy_model::Drained();
        }
    }
    check();
    assert(UbmemWindowCopyPipeline::kScratchBytes<=32000);
    assert(4096+UbmemWindowCopyPipeline::kOverflowBytes<=16384);
    assert(UbmemFinalCopyPipeline::kScratchBytes<=2*kSlots*sizeof(uint32_t));
    std::cout<<"UBmem fused copy model PASS B="<<SIM_BATCH<<" token="<<kTokenBytes
             <<" paired="<<V9_UBMEM_GROUPED_COPY
             <<" final_stages="<<V9_UBMEM_FINAL_COPY_STAGES
             <<" tail="<<tailTokens
             <<" window_ub="<<UbmemWindowCopyPipeline::kScratchBytes
             <<" overflow="<<UbmemWindowCopyPipeline::kOverflowBytes
             <<" final_ub="<<UbmemFinalCopyPipeline::kScratchBytes<<"\n";
}

#if V9_UBMEM_GROUPED_COPY
struct alignas(32) FinalOutput {std::array<uint8_t,SIM_RESIDENT*kTokenBytes> bytes{};};

template<uint32_t Stages>
void CheckFinalLookahead() {
    using Pipeline=UbmemPairCopyPipelineT<Stages>;
    constexpr uint32_t count=3*2*Stages+1;
    std::vector<uint8_t> source(count*kTokenBytes,0x39);
    auto allocation=std::make_unique<FinalOutput>();auto& output=allocation->bytes;
    UbmemRegion region{kUbmemMagic,kUbmemVersion,0x300000,Address(source.data()),source.size(),7,
        Address(output.data()),output.size(),0,kUbmemHostLocal};
    Context ctx{};ctx.ubmemRegion=Address(&region);
    // The production final call may borrow selected + locations, but never
    // keys at byte 65536. Fill all unused bytes to detect late DMA spill.
    std::vector<uint8_t> scratch(2*kSlots*sizeof(uint32_t)+64,0xa7);
    LocalTensor<uint8_t> local{scratch.data(),Pipeline::kScratchBytes};
    copy_model::owners.clear();copy_model::worker=0;copy_model::peakLoadRows=0;
    Pipeline pipe(&ctx,Address(output.data()),local);
    for(uint32_t i=0;i<count;++i) {
        assert(pipe.Add(region.peerBase+uint64_t(i)*kTokenBytes,i));
        if(i+1<2*Stages)assert(copy_model::stores.empty());
        if(i+1==2*Stages) {
            assert(copy_model::peakLoadRows==2*Stages);
            assert(copy_model::stores.size()==1 && copy_model::storesDone==0);
            for(auto value:output)assert(value==0); // queued stores are genuinely delayed
        }
    }
    assert(pipe.Finish()==count);
    assert(copy_model::peakLoadRows==2*Stages);
    assert(std::memcmp(output.data(),source.data(),source.size())==0);
    for(size_t i=Pipeline::kScratchBytes;i<scratch.size();++i)assert(scratch[i]==0xa7);
    copy_model::Drained();
}

void CheckFinalBorrow(bool skewed) {
    // Invoke the actual final caller, with the same live keys/ranks/slots
    // offsets as ReplicatedFinal. Their contents must survive delayed DMA.
    constexpr uint32_t count=257;
    std::vector<uint8_t> source(count*kTokenBytes);
    for(size_t i=0;i<source.size();++i)source[i]=uint8_t((i*19+i/5)%251);
    auto allocation=std::make_unique<FinalOutput>();auto& output=allocation->bytes;
    std::vector<uint8_t> expected(output.size());
    UbmemRegion region{kUbmemMagic,kUbmemVersion,0x400000,Address(source.data()),source.size(),7,
        Address(output.data()),output.size(),0,kUbmemHostLocal};
    std::vector<uint64_t> owners((kFinalOwnerBytes+7)/8);
    Context ctx{};ctx.ubmemRegion=Address(&region);ctx.owners=Address(owners.data());
    Row row{};row.output=Address(output.data());
    copy_model::owners.clear();
    for(uint32_t worker=0;worker<kFinalWorkers;++worker) {
        copy_model::worker=worker;
        std::vector<uint32_t> scratch(kFinalScratchWords,0xabababab);
        LocalTensor<uint32_t> local{scratch.data(),scratch.size()*sizeof(uint32_t)};
        auto keys=local[2*kSlots],ranks=local[3*kSlots],slots=local[kFinalSlotsOffset];
        for(uint32_t rank=0;rank<count;++rank) {
            uint64_t peer=region.peerBase+uint64_t(rank)*kTokenBytes;
            keys.SetValue(rank,uint32_t(peer));ranks.SetValue(rank,uint32_t(peer>>32));
            // All skewed slots share the old sector owner. Rank ownership
            // must still cover each whole aligned row exactly once.
            slots.SetValue(rank,skewed?rank*8:rank*13%512);
        }
        auto before=scratch;
        auto beforeReads=copy_model::scalarLocalReads;
        PublishFinalWorker(&ctx,&row,0,worker,0,count,0,0,local,keys,ranks,slots,local[kFinalControlOffset]);
        auto scalarReads=copy_model::scalarLocalReads-beforeReads;
        auto* post=FinalPostAt(&ctx,worker);
        uint32_t wanted=0;
        for(uint32_t rank=0;rank<count;++rank)if(V9_UBMEM_ORDERED_PAIR?
            (rank>=count*worker/kFinalWorkers && rank<count*(worker+1)/kFinalWorkers):
            UbmemDestinationOwner(slots.GetValue(rank),kFinalWorkers)==worker) {
            ++wanted;
            std::memcpy(expected.data()+uint64_t(slots.GetValue(rank))*kTokenBytes,
                source.data()+uint64_t(rank)*kTokenBytes,kTokenBytes);
        }
        assert(post->error==0 && post->pending==0 && post->count==wanted && post->head==wanted);
        assert(scalarReads==(V9_UBMEM_ORDERED_PAIR?3*wanted:count+2*wanted));
        if(V9_UBMEM_ORDERED_PAIR)assert(wanted==count/kFinalWorkers || wanted==count/kFinalWorkers+1);
        constexpr size_t usedWords=(UbmemFinalCopyPipeline::kScratchBytes+3)/4;
        assert(std::equal(scratch.begin()+usedWords,scratch.end(),before.begin()+usedWords));
        copy_model::Drained();
    }
    assert(std::memcmp(output.data(),expected.data(),output.size())==0);
    std::cout<<"Final actual caller PASS stages="<<V9_UBMEM_FINAL_COPY_STAGES
             <<" ordered="<<V9_UBMEM_ORDERED_PAIR<<" skewed="<<skewed
             <<" borrowed_bytes="<<UbmemFinalCopyPipeline::kScratchBytes<<" live_keys_offset="<<2*kSlots*4<<"\n";
}

void CheckPairCommands() {
    constexpr uint32_t stride=SIM_RESIDENT+384,sourceRows=128;
    const size_t outputBytes=uint64_t(2)*stride*kTokenBytes;
    const size_t storageBytes=(outputBytes+128+63)/64*64;
    std::unique_ptr<uint8_t,decltype(&std::free)> allocation(
        static_cast<uint8_t*>(std::aligned_alloc(64,storageBytes)),&std::free);
    assert(allocation);
    auto* output=allocation.get()+64;
    std::vector<uint8_t> source(sourceRows*kTokenBytes+3),expected(outputBytes);
    for(size_t i=0;i<source.size();++i)source[i]=uint8_t((i*17+i/23)%251);
    UbmemRegion region{kUbmemMagic,kUbmemVersion,0x200000,Address(source.data()),source.size(),7,
        Address(output),outputBytes,0,kUbmemHostLocal};
    Context ctx{};ctx.ubmemRegion=Address(&region);
    constexpr uint32_t scratchBytes=UbmemPairCopyPipeline::kScratchBytes;
    std::vector<uint8_t> scratch(scratchBytes+64,0x8b);
    LocalTensor<uint8_t> local{scratch.data(),scratchBytes};
    struct Copy {uint64_t peer,base;uint32_t slot;};
    size_t totalBaseLoads=0,totalPairLoads=0,totalBaseStores=0,totalPairStores=0;
    // Exact pair reduction, strided gaps, reversed/repeated addresses, odd
    // tails, multiple ring wraps and pairs spanning stable-owner rows.
    for(uint32_t pattern=0;pattern<10;++pattern)for(uint32_t count:{0U,1U,2U,7U,8U,9U,17U}) {
        std::vector<Copy> copies;
        std::fill(expected.begin(),expected.end(),0xa9);
        for(uint32_t i=0;i<count;++i) {
            uint32_t sourceRow=(pattern==1 || pattern>=7)?(i^1U)*2:pattern==2?(i/2)*2:i*2;
            uint32_t slot=(pattern==3 || pattern==8)?(i^1U)*3:(pattern==4 || pattern==7)?(i/2)*3:i*3;
            uint32_t owner=pattern==5?i%2:pattern==6?(i%2?0:1):0;
            if(pattern==5 || pattern==6)slot=(i/2)*3;
            if(pattern==9)slot=(i/2)*3;
            uint64_t overlap=pattern==9?(i%2)*32:0;
            uint64_t offset=3+uint64_t(sourceRow)*kTokenBytes;
            copies.push_back({region.peerBase+offset,Address(output)+uint64_t(owner)*stride*kTokenBytes+overlap,slot});
            std::memcpy(expected.data()+(uint64_t(owner)*stride+slot)*kTokenBytes+overlap,source.data()+offset,kTokenBytes);
        }
        auto exercise=[&](auto& pipe) {
            // Failure before any accepted row must not acquire an event.
            assert(!pipe.AddAt(region.peerBase-1,Address(output),0));
            assert(!pipe.AddAt(region.peerBase+region.bytes,Address(output),0));
            for(uint32_t i=0;i<copies.size();++i) {
                const auto& item=copies[i];
                assert(pipe.AddAt(item.peer,item.base,item.slot));
                if(i==0) {
                    // Preserve an accepted but not yet paired row on failure.
                    assert(!pipe.AddAt(region.peerBase+region.bytes-1,item.base,1));
                    assert(!pipe.AddAt(item.peer,item.base,SIM_RESIDENT));
                    assert(!pipe.AddAt(item.peer,region.outputBase+region.outputBytes,0));
                }
            }
            assert(pipe.Finish()==count);copy_model::Drained();
            assert(std::memcmp(output,expected.data(),outputBytes)==0);
            for(size_t i=0;i<64;++i)assert(allocation.get()[i]==0xa9 && output[outputBytes+i]==0xa9);
            for(size_t i=scratchBytes;i<scratch.size();++i)assert(scratch[i]==0x8b);
        };
        std::memset(allocation.get(),0xa9,storageBytes);copy_model::owners.clear();copy_model::worker=0;
        auto beforeLoads=copy_model::loadCommands,beforeStores=copy_model::storeCommands;
        UbmemCopyPipeline baseline(&ctx,Address(output),local);exercise(baseline);
        auto baseLoads=copy_model::loadCommands-beforeLoads,baseStores=copy_model::storeCommands-beforeStores;
        assert(baseLoads==count && baseStores==count);
        std::memset(allocation.get(),0xa9,storageBytes);copy_model::owners.clear();
        beforeLoads=copy_model::loadCommands;beforeStores=copy_model::storeCommands;
        UbmemPairCopyPipeline paired(&ctx,Address(output),local);exercise(paired);
        auto pairLoads=copy_model::loadCommands-beforeLoads,pairStores=copy_model::storeCommands-beforeStores;
        assert(pairLoads<=baseLoads && pairStores<=baseStores);
        bool singleLoads=pattern==2 || pattern==7 || pattern==9 || (!V9_UBMEM_ORDERED_PAIR && (pattern==1 || pattern==8));
        bool singleStores=pattern==3 || pattern==4 || pattern==6 || pattern==7 || pattern==9 ||
            (V9_UBMEM_ORDERED_PAIR?pattern==1:pattern==8);
        assert(pairLoads==(singleLoads?count:(count+1)/2));
        assert(pairStores==(singleStores?count:(count+1)/2));
        if(pattern==0) {
            totalBaseLoads+=baseLoads;totalPairLoads+=pairLoads;
            totalBaseStores+=baseStores;totalPairStores+=pairStores;
        }
    }
    // Empty/invalid regions, a complete last record and run-state failures.
    {
        std::memset(allocation.get(),0xa9,storageBytes);copy_model::owners.clear();
        UbmemPairCopyPipeline pipe(&ctx,Address(output),local);
        assert(!pipe.AddAt(region.peerBase,Address(output)+1,0));
        assert(!pipe.BeginRunAt(Address(output)+1,0,1));
        assert(pipe.BeginRunAt(Address(output),SIM_RESIDENT-1,1));
        assert(!pipe.AddNext(region.peerBase+region.bytes-1));
        assert(pipe.AddNext(region.peerBase+region.bytes-kTokenBytes));
        assert(!pipe.AddNext(region.peerBase));
        assert(pipe.Finish()==1);copy_model::Drained();
        assert(std::memcmp(output+uint64_t(SIM_RESIDENT-1)*kTokenBytes,
            source.data()+source.size()-kTokenBytes,kTokenBytes)==0);
        region.generation=0;
        UbmemPairCopyPipeline invalid(&ctx,Address(output),local);
        assert(!invalid.Add(region.peerBase,0) && !invalid.BeginRunAt(Address(output),0,1));
        assert(invalid.Finish()==0);copy_model::Drained();
    }
    assert(totalPairLoads<totalBaseLoads && totalPairStores<totalBaseStores);
    std::cout<<"Paired actual-function model PASS: forward MTE2 "<<totalBaseLoads<<" -> "<<totalPairLoads
             <<", MTE3 "<<totalBaseStores<<" -> "<<totalPairStores<<"; byte-identical\n";
}
#endif

int main() {
    for(uint32_t tail:{0U,384U,640U})CheckOutputLayout(tail);
#if V9_UBMEM_GROUPED_COPY
    CheckFinalLookahead<4>();CheckFinalLookahead<8>();
    CheckFinalBorrow(false);CheckFinalBorrow(true);
    CheckPairCommands();
#endif
}
