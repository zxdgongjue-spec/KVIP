// SPDX-License-Identifier: Apache-2.0
#include "copy_model.h"
#include "context.h"
#include "ubmem_copy.h"
#include "final_scratch.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
namespace ubmem_device {inline void CleanCacheLines(uint8_t*,size_t){}}
#include "final_worker_submit.h"
template<class T>T asc_ldcg(T* p){return *p;}
void asc_atomic_or(unsigned* p,unsigned value){*p|=value;}
struct{unsigned x=0;}threadIdx;
namespace rotating {
#include "load_plan_under_test.h"
}
using namespace rotating;
uint64_t Address(const void* p){return reinterpret_cast<uintptr_t>(p);}
int main(){
    constexpr unsigned b=1,owner=2,sourceRows=513,tail=384,stride=SIM_RESIDENT+tail;
    const size_t outputBytes=3ULL*stride*kTokenBytes;
    std::unique_ptr<uint8_t,decltype(&std::free)> output(static_cast<uint8_t*>(std::aligned_alloc(64,(outputBytes+63)/64*64)),&std::free);
    assert(output);std::vector<uint8_t> source(sourceRows*kTokenBytes),expected(outputBytes);
    for(size_t i=0;i<source.size();++i)source[i]=uint8_t((i*19+i/7)%251);
    UbmemRegion region{kUbmemMagic,kUbmemVersion,0x100000,Address(source.data()),source.size(),7,
        Address(output.get()),outputBytes,0,kUbmemHostLocal};
    std::vector<uint64_t> owners((kFinalOwnerBytes+7)/8),addresses(SIM_HISTORY),fallback(SIM_BATCH*kSlots);
    std::vector<int32_t> blocks(SIM_BATCH*kSlots),offsets(SIM_BATCH*kSlots),destinations(SIM_BATCH*kSlots);
    unsigned rowOwners[SIM_BATCH]={0,owner};
    for(unsigned token=0;token<SIM_HISTORY;++token)addresses[token]=region.peerBase+uint64_t(token%sourceRows)*kTokenBytes;
    Context ctx{};ctx.ubmemRegion=Address(&region);ctx.owners=Address(owners.data());
    ctx.rowOwners=Address(rowOwners);ctx.outputStride=stride;
    ctx.sourceBlocks=Address(blocks.data());ctx.sourceOffsets=Address(offsets.data());ctx.destinations=Address(destinations.data());
    Row row{};row.addresses=Address(addresses.data());row.output=Address(output.get())+uint64_t(owner)*stride*kTokenBytes;
    for(bool direct:{false,true})for(unsigned count:{0U,1U,257U}){
        std::memset(output.get(),0x9b,outputBytes);std::fill(expected.begin(),expected.end(),0x9b);copy_model::owners.clear();
        ctx.fallbackAddresses=direct?Address(fallback.data()):0;
        for(unsigned rank=0;rank<count;++rank){
            const unsigned token=(rank*47+1031)%SIM_HISTORY,slot=(rank*17)%1024,d=b*kSlots+rank;
            fallback[d]=addresses[token];blocks[d]=int32_t(b*(SIM_HISTORY/128)+token/128);
            offsets[d]=int32_t(token%128*kTokenBytes);destinations[d]=int32_t(owner*stride+slot);
            std::memcpy(expected.data()+uint64_t(owner*stride+slot)*kTokenBytes,
                source.data()+uint64_t(token%sourceRows)*kTokenBytes,kTokenBytes);
        }
        unsigned total=0;
        for(unsigned worker=0;worker<kFinalWorkers;++worker){
            std::vector<uint32_t> scratch(kFinalScratchWords,0xa5a5a5a5);
            LocalTensor<uint32_t> local{scratch.data(),scratch.size()*4};
            auto keys=local[2*kSlots],ranks=local[3*kSlots],slots=local[kFinalSlotsOffset],ctl=local[kFinalControlOffset];
            unsigned error=0;
            for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
                LoadFinalCopyPlan(&ctx,&row,b,count,keys.data,ranks.data,slots.data,&error);
            assert(!error);
            for(unsigned rank=0;rank<count;++rank){
                assert((uint64_t(keys.GetValue(rank))|(uint64_t(ranks.GetValue(rank))<<32))==fallback[b*kSlots+rank]);
                assert(slots.GetValue(rank)==uint32_t(destinations[b*kSlots+rank])-owner*stride);
            }
            assert(keys.GetValue(count)==0xa5a5a5a5 && ranks.GetValue(count)==0xa5a5a5a5 && slots.GetValue(count)==0xa5a5a5a5);
            copy_model::worker=worker;
            PublishFinalWorker(&ctx,&row,b,worker,0,count,b*kSlots+count*worker/kFinalWorkers,error,local,keys,ranks,slots,ctl);
            auto post=FinalPostAt(&ctx,b*kFinalWorkers+worker);
            assert(!post->error && !post->pending && post->head==post->count);total+=post->count;
            copy_model::Drained();
        }
        assert(total==count && std::memcmp(output.get(),expected.data(),outputBytes)==0);
    }
    // Invalid legacy descriptors fail closed before copy. Loading zero rows
    // may not inspect descriptors; the output and all event credits stay idle.
    ctx.fallbackAddresses=0;
    std::vector<uint32_t> scratch(kFinalScratchWords,0);
    LocalTensor<uint32_t> local{scratch.data(),scratch.size()*4};
    auto keys=local[2*kSlots],ranks=local[3*kSlots],slots=local[kFinalSlotsOffset],ctl=local[kFinalControlOffset];
    for(unsigned kind=0;kind<3;++kind){
        blocks[b*kSlots]=kind==0?-1:int32_t(b*(SIM_HISTORY/128));
        offsets[b*kSlots]=kind==1?1:kind==2?int32_t(128*kTokenBytes):0;
        unsigned error=0;threadIdx.x=0;
        LoadFinalCopyPlan(&ctx,&row,b,1,keys.data,ranks.data,slots.data,&error);
        assert(error==kUbmemInvalidAddress);
        auto loadCount=copy_model::loadCommands,storeCount=copy_model::storeCommands;
        PublishFinalWorker(&ctx,&row,b,0,1,1,b*kSlots,error,local,keys,ranks,slots,ctl);
        auto post=FinalPostAt(&ctx,b*kFinalWorkers);
        assert(post->error && !post->count && !post->head && !post->pending);
        assert(copy_model::loadCommands==loadCount && copy_model::storeCommands==storeCount);
        copy_model::Drained();
    }
}
