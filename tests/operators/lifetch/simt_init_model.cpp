// SPDX-License-Identifier: Apache-2.0
// Appended to the existing delayed-DMA model by test_simt_init.py.
#include <cstring>
#include <iostream>

uint64_t digest=1469598103934665603ULL;
void Hash(const void* data,size_t bytes) {
    auto p=static_cast<const unsigned char*>(data);
    for(size_t i=0;i<bytes;++i)digest=(digest^p[i])*1099511628211ULL;
}
int main() {
    constexpr uint32_t B=SIM_BATCH,H=SIM_HISTORY,R=SIM_RESIDENT,guard=32;
    constexpr uint32_t events=(kRounds?kRounds:1)*kServiceLanes;
    constexpr uint32_t ownerWords=(kFinalPostOffset+B*kFinalWorkers*sizeof(FinalPost))/4;
    std::vector<Row> rows(B+1);
    std::vector<uint32_t> progress(B*4*32+guard),owners(ownerWords+guard),status(B*32+guard);
    std::vector<uint32_t> event(events*32+guard),transportStatus(B*64+guard),count(guard+1);
    std::vector<uint32_t> map(B*H+guard),selected(B*kSlots+guard),changes(B*kChangeCapacity+guard);
    std::vector<uint32_t> slots(B*kChangeCapacity+guard),lru(B*kLruSlots),reserved(B*R+guard);
    std::vector<uint32_t> ends(B*4),bounds(B*2),changeCount(B+guard);
    Context ctx{};
    ctx.rows=uint64_t(rows.data());ctx.progress=uint64_t(progress.data());ctx.owners=uint64_t(owners.data());
    ctx.rowStatus=uint64_t(status.data());ctx.events=uint64_t(event.data());ctx.count=uint64_t(count.data());
    ctx.finalStatus=ctx.ubmemStatus=uint64_t(transportStatus.data());ctx.mapping=uint64_t(map.data());
    ctx.selected=uint64_t(selected.data());ctx.changes=uint64_t(changes.data());ctx.changeSlots=uint64_t(slots.data());
    ctx.lruSlots=uint64_t(lru.data());ctx.reservedHot=uint64_t(reserved.data());ctx.queryEnds=uint64_t(ends.data());
    ctx.tailBounds=uint64_t(bounds.data());ctx.changeCount=uint64_t(changeCount.data());ctx.outputStride=R+384;
    reservedBegin=reserved.data();reservedEnd=reserved.data()+B*R;
    mappingBegin=map.data();mappingEnd=map.data()+B*H;
    std::fill(map.begin(),map.end(),91);
    for(uint32_t round=0;round<8;++round) {
        std::fill(ends.begin(),ends.end(),0);
        for(auto* values:{&progress,&owners,&status,&event,&transportStatus,&count,&selected,&changes,&slots,&reserved,&changeCount})
            std::fill(values->begin(),values->end(),91);
        std::memset(rows.data()+B,0x5a,sizeof(Row));
        ctx.tailBounds=round==2?0:uint64_t(bounds.data());
        modelError=round==3?0x10000:round==4?0x20000:0;
        for(uint32_t j=0;j<B;++j) {
            // Reverse the row schedule too: adjacent changeCount words must
            // not depend on scalar cache-line writeback or row launch order.
            uint32_t b=round%2?B-1-j:j;
            uint64_t oldState=0;
#if V9_UBMEM_TAGGED_MAP
            oldState=rows[b].mappingState;
            if(round==0 || round==5)oldState=0;
            if(round==6)oldState=(uint64_t(kMappingEpochMagic)<<32)|kMappingEpochMax;
            if(round==7)oldState=(uint64_t(kMappingEpochMagic-1)<<32)|7;
#endif
            std::memset(&rows[b],0x5a,sizeof(Row));
#if V9_UBMEM_TAGGED_MAP
            rows[b].mappingState=oldState;
            auto generation=NextMappingGeneration(oldState);
#else
            MappingGeneration generation{0,0,true};
#endif
            for(uint32_t i=0;i<kLruSlots;++i)lru[b*kLruSlots+i]=(i+round*173+b*991)%R;
            const bool active=round!=1 && (b+round)%3!=0;
            if(active)ends[b*4+round%4]=H-1;
            bounds[b*2]=H-384;bounds[b*2+1]=round==4?H:H-128;
            uint32_t oldCalls=vfCalls;
            const auto values=GetInitRowValues(&ctx,b);
            Row expected{};
            auto first=reinterpret_cast<uint64_t*>(&expected);
            first[2]=values.addresses;first[3]=values.output;
            expected.limit=kSlots;expected.prefetchLimit=ctx.prefetchLimit;expected.error=modelError;
#if MODEL_URMA
            first[0]=0x102030405060ULL+b;first[1]=0x998877665544ULL+b;
            expected.initialHead=uint64_t(0xffffffffU)+round;
            expected.queueLimit=511;
            auto reservedRow=InitializeRound(B,H,R,kRounds,kServiceLanes,kFinalWorkers,kFinalPostOffset,&ctx,b);
#if V9_SIMT_INIT
            Simt::VF_CALL<InitializeRoundSimt>(Simt::Dim3{256,1,1},B,H,R,kRounds,kServiceLanes,
                kFinalWorkers,kFinalPostOffset,&ctx,b,reservedRow,first[0],first[1],values.addresses,
                values.output,expected.initialHead,expected.queueLimit,modelError);
#else
            Simt::VF_CALL<InitializeTailMappings>(Simt::Dim3{256,1,1},H,R,&ctx,b,reservedRow);
            // The original entry fills scalar Row fields after bulk init.
            rows[b]=expected;
#endif
#else
            expected.mappingState=generation.state;expected.copyLimit=kLruSlots;
            InitializeRound(&ctx,b);
#if !V9_SIMT_INIT
            rows[b].reserved1=rows[b].reserved2=0;rows[b].addresses=values.addresses;rows[b].output=values.output;
            rows[b].limit=kSlots;rows[b].prefetchLimit=ctx.prefetchLimit;rows[b].copyLimit=kLruSlots;rows[b].error=modelError;
#endif
#endif
            assert(vfCalls==oldCalls+1);
            assert(std::memcmp(&rows[b],&expected,sizeof(Row))==0);
            assert(changeCount[b]==kLruSlots);
            for(uint32_t i=0;i<H;++i) {
                uint32_t wanted=UINT32_MAX;
                if(ctx.tailBounds && i>=bounds[b*2] && i<bounds[b*2+1])wanted=R+kSlots+i%384;
                assert(DecodeMappingSlot(map[b*H+i],generation.tag,V9_UBMEM_TAGGED_MAP?kMappingSlotMask:UINT32_MAX)==wanted);
            }
            for(uint32_t i=0;i<kLruSlots;++i)assert(slots[b*kChangeCapacity+i]==lru[b*kLruSlots+i]);
            for(uint32_t i=kLruSlots;i<kChangeCapacity;++i)assert(slots[b*kChangeCapacity+i]==91);
            for(uint32_t i=0;i<R;++i) {
                uint32_t wanted=91;
                if(active) {
                    uint32_t ordinal=(i+R-(round*173+b*991)%R)%R;
                    wanted=ordinal<kLruSlots?ordinal:UINT32_MAX;
                }
                assert(reserved[b*R+i]==wanted);
            }
        }
        for(auto* values:{&progress,&owners,&status,&event,&count}) {
            uint32_t valid=uint32_t(values->size())-guard;
            assert(std::all_of(values->begin(),values->begin()+valid,[](uint32_t x){return x==0;}));
        }
        const uint32_t transportWords=MODEL_URMA?B*64:B*32;
        assert(std::all_of(transportStatus.begin(),transportStatus.begin()+transportWords,[](uint32_t x){return x==0;}));
        assert(std::all_of(transportStatus.begin()+transportWords,transportStatus.end(),[](uint32_t x){return x==91;}));
        for(auto* values:{&progress,&owners,&status,&event,&transportStatus,&count,&selected,&changes,&slots,&reserved,&changeCount}) {
            assert(std::all_of(values->end()-guard,values->end(),[](uint32_t x){return x==91;}));
            Hash(values->data(),values->size()*4);
        }
        for(uint32_t i=0;i<B*kChangeCapacity;++i)assert(changes[i]==UINT32_MAX);
        for(uint32_t i=0;i<B*kSlots;++i)assert(selected[i]==(MODEL_URMA?91:UINT32_MAX));
        assert(std::all_of(map.end()-guard,map.end(),[](uint32_t x){return x==91;}));
        for(uint32_t i=0;i<sizeof(Row);++i)assert(reinterpret_cast<unsigned char*>(rows.data()+B)[i]==0x5a);
        Hash(rows.data(),rows.size()*sizeof(Row));Hash(map.data(),map.size()*4);
    }
    std::cout<<digest<<' '<<copies<<' '<<loads<<' '<<zeroFills<<' '<<vfCalls<<'\n';
}
