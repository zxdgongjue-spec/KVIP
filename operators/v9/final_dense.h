#pragma once
namespace rotating {
// Demo FinalResolve, applied to bounded 8K map chunks. Only compacted cold
// TopK positions are revisited per chunk; hot entries never enter the loop.
__simd_vf__ inline void DenseChunk(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* positions,
    __ubuf__ uint32_t* required,__ubuf__ uint32_t* mapping,__ubuf__ uint32_t* tokens,
    __ubuf__ uint32_t* locations,__ubuf__ uint32_t* count,
    uint32_t base,uint32_t span,uint32_t coldCount,uint32_t slotBase) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    auto lane=CreateMask<uint32_t,MaskPattern::VL1>();
    RegTensor<uint32_t> zero,one,position,token,index,flags,old,rank,slots,written,total,broadcast,zeroIndex,baseVector;
    Duplicate(baseVector,base);Duplicate(zero,uint32_t(0));Duplicate(one,uint32_t(1));Duplicate(zeroIndex,uint32_t(0));
    for(uint16_t i=0;i<span;i+=64)StoreAlign(required+i,zero,all);
    LocalMemBar<MemType::VEC_STORE,MemType::VEC_STORE>();
    // Unique IDs within each TopK; barriers serialize identical cross-query
    // flag writes, as in the original demo.
    for(uint16_t q=0;q<4;++q) {
        uint32_t remaining=coldCount;
        for(uint16_t i=0;i<coldCount;i+=64) {
            auto valid=UpdateMask<uint32_t>(remaining);
            LoadAlign(position,positions+i);Gather(token,selected,position,valid);
            RegTensor<uint32_t> query;
            ShiftRights(query,position,int16_t(11),valid);
            Sub(index,token,baseVector,valid);
            MaskReg inside,owned;
            CompareScalar<uint32_t,CMPMODE::LT>(inside,index,span,valid);
            CompareScalar<uint32_t,CMPMODE::EQ>(owned,query,uint32_t(q),valid);
            MaskAnd(inside,inside,owned,valid);Scatter(required,one,index,inside);
        }
        LocalMemBar<MemType::VEC_STORE,MemType::VEC_STORE>();
    }
    LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
    LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(written,count);
    RegTensor<int32_t> tokenSequence;
    Arange(tokenSequence,int32_t(base));
    for(uint16_t i=0;i<span;i+=64) {
        LoadAlign(flags,required+i);LoadAlign(old,mapping+i);
        MaskReg take,missing;
        CompareScalar<uint32_t,CMPMODE::NE>(take,flags,uint32_t(0),all);
        CompareScalar<uint32_t,CMPMODE::EQ>(missing,old,uint32_t(-1),all);
        MaskAnd(take,take,missing,all);
        Unsqueeze(rank,take);Add(index,rank,written,all);Adds(slots,index,slotBase,all);
        Select(slots,slots,old,take);StoreAlign(mapping+i,slots,all);
        Scatter(tokens,reinterpret_cast<RegTensor<uint32_t>&>(tokenSequence),index,take);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),take);
        ReduceSum(total,flags,all);Gather(broadcast,total,zeroIndex);Add(written,written,broadcast,all);
        Adds(tokenSequence,tokenSequence,int32_t(64),all);
    }
    LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
    uint32_t remaining=coldCount;
    for(uint16_t i=0;i<coldCount;i+=64) {
        auto valid=UpdateMask<uint32_t>(remaining);
        LoadAlign(position,positions+i);Gather(token,selected,position,valid);
        Sub(index,token,baseVector,valid);
        MaskReg inside;CompareScalar<uint32_t,CMPMODE::LT>(inside,index,span,valid);
        Gather(slots,mapping,index,inside);Scatter(locations,slots,position,inside);
    }
    StoreAlign(count,written,lane);
}
// Collect arbitrary (not necessarily block-affine) remote addresses once.
// This also emits the descriptors needed by the small-SQ transport fallback.
template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void DenseRecords(__ubuf__ uint32_t* tokens,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* remoteLow,__ubuf__ uint32_t* remoteHigh,__ubuf__ uint32_t* control,
    __gm__ uint64_t* addresses,__gm__ int32_t* blocks,__gm__ int32_t* offsets,
    __gm__ int32_t* destinations,__gm__ uint32_t* total,__gm__ uint32_t* status,
    uint32_t count,uint32_t slotBase,uint32_t row) {
    const uint32_t tid=threadIdx.x;
    if(!tid)control[1]=asc_atomic_add(total,count);
    asc_syncthreads();
    for(uint32_t i=tid;i<count;i+=Threads) {
        const uint32_t token=tokens[i],slot=slotBase+i,dst=control[1]+i;
        const uint64_t remote=asc_ldcg(addresses+token);
        // remoteHigh aliases tokens; each lane reads its token before replacing it.
        remoteLow[i]=uint32_t(remote);remoteHigh[i]=uint32_t(remote>>32);slots[i]=slot;
        asc_stcg(blocks+dst,int32_t(row*(SIM_HISTORY/128)+token/128));
        asc_stcg(offsets+dst,int32_t((token%128)*kTokenBytes));
        asc_stcg(destinations+dst,int32_t(row*(SIM_RESIDENT+kSlots)+slot));
    }
    if(!tid) { asc_stcg(status,0U);asc_stcg(status+1,count);asc_stcg(status+2,control[1]); }
    asc_threadfence();
}
__aicore__ inline void DenseFinal(__gm__ Context* ctx,__gm__ Row* row,
    __gm__ uint32_t* status,LocalTensor<uint32_t> local,uint32_t coldCount,uint32_t b) {
    constexpr uint32_t seenWords=(SIM_HISTORY+31)/32;
    auto selected=local;auto locations=local[kSlots];
    auto required=local[2*kSlots];auto mapping=local[3*kSlots];auto tokens=local[4*kSlots];
    auto positions=local[5*kSlots+seenWords+64];auto control=local[5*kSlots+seenWords];
    control.SetValue(0,0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    GlobalTensor<uint32_t> map,locationsGm;
    map.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*SIM_HISTORY);
    locationsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots);
    if(coldCount)for(uint32_t base=SIM_RESIDENT;base<SIM_HISTORY;base+=kSlots) {
        const uint32_t span=SIM_HISTORY-base<kSlots ? SIM_HISTORY-base : kSlots;
        DataCopy(mapping,map[base],span);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        DenseChunk(reinterpret_cast<__ubuf__ uint32_t*>(selected.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(positions.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(required.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(mapping.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(tokens.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(locations.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(control.GetPhyAddr()),base,span,coldCount,SIM_RESIDENT+row->slots);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(map[base],mapping,span);
        // Next MTE2 load reuses the same UB map; a V-only wait does not protect it.
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(locationsGm,locations,kSlots);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t count=control.GetValue(0);
    if(count) {
        Simt::VF_CALL<DenseRecords<512>>(Simt::Dim3{512,1,1},
            reinterpret_cast<__ubuf__ uint32_t*>(tokens.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(mapping.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(required.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(tokens.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(control.GetPhyAddr()),
            reinterpret_cast<__gm__ uint64_t*>(row->addresses),
            reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),
            reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
            reinterpret_cast<__gm__ int32_t*>(ctx->destinations),
            reinterpret_cast<__gm__ uint32_t*>(ctx->count),status,count,SIM_RESIDENT+row->slots,b);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    auto trace=reinterpret_cast<__gm__ uint64_t*>(ctx->owners)+b*16;trace[2]=GetSystemCycle();
    SubmitFinal(row,mapping,selected,required,tokens,control[8],count);
    status[3]=count && !row->finalPending ? 1U : 0U;
}
}
