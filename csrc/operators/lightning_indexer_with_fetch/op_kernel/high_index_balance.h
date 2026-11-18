// SPDX-License-Identifier: Apache-2.0
#pragma once

// Included inside high_post_index. Resolve/Post retain their queue ownership;
// Index work uses the actual number of tuples in each permitted request.
template<bool Scatter>
__aicore__ inline void TransferSegments(__gm__ Args* a, const Segment* segments, uint32_t n,
        uint32_t history, uint64_t stride, bool useSpill, __ubuf__ Tuple* local) {
    for (uint32_t i=0; i<n; ++i) {
        const Segment& t=segments[i];
        if constexpr (Scatter) { if (t.s0==t.s1) continue; }
        auto edges=reinterpret_cast<__gm__ Edge*>(a->highIndex+uint64_t(t.row)*kRowBytes);
        auto payload=reinterpret_cast<__gm__ Tuple*>(edges+kCapacity);
        auto keys=reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(t.owner)*history*16;
        auto scales=reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(t.owner)*history;
        auto addr=reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(t.owner)*stride;
        auto logical=reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(t.owner)*history;
        if (useSpill) {
            Simt::VF_CALL<Transfer<Scatter,true>>(Simt::Dim3{256,1,1},edges,t.p0,t.p1,t.s0,t.s1,
                keys,scales,addr,logical,payload,local);
        } else {
            Simt::VF_CALL<Transfer<Scatter,false>>(Simt::Dim3{256,1,1},edges,t.p0,t.p1,t.s0,t.s1,
                keys,scales,addr,logical,payload,local+t.localOffset);
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
}

__aicore__ inline void ApplyGlobalIndex(__gm__ Args* a, __gm__ Context* ctx, uint32_t b,
        uint32_t block, uint32_t batch, uint32_t cores, uint32_t lane, uint32_t lanes,
        __gm__ uint64_t* update, __gm__ uint64_t* ticks, bool profile) {
    TPipe pipe; TBuf<TPosition::VECCALC> headerBuf;
    pipe.InitBuffer(headerBuf,16*128);
    auto h=headerBuf.Get<uint32_t>();
    GlobalTensor<uint32_t> table;
    table.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->workerStatus));
    DataCopy(h,table,batch*32);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    uint32_t total=0, totalLong=0, counts[16], pairs[16];
    for (uint32_t row=0; row<batch; ++row) {
        bool ok=h.GetValue(row*32)==0;
        pairs[row]=ok ? h.GetValue(row*32+3) : 0;
        uint32_t longCount=ok ? h.GetValue(row*32+4) : 0;
        counts[row]=2*pairs[row]+longCount;
        totalLong+=longCount;
        total+=counts[row];
    }
    uint32_t start=total*block/cores,end=total*(block+1)/cores;
    uint32_t base=0, n=0, localCount=0, pairCount=0;
    Segment segments[16];
    for (uint32_t row=0; row<batch; ++row) {
        uint32_t rowEnd=base+counts[row];
        if (start<rowEnd && end>base && counts[row]) {
            auto& work=segments[n];
            if (PartitionRow(start,end,base,counts[row],pairs[row],work)) {
                work.row=row; work.owner=h.GetValue(row*32+6); work.localOffset=localCount;
                localCount+=work.s1-work.s0; pairCount+=work.p1-work.p0; ++n;
            }
        }
        base=rowEnd;
    }
    auto ownEdges=reinterpret_cast<__gm__ Edge*>(a->highIndex+uint64_t(b)*kRowBytes);
    auto ownSpill=reinterpret_cast<__gm__ Tuple*>(ownEdges+kCapacity);
    uint32_t invalid=h.GetValue(b*32)==0 ? h.GetValue(b*32+5) : 0;
    uint32_t i0=invalid*lane/lanes,i1=invalid*(lane+1)/lanes;
    if(i1>i0) {
        uint32_t owner=h.GetValue(b*32+6);
        auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*a->resident;
        Simt::VF_CALL<CommitInvalidations>(Simt::Dim3{256,1,1},&ownSpill[0].padding,i0,i1,valid);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    uint64_t stride=update[7];
    bool useSpill=localCount>kLocalTuples;
    TBuf<TPosition::VECCALC> payloadBuf;
    auto local=reinterpret_cast<__ubuf__ Tuple*>(h.GetPhyAddr());
    if (localCount && !useSpill) {
        pipe.InitBuffer(payloadBuf, ((localCount*sizeof(Tuple)+31)/32)*32);
        local=reinterpret_cast<__ubuf__ Tuple*>(payloadBuf.Get<uint32_t>().GetPhyAddr());
    }
    TransferSegments<false>(a,segments,n,a->history,stride,useSpill,local);
    // Uniform across all AIVs. Pure disjoint swaps have no staged sources to protect.
    if (totalLong) SyncAll();
    if (profile) ticks[12]=GetSystemCycle();
    if (localCount) TransferSegments<true>(a,segments,n,a->history,stride,useSpill,local);
    if (profile) {
        ticks[2]=0x50494831U; ticks[5]=GetSystemCycle();
        ticks[9]=2*pairCount+localCount; ticks[10]=ticks[9];
        ticks[16]=pairCount; ticks[17]=localCount;
        ticks[18]=useSpill ? localCount : 0; ticks[19]=i1-i0;
    }
    pipe.Destroy();
}
