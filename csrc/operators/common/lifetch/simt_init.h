// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stddef.h>

namespace rotating {
// Control records are consumed only by later kernels. Give every word one
// SIMT writer, including packed count words shared by adjacent AIVs; scalar
// cache-line writeback must not publish these records after this function.
__simt_callee__ inline void InitControlWords(uint64_t address,uint32_t words) {
    auto output=reinterpret_cast<__gm__ uint32_t*>(address);
    for(uint32_t i=threadIdx.x;i<words;i+=256)asc_stcg(output+i,uint32_t(0));
}

template<class RowType,class ProgressType,class EventType,class PostType,class ContextType>
__simt_callee__ inline void InitializeSimtControl(__gm__ ContextType* ctx,uint32_t b,
    uint32_t batches,uint32_t rounds,uint32_t lanes,uint32_t workers,uint32_t postOffset,
    uint64_t transportStatus,uint32_t transportStatusWords,
    uint64_t first,uint64_t second,uint64_t addresses,uint64_t output,uint64_t initialHead,
    uint32_t queueLimit,uint32_t error) {
    static_assert(sizeof(RowType)==1024 && offsetof(RowType,limit)%8==0,
                  "SIMT init requires the existing aligned Row ABI");
    static_assert(offsetof(RowType,addresses)==16 && offsetof(RowType,output)==24 &&
                  offsetof(RowType,error)==44 && offsetof(RowType,prefetchLimit)==offsetof(RowType,limit)+4,
                  "SIMT init Row field packing changed");
    constexpr uint32_t limits=offsetof(RowType,limit)/8;
    auto row=reinterpret_cast<__gm__ uint64_t*>(ctx->rows)+uint64_t(b)*sizeof(RowType)/8;
    // Write final values directly, rather than zeroing and then patching the
    // same words from another thread. No intra-VF barrier or scalar Row flush
    // is needed; the unchanged init kernel boundary publishes every row.
    for(uint32_t i=threadIdx.x;i<sizeof(RowType)/8;i+=256) {
        uint64_t value=0;
        if(i==0)value=first;
        else if(i==1)value=second;
        else if(i==2)value=addresses;
        else if(i==3)value=output;
        else if(i==offsetof(RowType,error)/8)value=uint64_t(error)<<32;
        else if(i==limits)value=uint64_t(kSlots)|(uint64_t(ctx->prefetchLimit)<<32);
        else if(i==limits+1)value=queueLimit;
        else if(i==limits+2)value=initialHead;
        asc_stcg(row+i,value);
    }
    InitControlWords(ctx->progress+uint64_t(b)*4*sizeof(ProgressType),4*sizeof(ProgressType)/4);
    InitControlWords(ctx->owners+uint64_t(b)*128,32);
    InitControlWords(ctx->owners+uint64_t(batches)*128+uint64_t(b)*workers*kFinalStateWords*4,
                     workers*kFinalStateWords);
    InitControlWords(ctx->owners+postOffset+uint64_t(b)*workers*sizeof(PostType),workers*sizeof(PostType)/4);
    InitControlWords(ctx->rowStatus+uint64_t(b)*kRowStatusWords*4,kRowStatusWords);
    if(transportStatus)InitControlWords(transportStatus,transportStatusWords);
    for(uint32_t e=b;e<(rounds?rounds:1)*lanes;e+=batches)
        InitControlWords(ctx->events+uint64_t(e)*sizeof(EventType),sizeof(EventType)/4);
    if(!b && !threadIdx.x)asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->count),uint32_t(0));
    if(ctx->changeSlots) {
        auto source=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+uint64_t(b)*kLruSlots;
        auto destination=reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+uint64_t(b)*kChangeCapacity;
        for(uint32_t i=threadIdx.x;i<kLruSlots;i+=256)asc_stcg(destination+i,asc_ldcg(source+i));
    }
    asc_threadfence();
}
}
