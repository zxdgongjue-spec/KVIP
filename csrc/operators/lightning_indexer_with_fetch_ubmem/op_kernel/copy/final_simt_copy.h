// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../device_context.h"

namespace rotating {
// cfg[0..3] remains the resolver's count/error/rank range. These four words
// are otherwise dead when the final plan has finished. A zero mode requests
// the unchanged MTE path; mode one proves every owned payload store completed.
constexpr uint32_t kSimtCopyScheduled=4,kSimtCopyCompleted=5,kSimtCopyReady=6,kSimtCopyFallback=7;

template<uint32_t Threads>
__simt_callee__ inline void CopyFinalPayloadSimt(__gm__ Context* ctx,uint32_t batch,uint32_t worker,
    uint32_t count,__ubuf__ uint32_t* remoteLow,__ubuf__ uint32_t* remoteHigh,
    __ubuf__ uint32_t* slots,__ubuf__ uint32_t* cfg) {
    static_assert(Threads>=32 && Threads%32==0,"Final SIMT copy requires complete warps");
    static_assert(kTokenBytes==656,"Final SIMT copy is a single-layer specialization");
    const uint32_t tid=threadIdx.x,lane=tid%32,warp=tid/32;
    if(!tid)for(uint32_t i=kSimtCopyScheduled;i<=kSimtCopyFallback;++i)cfg[i]=0;
    // The caller has finished all writers of the address/slot vectors. This
    // join also publishes the leader's reset before any worker adds counts.
    asc_syncthreads();
    auto region=reinterpret_cast<__gm__ UbmemRegion*>(asc_ldcg(&ctx->ubmemRegion));
    uint64_t peerBase=0,mappedBase=0,bytes=0,outputBase=0,outputBytes=0;
    bool valid=region!=nullptr;
    if(valid) {
        peerBase=asc_ldcg(&region->peerBase);mappedBase=asc_ldcg(&region->mappedBase);
        bytes=asc_ldcg(&region->bytes);outputBase=asc_ldcg(&region->outputBase);
        outputBytes=asc_ldcg(&region->outputBytes);
        valid=asc_ldcg(&region->magic)==kUbmemMagic && asc_ldcg(&region->version)==kUbmemVersion &&
            asc_ldcg(&region->locality)==kUbmemHostLocal && asc_ldcg(&region->generation)!=0 &&
            peerBase && mappedBase && outputBase && bytes>=kTokenBytes &&
            bytes<=UINT64_MAX-peerBase && bytes<=UINT64_MAX-mappedBase &&
            outputBytes>=uint64_t(SIM_RESIDENT)*kTokenBytes && outputBytes<=UINT64_MAX-outputBase &&
            !(outputBase&31);
    }
    auto row=reinterpret_cast<__gm__ Row*>(asc_ldcg(&ctx->rows))+batch;
    const uint64_t output=asc_ldcg(&row->output);
    uint32_t scheduled=0,error=valid?0:kUbmemInvalidRegion,fallback=0;
    for(uint32_t rank=tid;rank<count;rank+=Threads) {
        const uint32_t slot=slots[rank];
        // Both halves of every 32B sector stay on the same AIV, regardless
        // of the resolver/LRU order. No round-up or adjacent payload access.
        if((slot>>1)%kFinalWorkers!=worker)continue;
        ++scheduled;
        if(!valid)continue;
        const uint64_t peer=uint64_t(remoteLow[rank])|(uint64_t(remoteHigh[rank])<<32);
        const uint64_t sourceOffset=peer-peerBase,offset=output-outputBase;
        const uint64_t rowOffset=uint64_t(slot)*kTokenBytes;
        if(slot>=SIM_RESIDENT || sourceOffset>bytes-kTokenBytes || output<outputBase ||
           offset>outputBytes || rowOffset>outputBytes-offset ||
           kTokenBytes>outputBytes-offset-rowOffset) {
            error|=kUbmemInvalidAddress;continue;
        }
        // Production page rows are word-aligned. Preserve the original
        // arbitrary-byte input contract by delegating the entire AIV to MTE
        // before issuing any payload load/store if one row is unaligned.
        fallback|=uint32_t(((mappedBase+sourceOffset)|(output+rowOffset))&3)!=0;
    }
    if(scheduled)asc_atomic_add(cfg+kSimtCopyScheduled,scheduled);
    if(error)asc_atomic_or(cfg+1,error);
    if(fallback)asc_atomic_or(cfg+kSimtCopyFallback,fallback);
    asc_syncthreads();
    if(cfg[1] || cfg[kSimtCopyFallback])return;
    // One warp copies one row: 164 words = five complete 32-word passes and
    // four final words. Sixteen rows per AIV can make progress independently,
    // without a scalar token loop, UB payload staging or MTE event credits.
    for(uint32_t rank=warp;rank<count;rank+=Threads/32) {
        const uint32_t slot=slots[rank];
        if((slot>>1)%kFinalWorkers!=worker)continue;
        const uint64_t peer=uint64_t(remoteLow[rank])|(uint64_t(remoteHigh[rank])<<32);
        auto source=reinterpret_cast<__gm__ uint32_t*>(mappedBase+(peer-peerBase));
        auto destination=reinterpret_cast<__gm__ uint32_t*>(output+uint64_t(slot)*kTokenBytes);
        for(uint32_t word=lane;word<kTokenBytes/4;word+=32)
            asc_stcg(destination+word,asc_ldcg(source+word));
    }
    // Every lane releases its GM writes before the leader publishes the
    // completed count; the caller retains the original VF->scalar V_S join.
    asc_threadfence();
    asc_syncthreads();
    if(!tid){cfg[kSimtCopyCompleted]=cfg[kSimtCopyScheduled];cfg[kSimtCopyReady]=1;}
}
}
