// SPDX-License-Identifier: Apache-2.0
#pragma once

// Each request has four vector cores. For a long permutation a core owns
// four adjacent key words in every tuple, and at most one companion column.
// Its private buffer therefore contains all originals of the fields it can overwrite.
template<bool Scatter, uint32_t Width>
__simt_callee__ inline void CopyOwnedFieldGroup(__gm__ Edge* edges, uint32_t first, uint32_t singles,
        uint32_t lane, uint32_t field, __gm__ uint64_t* keys, __gm__ uint32_t* scales,
        __gm__ uint64_t* addr, __gm__ uint64_t* logical, __ubuf__ uint64_t* saved) {
    uint32_t pos[Width];
    uint64_t keysWord[Width];
    auto metadata=saved+uint64_t(singles)*4;
#pragma unroll
    for (uint32_t j=0;j<Width;++j) {
        auto edge=edges+kCapacity-1-first-j*64;
        pos[j]=Scatter ? asc_ldcg(&edge->dst) : asc_ldcg(&edge->src);
    }
#pragma unroll
    for (uint32_t j=0;j<Width;++j) {
        if constexpr (Scatter) keysWord[j]=saved[uint64_t(first+j*64)*4+field];
        else keysWord[j]=asc_ldcg(keys+uint64_t(pos[j])*16+4*lane+field);
    }
    if (!field && lane<2) {
        auto data=(lane==0 ? addr : logical);
        uint64_t values[Width];
#pragma unroll
        for (uint32_t j=0;j<Width;++j) {
            if constexpr (Scatter) values[j]=metadata[first+j*64];
            else values[j]=asc_ldcg(data+pos[j]);
        }
#pragma unroll
        for (uint32_t j=0;j<Width;++j) {
            if constexpr (Scatter) asc_stcg(data+pos[j],values[j]);
            else metadata[first+j*64]=values[j];
        }
    } else if (!field && lane==2) {
        auto scalesUB=reinterpret_cast<__ubuf__ uint32_t*>(metadata);
        uint32_t values[Width];
#pragma unroll
        for (uint32_t j=0;j<Width;++j) {
            if constexpr (Scatter) values[j]=scalesUB[first+j*64];
            else values[j]=asc_ldcg(scales+pos[j]);
        }
#pragma unroll
        for (uint32_t j=0;j<Width;++j) {
            if constexpr (Scatter) asc_stcg(scales+pos[j],values[j]);
            else scalesUB[first+j*64]=values[j];
        }
    }
#pragma unroll
    for (uint32_t j=0;j<Width;++j) {
        if constexpr (Scatter) asc_stcg(keys+uint64_t(pos[j])*16+4*lane+field,keysWord[j]);
        else saved[uint64_t(first+j*64)*4+field]=keysWord[j];
    }
}

template<bool Scatter>
__simt_callee__ inline void TransferOwnedPhase(__gm__ Edge* edges, uint32_t p0, uint32_t p1, uint32_t singles,
        uint32_t lane, __gm__ uint64_t* keys, __gm__ uint32_t* scales, __gm__ uint64_t* addr,
        __gm__ uint64_t* logical, __ubuf__ uint64_t* saved) {
    if constexpr (!Scatter) TransferPairSlice(edges,p0,p1,keys,scales,addr,logical);
    const uint32_t field=threadIdx.x%4;
    uint32_t i=threadIdx.x/4;
    for(;i+64<singles;i+=128)
        CopyOwnedFieldGroup<Scatter,2>(edges,i,singles,lane,field,keys,scales,addr,logical,saved);
    if(i<singles) CopyOwnedFieldGroup<Scatter,1>(edges,i,singles,lane,field,keys,scales,addr,logical,saved);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void TransferOwnedFields(__gm__ Edge* edges, uint32_t p0, uint32_t p1, uint32_t singles,
        uint32_t lane, __gm__ uint64_t* keys, __gm__ uint32_t* scales, __gm__ uint64_t* addr,
        __gm__ uint64_t* logical, __ubuf__ uint64_t* saved, __gm__ uint32_t* valid,
        uint32_t invalidFirst, uint32_t invalidEnd) {
    TransferOwnedPhase<false>(edges,p0,p1,singles,lane,keys,scales,addr,logical,saved);
    // A destination can be another thread's source. All 256 threads, even
    // those without an edge, complete their own gather before any scatter.
    asc_threadfence();
    asc_syncthreads();
    TransferOwnedPhase<true>(edges,p0,p1,singles,lane,keys,scales,addr,logical,saved);
    // The log lives in GM padding; this field-owned path uses private UB for
    // all tuple payloads. Successful hot publications join all Done flags.
    auto pending=reinterpret_cast<__gm__ Tuple*>(edges+kCapacity);
    for(uint32_t i=invalidFirst+threadIdx.x;i<invalidEnd;i+=256)
        asc_stcg(valid+asc_ldcg(&pending[i].padding),uint32_t(0));
    asc_threadfence();
}

__aicore__ inline void ApplyOwnedFields(__gm__ Args* a,const Segment& part,uint32_t singles,
        uint32_t lane,uint64_t stride,__gm__ uint64_t* ticks,bool profile,
        __gm__ uint32_t* valid,uint32_t invalidFirst,uint32_t invalidEnd) {
    auto edges=reinterpret_cast<__gm__ Edge*>(a->highIndex+uint64_t(part.row)*kRowBytes);
    auto keys=reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(part.owner)*a->history*16;
    auto scales=reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(part.owner)*a->history;
    auto addr=reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(part.owner)*stride;
    auto logical=reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(part.owner)*a->history;
    uint32_t bytes=singles*(lane<2 ? 40U : lane==2 ? 36U : 32U);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf,(bytes+31U)&~31U);
    auto saved=reinterpret_cast<__ubuf__ uint64_t*>(buf.Get<uint32_t>().GetPhyAddr());
    Simt::VF_CALL<TransferOwnedFields>(Simt::Dim3{256,1,1},edges,part.p0,part.p1,singles,lane,
        keys,scales,addr,logical,saved,valid,invalidFirst,invalidEnd);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    // There is no separately timestamped scalar gather endpoint in this dispatch.
    if (profile) { ticks[12]=0; ticks[13]=bytes; }
    pipe.Destroy();
}
