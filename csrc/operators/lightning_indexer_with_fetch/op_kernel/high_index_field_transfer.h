// SPDX-License-Identifier: Apache-2.0
#pragma once

// Each request has four vector cores. For a long permutation a core owns
// four adjacent key words in every tuple, and at most one companion column.
// Its private buffer therefore contains all originals of the fields it can overwrite.
template<bool Scatter>
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void TransferOwnedFields(__gm__ Edge* edges, uint32_t p0, uint32_t p1, uint32_t singles,
        uint32_t lane, __gm__ uint64_t* keys, __gm__ uint32_t* scales, __gm__ uint64_t* addr,
        __gm__ uint64_t* logical, __ubuf__ uint64_t* saved) {
    if constexpr (!Scatter) TransferPairSlice(edges,p0,p1,keys,scales,addr,logical);
    const uint32_t field=threadIdx.x%4;
    auto metadata=saved+uint64_t(singles)*4;
    for (uint32_t i=threadIdx.x/4; i<singles; i+=64) {
        auto edge=edges+kCapacity-1-i;
        uint32_t pos=Scatter ? asc_ldcg(&edge->dst) : asc_ldcg(&edge->src);
        auto key=keys+uint64_t(pos)*16+4*lane+field;
        auto word=saved+uint64_t(i)*4+field;
        if constexpr (Scatter) asc_stcg(key,*word);
        else *word=asc_ldcg(key);

        if (!field && lane<2) {
            auto data=(lane==0 ? addr : logical)+pos;
            if constexpr (Scatter) asc_stcg(data,metadata[i]);
            else metadata[i]=asc_ldcg(data);
        } else if (!field && lane==2) {
            auto values=reinterpret_cast<__ubuf__ uint32_t*>(metadata);
            if constexpr (Scatter) asc_stcg(scales+pos,values[i]);
            else values[i]=asc_ldcg(scales+pos);
        }
    }
    asc_threadfence();
}

__aicore__ inline void ApplyOwnedFields(__gm__ Args* a,const Segment& part,uint32_t singles,
        uint32_t lane,uint64_t stride,__gm__ uint64_t* ticks,bool profile) {
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
    Simt::VF_CALL<TransferOwnedFields<false>>(Simt::Dim3{256,1,1},edges,part.p0,part.p1,singles,lane,
        keys,scales,addr,logical,saved);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    if (profile) { ticks[12]=GetSystemCycle(); ticks[13]=bytes; }
    Simt::VF_CALL<TransferOwnedFields<true>>(Simt::Dim3{256,1,1},edges,part.p0,part.p1,singles,lane,
        keys,scales,addr,logical,saved);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    pipe.Destroy();
}
