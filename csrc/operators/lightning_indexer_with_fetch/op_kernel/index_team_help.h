// SPDX-License-Identifier: Apache-2.0
#pragma once

// One ticket lives on the final queue owner's scalar stack, across its drain.
// A published suffix may be claimed by another ready AIV during that drain.
struct TailTicket {
    uint32_t row=0, chunks=0, nextChunk=0, firstMoves=0, moves=0;
};
struct ReadyFourth {
    uint32_t row, chunks, moves, claim, remaining;
};

__aicore__ inline uint32_t ReadClaim(__gm__ uint32_t* claim) {
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(claim),sizeof(uint32_t));
    GlobalTensor<uint32_t> word;
    word.SetGlobalBuffer(claim,1);
    return word.GetValue(0);
}
__aicore__ inline void ReleaseClaim(__gm__ uint32_t* claim,uint32_t state) {
    GlobalTensor<uint32_t> word;
    word.SetGlobalBuffer(claim,1);
    word.SetValue(0,state);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(claim),sizeof(uint32_t));
}

__aicore__ inline uint32_t Cursor(uint32_t state) {
    return state&kFourthSuffixReady ? state&~kFourthSuffixReady : 0U;
}

__aicore__ inline ReadyFourth ProbeFourth(__gm__ Args* a,uint32_t row,bool prefixOnly) {
    ReadyFourth result{row,0,0,0,0};
    auto control=Control(a,row);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(control),sizeof(*control));
    if(!control->ready || control->error || !control->count || control->count>=kIndexEdgesOffset)return result;
    auto edges=Moves(a,row);
    uint32_t state=ReadClaim(edges+kClaimOffset);
    if(state && (prefixOnly || !(state&kFourthSuffixReady)))return result;
    auto q4=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+row*kFlagCount+4;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(q4),sizeof(*q4));
    if(!q4->ready || q4->error)return result;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(edges),kIndexEdgesOffset*sizeof(uint32_t));
    uint32_t chunks=control->count,first=Cursor(state);
    if(first>=chunks)return result;
    uint32_t moves=edges[chunks-1];
    uint32_t copied=first?edges[first-1]:0U;
    if(copied>=moves)return result;
    return {row,chunks,moves,state,moves-copied};
}

__aicore__ inline ReadyFourth PickFourth(__gm__ Args* a,uint32_t first,uint32_t second,bool prefixOnly) {
    auto x=ProbeFourth(a,first,prefixOnly);
    auto y=ProbeFourth(a,second,prefixOnly);
    return y.remaining>x.remaining?y:x;
}

// One lane gathers four key words for each of its moves before any lane may
// scatter. All old values stay in registers. KVAF's live UB and event-managed
// SQ posting buffers are untouched. A later planner chunk may read this
// chunk's writes, so never reorder chunks, even for the fourth field alone.
template<bool Elect>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void CopyFourthRange(
        __gm__ uint32_t* edges,__gm__ uint64_t* keys,uint32_t firstChunk,uint32_t endChunk,
        __gm__ uint32_t* claim,uint32_t expected,uint32_t owned) {
    const uint32_t tid=threadIdx.x;
    if constexpr(Elect){
        if(!tid)asc_atomic_cas(claim,expected,owned);
        asc_threadfence();asc_syncthreads();
        if(asc_ldcg(claim)!=owned)return;
    }
    uint32_t first=firstChunk?asc_ldcg(edges+firstChunk-1):0;
    constexpr uint32_t width=256,perThread=(kMaxMoves+width-1)/width;
    for(uint32_t c=firstChunk;c<endChunk;++c){
        uint32_t end=asc_ldcg(edges+c);
        uint32_t dest[perThread];
        uint64_t saved[perThread][4];
#pragma unroll
        for(uint32_t j=0;j<perThread;++j){
            uint32_t i=first+tid+j*width;
            if(i<end){
                uint32_t src=asc_ldcg(edges+kIndexEdgesOffset+2*i);
                dest[j]=asc_ldcg(edges+kIndexEdgesOffset+2*i+1)&0x7fffffffU;
#pragma unroll
                for(uint32_t w=0;w<4;++w)saved[j][w]=asc_ldcg(keys+uint64_t(src)*16+12+w);
            }
        }
        asc_syncthreads();
#pragma unroll
        for(uint32_t j=0;j<perThread;++j)if(first+tid+j*width<end){
#pragma unroll
            for(uint32_t w=0;w<4;++w)asc_stcg(keys+uint64_t(dest[j])*16+12+w,saved[j][w]);
        }
        asc_threadfence();asc_syncthreads();first=end;
    }
}

template<bool Elect>
__aicore__ inline void RunFourthRange(__gm__ Args* a,uint32_t row,uint32_t first,uint32_t end,
                                      uint32_t expected,uint32_t owned) {
    auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
    auto owners=reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners);
    uint32_t owner=owners?owners[row]:row;
    auto keys=reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*a->history*16;
    auto edges=Moves(a,row);
    Simt::VF_CALL<CopyFourthRange<Elect>>(Simt::Dim3{256,1,1},edges,keys,first,end,
                                          edges+kClaimOffset,expected,owned);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
}

__aicore__ inline void CopyOwnedFourth(__gm__ Args* a,uint32_t row,uint32_t first,uint32_t chunks) {
    RunFourthRange<false>(a,row,first,chunks,0,0);
}
__aicore__ inline void FinishFourth(__gm__ Args* a,uint32_t block,uint32_t batch,
                                    uint32_t row,uint32_t moves,uint64_t begin) {
    if(a->prefetchCount&0x80000000U){
        auto trace=reinterpret_cast<__gm__ uint64_t*>(a->times)+batch*256+4*batch*32;
        TraceField(trace,batch,row,3,block,moves,begin,begin,GetSystemCycle(),0);
    }
    Publish(Control(a,row)+4,0,0);
}

// Called exactly once for a final queue after all its Posts. No waiting on
// another owner. One successful probe executes at most one planner chunk.
__aicore__ __attribute__((noinline)) inline bool TryBeforeCq(
        __gm__ Args* a,uint32_t block,uint32_t batch,TailTicket& ticket) {
    const uint32_t queue=block-3*batch;
    if(((a->prefetchCount>>16)&0xffU)!=2 || batch!=16 || queue>=8)return false;
    // Own Q, same row parity. The eight pairs partition all sixteen rows.
    const uint32_t first=4*(queue/2)+queue%2;
    auto target=PickFourth(a,first,first+2,true);
    if(!target.moves)return false;
    auto edges=Moves(a,target.row);
    uint32_t c=0;
    while(c<target.chunks && !edges[c])++c;
    if(c==target.chunks)return false;
    const uint32_t busy=kFourthPrefixBusy|(block+1);
    uint64_t begin=(a->prefetchCount&0x80000000U)?GetSystemCycle():0;
    RunFourthRange<true>(a,target.row,c,c+1,0,busy);
    auto claim=edges+kClaimOffset;
    if(ReadClaim(claim)!=busy)return false;
    uint32_t firstMoves=edges[c];
    ticket={target.row,target.chunks,c+1,firstMoves,target.moves};
    // Later chunks can be empty. If all moves are complete, publish now.
    if(firstMoves==target.moves)ticket.nextChunk=target.chunks;
    ReleaseClaim(claim,ticket.nextChunk==ticket.chunks?block+1:(kFourthSuffixReady|ticket.nextChunk));
    if(ticket.nextChunk==ticket.chunks)FinishFourth(a,block,batch,target.row,target.moves,begin);
    return true;
}

__aicore__ __attribute__((noinline)) inline bool FinishAfterCq(
        __gm__ Args* a,uint32_t block,uint32_t batch,const TailTicket& ticket) {
    if(ticket.nextChunk>=ticket.chunks)return false;
    uint64_t begin=(a->prefetchCount&0x80000000U)?GetSystemCycle():0;
    auto claim=Moves(a,ticket.row)+kClaimOffset;
    uint32_t expected=kFourthSuffixReady|ticket.nextChunk;
    if(ReadClaim(claim)!=expected)return false;
    RunFourthRange<true>(a,ticket.row,ticket.nextChunk,ticket.chunks,expected,block+1);
    if(ReadClaim(claim)!=block+1)return false;
    FinishFourth(a,block,batch,ticket.row,ticket.moves,begin);
    return true;
}

// Called only after the owner has discharged its own queue or row duties.
// The 48 row workers probe one candidate each immediately before retirement;
// the other 16 can probe two. No worker waits for a remote plan or for a win.
__aicore__ __attribute__((noinline)) inline void HelpAfterLocalWork(__gm__ Args* a,uint32_t block,uint32_t batch) {
    if(((a->prefetchCount>>16)&0xffU)!=2 || batch!=16)return;
    ReadyFourth target{};
    if(block<3*batch){
        const uint32_t row=(block/3+1+5*(block%3))%batch;
        auto done=Control(a,row)+4;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(done),sizeof(*done));
        if(done->ready)return;
        target=ProbeFourth(a,row,false);
    }else{
        const uint32_t lane=block-3*batch;
        const uint32_t first=lane<8?(lane+1)%batch:lane-8;
        target=PickFourth(a,first,(first+1)%batch,false);
    }
    if(!target.moves)return;
    auto claim=Moves(a,target.row)+kClaimOffset;
    uint32_t cursor=Cursor(target.claim);
    uint64_t begin=(a->prefetchCount&0x80000000U)?GetSystemCycle():0;
    RunFourthRange<true>(a,target.row,cursor,target.chunks,target.claim,block+1);
    if(ReadClaim(claim)==block+1)FinishFourth(a,block,batch,target.row,target.moves,begin);
}
