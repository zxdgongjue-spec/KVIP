// Experimental reusable AIV-only selection module. Not a replacement for
// streaming final classification/publication. All local ranges must have >=K.
// Component experiment: exact main TopK, independent AIV-only worker grid.
// QK/scoring, resident threshold and transport are outside this measurement.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "indexer_types.h"
#include "topk.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_functions.h"
using namespace AscendC;

constexpr uint32_t K = 2048, Trunk = 16384;
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void Remap(__ubuf__ uint32_t* positions, __gm__ uint32_t* ids) {
    uint32_t tid=Simt::GetThreadIdx<0>();
    for(uint32_t i=tid;i<K;i+=256) positions[i]=ids[positions[i]];
}

class TopKWorker {
    TPipe pipe;
    TBuf<TPosition::VECCALC> valueBuf,indexBuf,scoreBuf,sharedBuf,tickBuf;
    LocalTensor<uint16_t> values,scores;
    LocalTensor<uint32_t> indices,shared;
    topk::LITopk<uint16_t> op;
public:
    __aicore__ inline void Init() {
        op.Init(K,Trunk);
        pipe.InitBuffer(valueBuf,(K+Trunk)*2);
        pipe.InitBuffer(indexBuf,(K+64)*4);
        pipe.InitBuffer(scoreBuf,K*2);
        pipe.InitBuffer(sharedBuf,op.GetSharedTmpBufferSize());
        pipe.InitBuffer(tickBuf,32);
        values=valueBuf.Get<uint16_t>(); indices=indexBuf.Get<uint32_t>();
        scores=scoreBuf.Get<uint16_t>();shared=sharedBuf.Get<uint32_t>();
        op.InitBuffers(shared,indices);
    }
    __aicore__ inline void Run(__gm__ uint16_t* src,uint32_t length,
                              __gm__ uint32_t* dstIdx,__gm__ uint16_t* dstScores,
                              uint32_t offset,__gm__ uint32_t* idMap=nullptr,
                              uint32_t producers=0,uint32_t history=0,
                              uint32_t rowStride=0,uint32_t query=0) {
        GlobalTensor<uint16_t> input,outputScore;
        GlobalTensor<uint32_t> outputIdx;
        input.SetGlobalBuffer(src);outputIdx.SetGlobalBuffer(dstIdx);
        outputScore.SetGlobalBuffer(dstScores);
        uint32_t loops=(length+Trunk-1)/Trunk;
        for(uint32_t loop=0;loop<loops;++loop) {
            uint32_t remaining=length-loop*Trunk;
            uint32_t count=remaining<Trunk?remaining:Trunk,prefix=loop?K:0;
            // No stale padding can enter a short final trunk.
            Duplicate(values,uint16_t(0),K+Trunk);
            if(loop) DataCopy(values,scores,K);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            DataCopyExtParams p{1,count*2,0,0,0};
            DataCopyPadExtParams<uint16_t> pad{false,0,0,0};
            if(!producers) {
                DataCopyPad(values[prefix],input[loop*Trunk],p,pad);
            } else {
                // Global history coordinates; padding in producer rows is skipped.
                uint32_t trunkBegin=offset+loop*Trunk,trunkEnd=trunkBegin+count;
                uint32_t pages=history/128;
                // ceil((page+1)*P/pages)-1, not floor(page*P/pages).
                uint32_t first=((trunkBegin/128+1)*producers-1)/pages;
                for(uint32_t s=first;s<producers;++s) {
                    uint32_t pieceBegin=pages*s/producers*128,pieceEnd=pages*(s+1)/producers*128;
                    if(pieceBegin>=trunkEnd)break;
                    uint32_t lo=trunkBegin>pieceBegin?trunkBegin:pieceBegin;
                    uint32_t hi=trunkEnd<pieceEnd?trunkEnd:pieceEnd;
                    if(lo<hi) {
                        uint64_t row=((uint64_t(query/4)*producers+s)*4+query%4)*rowStride;
                        DataCopyExtParams part{1,(hi-lo)*2,0,0,0};
                        DataCopyPad(values[prefix+lo-trunkBegin],input[row+lo-pieceBegin],part,pad);
                    }
                }
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            op(values,indices,scores,(prefix+count+255)/256*256,loop,loops,true);
            PipeBarrier<PIPE_V>();
        }
        if(idMap) {
            Simt::VF_CALL<Remap>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(indices.GetPhyAddr()),idMap);
        } else if(offset) {
            auto signedIndices=indices.ReinterpretCast<int32_t>();
            Adds(signedIndices,signedIndices,int32_t(offset),K);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(outputIdx,indices,K);DataCopy(outputScore,scores,K);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    __aicore__ inline uint64_t Begin() {
        uint64_t start=GetSystemCycle();
        // First V work, and its V->MTE2 input chain, start after PIPE_S read.
        SetFlag<HardEvent::S_V>(EVENT_ID2);
        WaitFlag<HardEvent::S_V>(EVENT_ID2);
        return start;
    }
    __aicore__ inline void Finish(uint64_t start,__gm__ uint64_t* dst) {
        // Run retired output DMA into PIPE_V. GetSystemCycle executes in
        // PIPE_S, so add the missing V->S dependency before the end marker.
        SetFlag<HardEvent::V_S>(EVENT_ID2);
        WaitFlag<HardEvent::V_S>(EVENT_ID2);
        auto local=tickBuf.Get<uint64_t>();
        local.SetValue(0,start);local.SetValue(1,GetSystemCycle());
        local.SetValue(2,GetBlockIdx());local.SetValue(3,0);
        GlobalTensor<uint64_t> output;output.SetGlobalBuffer(dst);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopy(output,local,4);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        pipe.Destroy();
    }
};
#endif

extern "C" __global__ __aicore__ void split_topk_local(
    GM_ADDR score,GM_ADDR outIds,GM_ADDR outScores,uint32_t queries,
    uint32_t history,uint32_t shards,uint32_t workers,GM_ADDR ticks) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        TopKWorker w;w.Init();uint64_t start=w.Begin();
        for(uint32_t task=GetBlockIdx();task<queries*shards;task+=workers) {
            uint32_t q=task/shards,s=task%shards;
            uint32_t begin=(history/256*s/shards)*256;
            uint32_t end=(history/256*(s+1)/shards)*256;
            w.Run(reinterpret_cast<__gm__ uint16_t*>(score)+q*history+begin,end-begin,
                reinterpret_cast<__gm__ uint32_t*>(outIds)+task*K,
                reinterpret_cast<__gm__ uint16_t*>(outScores)+task*K,begin);
        }
        w.Finish(start,reinterpret_cast<__gm__ uint64_t*>(ticks)+GetBlockIdx()*4);
    }
#endif
}

extern "C" __global__ __aicore__ void split_topk_merge(
    GM_ADDR inIds,GM_ADDR inScores,GM_ADDR outIds,GM_ADDR outScores,
    uint32_t queries,uint32_t shards,uint32_t workers,GM_ADDR ticks) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        TopKWorker w;w.Init();uint64_t start=w.Begin();
        for(uint32_t q=GetBlockIdx();q<queries;q+=workers)
            w.Run(reinterpret_cast<__gm__ uint16_t*>(inScores)+q*shards*K,shards*K,
                reinterpret_cast<__gm__ uint32_t*>(outIds)+q*K,
                reinterpret_cast<__gm__ uint16_t*>(outScores)+q*K,0,
                reinterpret_cast<__gm__ uint32_t*>(inIds)+q*shards*K);
        w.Finish(start,reinterpret_cast<__gm__ uint64_t*>(ticks)+GetBlockIdx()*4);
    }
#endif
}

// P producer rows -> T exact selection ranges; no full score rearrangement.
extern "C" __global__ __aicore__ void history_topk_local(
    GM_ADDR input,GM_ADDR outIds,GM_ADDR outScores,uint32_t queries,
    uint32_t history,uint32_t rowStride,uint32_t producers,uint32_t topkShards,
    uint32_t workers,GM_ADDR ticks) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        TopKWorker w;w.Init();uint64_t start=w.Begin();
        for(uint32_t task=GetBlockIdx();task<queries*topkShards;task+=workers) {
            uint32_t q=task/topkShards,s=task%topkShards,b=q/4,r=q%4;
            uint32_t begin=history/128*s/topkShards*128,end=history/128*(s+1)/topkShards*128;
            auto source=reinterpret_cast<__gm__ uint16_t*>(input);
            auto ids=reinterpret_cast<__gm__ uint32_t*>(outIds)+task*K;
            auto scores=reinterpret_cast<__gm__ uint16_t*>(outScores)+task*K;
            if(producers==topkShards) {
                uint64_t row=(uint64_t(b*producers+s)*4+r)*rowStride;
                w.Run(source+row,end-begin,ids,scores,begin);
            } else {
                w.Run(source,end-begin,ids,scores,begin,nullptr,producers,history,rowStride,q);
            }
        }
        w.Finish(start,reinterpret_cast<__gm__ uint64_t*>(ticks)+GetBlockIdx()*4);
    }
#endif
}
