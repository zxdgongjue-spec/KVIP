// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifndef V9_ROLLING_THRESHOLD
#define V9_ROLLING_THRESHOLD 0
#endif
static_assert(V9_ROLLING_THRESHOLD==0 || V9_ROLLING_THRESHOLD==1,
              "Rolling threshold requires an explicit boolean");

namespace QLIKernel {
// These methods use the existing prefix GM score/index rows as persistent
// state. Service borrows every TopK UB buffer, so no rolling state may remain
// there across a service call. Only this query's AIV reads/writes its row.
template<class QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessRollingTopK(const QLICommon::RunInfo& info) {
#if V9_ROLLING_THRESHOLD
    const uint32_t available=(info.s2Idx+1)*s2BaseSize_;
    if(!rollingReady_ || (!info.isLastS2InnerLoop && available%trunkLen_))return;
    const uint32_t curS1=info.gS1Idx*s1BaseSize_;
    const uint32_t row=blockId_%2,query=info.bIdx*SIM_Q+curS1+row;
    int32_t bound=constInfo_.attenMaskFlag?
        int32_t(info.actS2SizeOrig)-int32_t(info.actS1Size)+int32_t(curS1+row)+1:int32_t(info.actS2Size);
    auto context=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    if(context->queryEnds)bound=Min(bound,int32_t(reinterpret_cast<__gm__ uint32_t*>(context->queryEnds)[query]));
    const uint32_t end=bound>0?Min(available,uint32_t(bound)):0;
    if(end<=rollingEnd_)return;
    // The caller reaches this point only after a complete 16K score block,
    // or the final partial block. The original final TopK uses these exact
    // same boundaries, padding, score keys and previous-index order.
    const uint32_t count=end-rollingEnd_,padded=QLICommon::Align(topkCount_+count,uint32_t(256));
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    if(padded>topkCount_+count) {
        const uint32_t first=(topkCount_+count)/256*256;
        Duplicate(mrgValueLocal_[first],SCORE_T(0),padded-first);
    }
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
    constexpr uint32_t entries=SIM_Q*2048,rowBytes=entries*6+SIM_Q*256;
    GlobalTensor<uint16_t> values;
    GlobalTensor<uint32_t> indices;
    values.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(prefixOutput_+(query/SIM_Q)*rowBytes));
    indices.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(prefixOutput_+(query/SIM_Q)*rowBytes+entries*2));
    DataCopy(mrgValueLocal_,values[(query%SIM_Q)*2048],2048);
    // LITopk's previous-index bank is sharedTmp[0]. MergeNext always gathers
    // into indicesOut, and receives the original absolute trunk offset.
    DataCopy(topkSharedTmpLocal_,indices[(query%SIM_Q)*2048],2048);
    DataCopyExtParams copy{1,count*uint32_t(sizeof(SCORE_T)),0,0,0};
    DataCopyPadExtParams<SCORE_T> pad{true,0,0,0};
    const uint64_t scoreStride=QLICommon::Align(uint64_t(constInfo_.kSeqSize),uint64_t(s2BaseSize_));
    DataCopyPad(mrgValueLocal_[topkCount_],scoreGm[row*scoreStride+rollingEnd_],copy,pad);
    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
#if V9_PRUNED_TOPK
    topkOp_.MergePruned(mrgValueLocal_,scoreOutLocal_,prunedPositions_,count,rollingEnd_,1,0);
#else
    topkOp_.MergeNext(mrgValueLocal_,indicesOutLocal_,scoreOutLocal_,padded,rollingEnd_);
#endif
    SaveTopKState(query);
    TraceThreshold(query,end,available,3);
    rollingEnd_=end;
    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    // scoreOut is also the original progress-table scratch. Rebuild it only
    // after saving the rolling scores; the next tile publishes from this bank.
    PrepareRotationProgress((available/128/64)*64);
#endif
}

template<class QLIT>
__aicore__ inline void QLIVector<QLIT>::PublishRollingTopK(const QLICommon::RunInfo& info) {
    const uint32_t query=info.bIdx*SIM_Q+info.gS1Idx*s1BaseSize_+blockId_%2;
    constexpr uint32_t entries=SIM_Q*2048,rowBytes=entries*6+SIM_Q*256;
    GlobalTensor<uint32_t> indices;
    indices.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(prefixOutput_+(query/SIM_Q)*rowBytes+entries*2));
    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    DataCopy(indicesOutLocal_,indices[(query%SIM_Q)*2048],2048);
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    DataCopyExtParams copy{1,2048*sizeof(uint32_t),0,0,0};
    DataCopyPad(indiceOutGm[info.indiceOutOffset+(info.gS1Idx*s1BaseSize_+blockId_%2)*2048],
                indicesOutLocal_.template ReinterpretCast<int32_t>(),copy);
    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}
} // namespace QLIKernel
