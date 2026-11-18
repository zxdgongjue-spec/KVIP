// Private experimental output sidecar; virtual score inputs remain unchanged.
#pragma once
#include "producer_plan.h"
#include "plan.h"
namespace li_producer {
class DeviceState {
    __gm__ const Bindings* a_=nullptr;
    li_qk::Task task_{};
    AscendC::TBuf<AscendC::TPosition::VECCALC> high_,tmp_,pad_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> retained_;
    topk::LITopk<uint16_t> localTopk_;
    uint32_t first_=0;
public:
    __aicore__ inline void Init(GM_ADDR binding,AscendC::TPipe* pipe,uint32_t block) {
        a_=reinterpret_cast<__gm__ const Bindings*>(binding);
        auto t=reinterpret_cast<__gm__ const li_qk::Task*>(a_->tasks)+block/2;
        task_={t->queryBegin,t->queryCount,t->begin,t->end,t->owner};
        first_=(block%2)*2;
        if(a_->route==2){pipe->InitBuffer(high_,2048);pipe->InitBuffer(tmp_,1024);pipe->InitBuffer(pad_,512);
            AscendC::Duplicate(high_.Get<uint32_t>(),uint32_t(0),512);}
        if(a_->route==3){pipe->InitBuffer(retained_,2*Shard*2);localTopk_.Init(K,Shard);}
    }
    __aicore__ inline bool Enabled() const {return a_!=nullptr;}
    __aicore__ inline void Tile(AscendC::LocalTensor<uint16_t> scores,uint32_t stride,
        uint32_t first,uint32_t count,uint32_t token) {
        using namespace AscendC;
        GlobalTensor<uint16_t> dst;dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(a_->canonical));
        DataCopyExtParams cp{uint16_t(count),256,(stride*2-256)/32,(a_->history-128)*2,0};
        DataCopyPad(dst[(uint64_t(task_.owner)*4+task_.queryBegin+first)*a_->history+task_.begin+token],scores,cp);
        if(a_->route==3)for(uint32_t q=0;q<count;++q){
            DataCopy(retained_.Get<uint16_t>()[(first+q-first_)*Shard+token],scores[q*stride],128);PipeBarrier<PIPE_ALL>();}
        if(a_->route==2)for(uint32_t q=0;q<count;++q){
            auto in=pad_.Get<uint16_t>();auto tmp=tmp_.Get<uint32_t>();auto sum=high_.Get<uint32_t>()[(first+q-first_)*256];
            Duplicate(in,uint16_t(0),256);PipeBarrier<PIPE_V>();DataCopy(in,scores[q*stride],128);PipeBarrier<PIPE_ALL>();
            topkb16gather::HistogramsHighVFImpl<uint16_t>(reinterpret_cast<__ubuf__ uint32_t*>(tmp.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(in.GetPhyAddr()),1,true);
            PipeBarrier<PIPE_V>();Adds(tmp.template ReinterpretCast<int32_t>(),tmp.template ReinterpretCast<int32_t>(),int32_t(-128),256);PipeBarrier<PIPE_V>();Add(sum,sum,tmp,256);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Finish(AscendC::LocalTensor<uint32_t> shared,
        AscendC::LocalTensor<uint32_t> indices,AscendC::LocalTensor<uint16_t> values) {
        using namespace AscendC;
        if(a_->route==2){GlobalTensor<uint32_t> dst;dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a_->high));
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            for(uint32_t q=0;q<2;++q)DataCopy(dst[((task_.owner*4+task_.queryBegin+first_+q)*Parts+task_.begin/Shard)*256],high_.Get<uint32_t>()[q*256],256);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);}
        if(a_->route==3){
            localTopk_.InitBuffers(shared,indices);GlobalTensor<uint16_t> scores;GlobalTensor<uint32_t> ids;
            scores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(a_->candidateScores));ids.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a_->candidateIds));
            for(uint32_t q=0;q<2;++q){auto input=retained_.Get<uint16_t>()[q*Shard];
                PipeBarrier<PIPE_ALL>();localTopk_(input,indices,values,Shard,0,1,true);PipeBarrier<PIPE_V>();
                Adds(indices.template ReinterpretCast<int32_t>(),indices.template ReinterpretCast<int32_t>(),int32_t(task_.begin),K);PipeBarrier<PIPE_V>();
                uint32_t offset=((task_.owner*4+task_.queryBegin+first_+q)*Parts+task_.begin/Shard)*K;
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopy(scores[offset],values,K);DataCopy(ids[offset],indices,K);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }
};
}
