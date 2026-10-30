#pragma once
namespace rotating {
__aicore__ inline __ubuf__ uint32_t* FinalPtr(LocalTensor<uint32_t> x) {return reinterpret_cast<__ubuf__ uint32_t*>(x.GetPhyAddr());}

__aicore__ inline __gm__ FinalPost* FinalPostAt(__gm__ Context* ctx,uint32_t core) {
    return reinterpret_cast<__gm__ FinalPost*>(ctx->owners+kFinalPostOffset)+core;
}
__aicore__ inline void PublishFinalWorker(__gm__ Context* ctx,__gm__ Row* row,
    uint32_t b,uint32_t w,uint32_t own,uint32_t batchCount,uint32_t descriptor,uint32_t error,
    LocalTensor<uint32_t> local,LocalTensor<uint32_t> remoteLow,LocalTensor<uint32_t> remoteHigh,
    LocalTensor<uint32_t> slots,LocalTensor<uint32_t> ctl,bool conflict=false) {
    auto args=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(ctx->args)+FinalQueue(b,w);
    auto post=FinalPostAt(ctx,b*kFinalWorkers+w);
    auto cfg=ctl[1024],meta=ctl[1280],queueLocal=ctl[1344];
    GlobalTensor<uint32_t> entityGm,queueGm;
    entityGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(args->channel_entity));
    queueGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(args->queue_state));
    DataCopy(meta,entityGm,48);DataCopy(queueLocal,queueGm,16);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    auto meta64=meta.ReinterpretCast<uint64_t>(),queue64=queueLocal.ReinterpretCast<uint64_t>();
    uint64_t head=queue64.GetValue(1),doorbell=meta64.GetValue(8);uint32_t depth=meta.GetValue(19);
    if(depth<2 || (depth&(depth-1)) || !meta64.GetValue(7) || !doorbell ||
       head>UINT64_MAX-own || queue64.GetValue(4))error|=64;
#if !V9_SIMULATION
    if(queueLocal.GetValue(1) && queueLocal.GetValue(0)!=kvaf_device::kQueueMagic)error|=64;
    queueLocal.SetValue(0,kvaf_device::kQueueMagic);queueLocal.SetValue(1,1);
#endif
    // Recycling a speculative slot must not race its still-in-flight prefetch.
    bool deferred=own && ((ctx->contract ? conflict : row->slots+batchCount>kSlots) || own>=depth);
    bool submit=own && !deferred && !error;
    if(submit){
        cfg.SetValue(1,6U<<8);cfg.SetValue(2,(meta.GetValue(20)&0xffffffU)|0x1000000U);cfg.SetValue(3,meta.GetValue(6)&0xfffffU);
        for(uint32_t j=0;j<4;++j)cfg.SetValue(4+j,meta.GetValue(22+j));
        cfg.SetValue(8,meta.GetValue(7));cfg.SetValue(9,0);cfg.SetValue(12,kTokenBytes);cfg.SetValue(13,meta.GetValue(12));
        cfg.SetValue(14,uint32_t(row->output));cfg.SetValue(15,uint32_t(row->output>>32));cfg.SetValue(16,uint32_t(head));cfg.SetValue(17,depth);
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        GlobalTensor<uint32_t> sq;sq.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(meta64.GetValue(7)));
        for(uint32_t begin=0;begin<own;begin+=1024){uint32_t n=own-begin<1024?own-begin:1024;
            FinalBuild(FinalPtr(remoteLow),FinalPtr(remoteHigh),FinalPtr(slots),FinalPtr(local),FinalPtr(cfg),begin,n,own);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            uint32_t start=uint32_t(head+begin)&(depth-1),first=n<depth-start?n:depth-start;
            DataCopy(sq[start*16],local,first*16);if(first<n)DataCopy(sq,local[first*16],(n-first)*16);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        queue64.SetValue(1,head+own);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(queueGm,queueLocal,16);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
#if V9_SIMULATION
        *reinterpret_cast<__gm__ uint64_t*>(doorbell)=head+own;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(doorbell),64);
#else
        st_dev(uint32_t(head+own),reinterpret_cast<__gm__ uint32_t*>(doorbell),0);
#endif
    }
    post->count=own;post->pending=submit;post->error=error;post->deferred=deferred;
    post->descriptor=descriptor;post->head=head;
    post->reserved=submit ? own : 0; // Actual submitted requests, including failed calls.
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
}
}
