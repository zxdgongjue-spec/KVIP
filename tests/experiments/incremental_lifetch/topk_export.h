// QLIVector member: called after ID recovery while the complete Q is in UB.
bool exportWindowCold_=false;
__aicore__ inline void ExportWindowCold(uint32_t query) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    uint32_t valid=ctx->queryEnds?reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)[query]:runtimeShape_.history;
    if(valid>runtimeShape_.history)valid=runtimeShape_.history;
    auto compact=topkSharedTmpLocal_[64];
    rotating::CompactColdTopk(reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(compact.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(topkSharedTmpLocal_.GetPhyAddr()),runtimeShape_.resident,valid);
    SetFlag<HardEvent::V_S>(EVENT_ID5);WaitFlag<HardEvent::V_S>(EVENT_ID5);
    uint32_t count=topkSharedTmpLocal_.GetValue(0);
    if(incrementalMetadataActive_)incrementalMetadata_.SetValue(incrementalMetadataIndex_*16+3,count);
    else {auto meta=IncrementalMeta(query);meta[3]=count;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);}
    if(count) {
        GlobalTensor<uint32_t> jobs;jobs.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+query*2048);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID5);WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
        DataCopy(jobs,compact,(count+7)/8*8);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID5);WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
    }
}
