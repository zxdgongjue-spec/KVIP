#pragma once
// WQE fields/owner/completion encoding are inherited from the Codex SIMD demo.
// Only the address gather and destination slots differ: long histories do not
// fit in UB, and final repair may safely recycle unreferenced speculative slots.
namespace rotating {
__simd_vf__ inline void FinalBuild(__ubuf__ uint32_t* remoteLow, __ubuf__ uint32_t* remoteHigh, __ubuf__ uint32_t* slots, __ubuf__ uint32_t* words,
    __ubuf__ uint32_t* config, uint32_t begin, uint32_t count, uint32_t total) {
    using namespace MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<int32_t> sequenceSigned;
    auto& sequence = reinterpret_cast<RegTensor<uint32_t>&>(sequenceSigned);
    RegTensor<uint32_t> index, offsets, remoteLo, remoteHi, localLo, localHi;
    RegTensor<uint32_t> baseLo, baseHi, delta, one, zero, carry, value, pi, depth, owner, control;
    LoadAlign<uint32_t, LoadDist::DIST_BRC_B32>(baseLo, config + 14);
    LoadAlign<uint32_t, LoadDist::DIST_BRC_B32>(baseHi, config + 15);
    LoadAlign<uint32_t, LoadDist::DIST_BRC_B32>(pi, config + 16);
    LoadAlign<uint32_t, LoadDist::DIST_BRC_B32>(depth, config + 17);
    Duplicate(one, uint32_t(1)); Duplicate(zero, uint32_t(0));
    uint32_t remaining = count;
    for (uint16_t j = 0; j < count; j += 64) {
        auto mask = UpdateMask<uint32_t>(remaining);
        Arange(sequenceSigned, int32_t(j));
        Adds(index, sequence, begin, mask);
        RegTensor<uint32_t> slot;
        LoadAlign(slot, slots + begin + j);
        LoadAlign(remoteLo,remoteLow+begin+j);LoadAlign(remoteHi,remoteHigh+begin+j);
        Muls(delta, slot, uint32_t(kTokenBytes), mask);
        Add(localLo, baseLo, delta, mask);
        MaskReg overflow;
        Compare<uint32_t, CMPMODE::LT>(overflow, localLo, baseLo, mask);
        Select(carry, one, zero, overflow); Add(localHi, baseHi, carry, mask);
        Add(value, pi, index, mask); And(value, value, depth, mask);
        MaskReg firstLap, completion;
        Compare<uint32_t, CMPMODE::EQ>(firstLap, value, zero, mask);
        Select(owner, one, zero, firstLap); ShiftLefts(owner, owner, int16_t(31), mask);
        CompareScalar<uint32_t, CMPMODE::EQ>(completion, index, total - 1, mask);
        Duplicate(control, uint32_t(0x30150000)); Duplicate(value, uint32_t(0x30360000));
        Select(control, value, control, completion); Or(control, control, owner, mask);
        // Transpose 16 field vectors into 64 complete WQEs in registers.
        // Contiguous stores avoid the stride-16 Scatter bank conflicts.
        RegTensor<uint32_t> f1,f2,f3,f4,f5,f6,f7,f8,f9,length,localToken,low,high;
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f1,config+1);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f2,config+2);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f3,config+3);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f4,config+4);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f5,config+5);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f6,config+6);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f7,config+7);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f8,config+8);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f9,config+9);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(length,config+12);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(localToken,config+13);
        // Interleave field groups of 8.
        Interleave(low,high,control,f8);control=low;f8=high;
        Interleave(low,high,f1,f9);f1=low;f9=high;
        Interleave(low,high,f2,remoteLo);f2=low;remoteLo=high;
        Interleave(low,high,f3,remoteHi);f3=low;remoteHi=high;
        Interleave(low,high,f4,length);f4=low;length=high;
        Interleave(low,high,f5,localToken);f5=low;localToken=high;
        Interleave(low,high,f6,localLo);f6=low;localLo=high;
        Interleave(low,high,f7,localHi);f7=low;localHi=high;
        // Interleave field groups of 4.
        Interleave(low,high,control,f4);control=low;f4=high;
        Interleave(low,high,f1,f5);f1=low;f5=high;
        Interleave(low,high,f2,f6);f2=low;f6=high;
        Interleave(low,high,f3,f7);f3=low;f7=high;
        Interleave(low,high,f8,length);f8=low;length=high;
        Interleave(low,high,f9,localToken);f9=low;localToken=high;
        Interleave(low,high,remoteLo,localLo);remoteLo=low;localLo=high;
        Interleave(low,high,remoteHi,localHi);remoteHi=low;localHi=high;
        // Interleave field groups of 2.
        Interleave(low,high,control,f2);control=low;f2=high;
        Interleave(low,high,f1,f3);f1=low;f3=high;
        Interleave(low,high,f4,f6);f4=low;f6=high;
        Interleave(low,high,f5,f7);f5=low;f7=high;
        Interleave(low,high,f8,remoteLo);f8=low;remoteLo=high;
        Interleave(low,high,f9,remoteHi);f9=low;remoteHi=high;
        Interleave(low,high,length,localLo);length=low;localLo=high;
        Interleave(low,high,localToken,localHi);localToken=low;localHi=high;
        // Interleave field groups of 1.
        Interleave(low,high,control,f1);control=low;f1=high;
        Interleave(low,high,f2,f3);f2=low;f3=high;
        Interleave(low,high,f4,f5);f4=low;f5=high;
        Interleave(low,high,f6,f7);f6=low;f7=high;
        Interleave(low,high,f8,f9);f8=low;f9=high;
        Interleave(low,high,remoteLo,remoteHi);remoteLo=low;remoteHi=high;
        Interleave(low,high,length,localToken);length=low;localToken=high;
        Interleave(low,high,localLo,localHi);localLo=low;localHi=high;
        StoreAlign(words+j*16+0,control,all);
        StoreAlign(words+j*16+64,f1,all);
        StoreAlign(words+j*16+128,f2,all);
        StoreAlign(words+j*16+192,f3,all);
        StoreAlign(words+j*16+256,f4,all);
        StoreAlign(words+j*16+320,f5,all);
        StoreAlign(words+j*16+384,f6,all);
        StoreAlign(words+j*16+448,f7,all);
        StoreAlign(words+j*16+512,f8,all);
        StoreAlign(words+j*16+576,f9,all);
        StoreAlign(words+j*16+640,remoteLo,all);
        StoreAlign(words+j*16+704,remoteHi,all);
        StoreAlign(words+j*16+768,length,all);
        StoreAlign(words+j*16+832,localToken,all);
        StoreAlign(words+j*16+896,localLo,all);
        StoreAlign(words+j*16+960,localHi,all);
    }
}

__aicore__ inline bool SubmitWqesAt(uint64_t entityAddress,uint64_t queueAddress,uint64_t output,LocalTensor<uint32_t> slots,LocalTensor<uint32_t> words,
    LocalTensor<uint32_t> remoteLow,LocalTensor<uint32_t> remoteHigh,LocalTensor<uint32_t> cfg,uint32_t count,uint32_t bankSize,bool final,__gm__ Row* row,uint32_t pending=0) {
    auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(queueAddress);
    // Intermediate service exclusively owns this queue; CQ draining starts
    // after LI completes. Keep its state on the DMA path, like the batch row.
    auto queueLocal=cfg[320];auto queue64=queueLocal.ReinterpretCast<uint64_t>();
    GlobalTensor<uint32_t> queueGm;queueGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(queueAddress));
    uint64_t head;
    if(final) {
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        head=queue->sq_producer_index;
    } else {
        DataCopy(queueLocal,queueGm,16);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        head=queue64.GetValue(1);
    }
    if(final) {row->finalRequests=count;row->finalHead=head;}
    if(!count)return false;
    // The entity is immutable while leased. Fetch its six cache lines with
    // one DMA instead of dependent scalar GM loads for individual fields.
    static_assert(sizeof(KvafBatchGetDirectUrmaEntity)==192,"entity ABI changed");
    auto meta=cfg[256];auto meta64=meta.ReinterpretCast<uint64_t>();
    GlobalTensor<uint32_t> entityGm;entityGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(entityAddress));
    DataCopy(meta,entityGm,48);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    uint32_t sqDepth=meta.GetValue(19);uint64_t sqAddress=meta64.GetValue(7),doorbell=meta64.GetValue(8);
    if(count>=sqDepth || (!final && pending+1>=meta.GetValue(31)))return false;
    cfg.SetValue(1,6U<<8);cfg.SetValue(2,(meta.GetValue(20)&0xffffffU)|0x1000000U);
    cfg.SetValue(3,meta.GetValue(6)&0xfffffU);
    for(uint32_t j=0;j<4;++j)cfg.SetValue(4+j,meta.GetValue(22+j));
    cfg.SetValue(8,meta.GetValue(7));cfg.SetValue(9,0);
    cfg.SetValue(12,kTokenBytes);cfg.SetValue(13,meta.GetValue(12));
    cfg.SetValue(14,uint32_t(output));cfg.SetValue(15,uint32_t(output>>32));
    cfg.SetValue(16,uint32_t(head));cfg.SetValue(17,sqDepth);
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    GlobalTensor<uint32_t> sq;sq.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sqAddress));
    // selected and locations are both dead: their contiguous 64KiB holds
    // 1024 WQEs, halving the dispatch/DMA fences for the usual final batch.
    for(uint32_t begin=0;begin<count;begin+=bankSize) {
        uint32_t n=count-begin<bankSize?count-begin:bankSize;
        FinalBuild(reinterpret_cast<__ubuf__ uint32_t*>(remoteLow.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(remoteHigh.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(slots.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(words.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),begin,n,count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        uint32_t start=uint32_t(head+begin)&(sqDepth-1),first=n<sqDepth-start?n:sqDepth-start;
        DataCopy(sq[start*16],words,first*16);
        if(first!=n)DataCopy(sq,words[first*16],(n-first)*16);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    if(!final) {
        queue64.SetValue(1,head+count);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(queueGm,queueLocal,16);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
#if V9_SIMULATION
        *reinterpret_cast<__gm__ uint64_t*>(doorbell)=head+count;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(doorbell),64);
#else
        st_dev(uint32_t(head+count),reinterpret_cast<__gm__ uint32_t*>(doorbell),0);
#endif
    } else {
#if V9_SIMULATION
        queue->sq_producer_index=head+count;
        *reinterpret_cast<__gm__ uint64_t*>(doorbell)=head+count;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(doorbell),64);
#else
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        kvaf_device::PendingPost post{doorbell,queueAddress,head+count,0U};
        kvaf_device::CommitPostScalarStage(&post);
#endif
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
    }
    return true;
}
__aicore__ inline bool SubmitWqes(__gm__ Row* row,LocalTensor<uint32_t> slots,LocalTensor<uint32_t> words,
    LocalTensor<uint32_t> remoteLow,LocalTensor<uint32_t> remoteHigh,LocalTensor<uint32_t> cfg,uint32_t count,uint32_t bankSize=1024,bool final=false) {
    return SubmitWqesAt(row->entity,row->queue,row->output,slots,words,remoteLow,remoteHigh,cfg,count,bankSize,final,row);
}
__aicore__ inline void SubmitFinal(__gm__ Row* row,LocalTensor<uint32_t> slots,LocalTensor<uint32_t> words,
    LocalTensor<uint32_t> remoteLow,LocalTensor<uint32_t> remoteHigh,LocalTensor<uint32_t> cfg,uint32_t count) {
    if(SubmitWqes(row,slots,words,remoteLow,remoteHigh,cfg,count,1024,true))row->finalPending=1;
}
}
