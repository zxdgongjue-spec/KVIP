        // AdvanceWindow joined its writes before entering FastService. Consume
        // that exact TopK snapshot; do not read producer progress again.
        V9_FETCH_PROFILE(scanBegin=GetSystemCycle();)
        const uint32_t unionWords=rt_history/32;
        // Hot residency is shared by all four Qs: load it once in contiguous
        // chunks, not through thousands of indexed GM reads per query.
        GlobalTensor<uint32_t> hotValid,hotReserved;
        uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
        hotValid.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*rt_resident);
        hotReserved.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot)+uint64_t(b)*rt_resident);
        for(uint32_t first=0;first<rt_resident;first+=2048) {
            if(!ctx->hotValid)Duplicate(scratch,uint32_t(1),2048);
            if(!ctx->reservedHot)Duplicate(scratch[2048],UINT32_MAX,2048);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID5);WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
            if(ctx->hotValid)DataCopy(scratch,hotValid[first],2048);
            if(ctx->reservedHot)DataCopy(scratch[2048],hotReserved[first],2048);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
            Simt::VF_CALL<BuildHotEligibility>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scratch[2048].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),first);
            PipeBarrier<PIPE_V>();
        }
        Duplicate(scratch,uint32_t(0),unionWords);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID5);WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
        GlobalTensor<uint32_t> topkIds;
        topkIds.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->incremental+rt_batch*512+
            uint64_t(b)*(4*2048*6+1024)+4*2048*2));
        auto queryIds=scratch[unionWords];
        for(uint32_t q=0;q<4;++q) {
            DataCopy(queryIds,topkIds[q*2048],2048);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
            Simt::VF_CALL<UnionTopkCandidates>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(queryIds.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),ctx,
                reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),b,q,rt_history,rt_resident);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID5);WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
        }
        Simt::VF_CALL<ExpandTopkUnion>(Simt::Dim3{256,1,1},
            reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),unionWords);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        uint32_t candidateCount=cfg.GetValue(40);
