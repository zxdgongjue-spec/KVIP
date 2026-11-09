// Shared resolver setup, hot/cold classification and workspace preparation.
    const uint32_t core=GetBlockIdx(),b=core/rt_finalWorkers,w=core%rt_finalWorkers;
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
    auto trace=reinterpret_cast<__gm__ uint64_t*>(ctx->owners)+b*16;
    uint64_t started=GetSystemCycle();
    TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,kFinalScratchBytes);auto local=scratch.Get<uint32_t>();
    auto selected=local,loc=local[kSlots],keys=local[2*kSlots],ranks=local[3*kSlots];
    auto ids=local[4*kSlots].ReinterpretCast<uint16_t>();
    auto slots=local[kFinalSlotsOffset],positions=local[kFinalPositionsOffset],ctl=local[kFinalControlOffset];
    GlobalTensor<uint32_t> input,output;input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+b*kSlots);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots);
    DataCopy(selected,input,kSlots);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if(rt_resident<=2*kSlots) {
        static_assert(6*kSlots<=kFinalControlOffset,"metadata staging overlaps resolver control");
        // Hash/ID/slot scratch is not live yet. Reuse it for two dense metadata
        // rows; release it before ColdPositions and the original resolver.
        GlobalTensor<uint32_t> metadata;
        uint32_t owner=ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
        if(ctx->hotValid) {
            metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+owner*rt_resident);
            DataCopy(keys,metadata,rt_resident);
        }
        if(ctx->reservedHot) {
            metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot)+b*rt_resident);
            DataCopy(local[4*kSlots],metadata,rt_resident);
        }
        uint32_t begin=0,end=0;
        if(ctx->tailBounds) {
            auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;
            begin=bounds[0];end=bounds[1];
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Simt::VF_CALL<AdaptColdPredicateLocal>(Simt::Dim3{512,1,1},rt_resident,FinalPtr(selected),FinalPtr(loc),
            FinalPtr(keys),reinterpret_cast<__ubuf__ int32_t*>(local[4*kSlots].GetPhyAddr()),
            ctx->hotValid!=0,ctx->reservedHot!=0,begin,end);
        PipeBarrier<PIPE_V>();
        ColdPositions(rt_history,rt_resident,FinalPtr(loc),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]));
    } else {
        // Larger supported resident layouts cannot hold both rows in this UB.
        Simt::VF_CALL<AdaptColdPredicate>(Simt::Dim3{512,1,1},rt_resident,ctx,b,FinalPtr(selected),FinalPtr(keys));
        PipeBarrier<PIPE_V>();
        ColdPositions(rt_history,rt_resident,FinalPtr(keys),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]));
    }
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t cold=ctl.GetValue(2048),error=ctl.GetValue(2056)|row->error|(row->slots>kSlots?32U:0U);

    if(!w){trace[0]=started;trace[1]=GetSystemCycle();}
