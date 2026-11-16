        V9_FETCH_PROFILE(scanBegin=GetSystemCycle();)
        // Counts occupy spare incremental metadata, exported alongside TopK.
        GlobalTensor<uint32_t> metadata,compact;
        metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->incremental)+b*128);
        compact.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*8192);
        DataCopy(cfg[128],metadata,128);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        uint32_t counts[4],maxCount=0;
        for(uint32_t q=0;q<4;++q){counts[q]=cfg.GetValue(128+q*32+6);if(counts[q]>maxCount)maxCount=counts[q];}
        cfg.SetValue(40,0);
        Duplicate(scratch,uint32_t(0),rt_history/32);
        auto input=scratch[rt_history/32];
        // Four 256-ID rows fit before the unchanged service output at 23808.
        // Dense tails use the same bounded loop; all reads finish before jobs
        // overwrite ctx->locations in the original downstream submission path.
        for(uint32_t start=0;start<maxCount;start+=256) {
            uint32_t width=maxCount-start;if(width>256)width=256;width=(width+7)/8*8;
            SetFlag<HardEvent::V_MTE2>(EVENT_ID5);WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
            DataCopyExtParams copy;copy.blockCount=4;copy.blockLen=width*4;copy.srcStride=(2048-width)*4;copy.dstStride=0;
            DataCopyPadExtParams<uint32_t> pad{false,0,0,0};
            DataCopyPad(input,compact[start],copy,pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            uint32_t n[4];for(uint32_t q=0;q<4;++q)n[q]=counts[q]>start?((counts[q]-start)<width?counts[q]-start:width):0;
            Simt::VF_CALL<UnionColdTopk>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(input.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),width,n[0],n[1],n[2],n[3]);
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        candidateCount=cfg.GetValue(40);if(candidateCount>kCandidateCapacity)candidateCount=kCandidateCapacity;
