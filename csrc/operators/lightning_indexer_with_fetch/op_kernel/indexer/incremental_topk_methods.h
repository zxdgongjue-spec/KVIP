// SPDX-License-Identifier: Apache-2.0
// Included inside QLIVector. All additions are absent from baseline builds.
#if KVIP_INCREMENTAL_TOPK
    TBuf<TPosition::VECCALC> incrementalBuf_;
    LocalTensor<uint16_t> incrementalScores_, incrementalOffsets_;
    LocalTensor<uint32_t> incrementalIds_;
    uint32_t incrementalCount_ = 0, incrementalEpoch_ = 1;
    uint32_t incrementalBegin_ = 0, incrementalEnd_ = 0;
    uint32_t incrementalRecentBegin_ = 0, incrementalRecentEnd_ = 0;
    uint32_t incrementalTailBegin_ = 0, incrementalTailEnd_ = 0;
    uint32_t incrementalReverseBegin_ = 0, incrementalReverseEnd_ = 0;
    bool incrementalReady_ = false;

    __aicore__ inline void SaveIncremental(uint32_t bank) {
        PipeBarrier<PIPE_V>();
        DataCopy(incrementalScores_, scoreOutLocal_, 2048);
        DataCopy(incrementalIds_, indicesOutLocal_, 2048);
        kvip_incremental::BroadcastThreshold(reinterpret_cast<__ubuf__ uint16_t*>(topkOp_.ThresholdResult().GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(prefixThreshold_[bank * 128].GetPhyAddr()));
        PipeBarrier<PIPE_V>();
        // scoreOut also backs the original progress table. Recreate that
        // table on its next publication, never publish TopK scores as progress.
        rotationProgressBase_ = UINT32_MAX;
    }

    __aicore__ inline void SeedIncremental(uint32_t query, uint32_t residentCount,
        uint32_t begin, uint32_t end, uint32_t causalEnd, uint32_t aligned) {
        topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, aligned, 0, 1, true);
        PipeBarrier<PIPE_V>();
        Simt::VF_CALL<kvip_incremental::Remap>(Simt::Dim3{256,1,1},
            reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(incrementalIds_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(incrementalOffsets_.GetPhyAddr()),
            0, 0, 0, residentCount, begin, 0, 0);
        PipeBarrier<PIPE_V>();
        auto ctx = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
        Simt::VF_CALL<kvip_incremental::TagSeed>(Simt::Dim3{256,1,1},ctx,query/4,runtimeShape_.resident,
            reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()));
        SaveIncremental(query % 2);
        incrementalCount_ = 0; incrementalEpoch_ = 1;
        incrementalBegin_ = runtimeShape_.resident;
        incrementalEnd_ = causalEnd;
        incrementalRecentBegin_ = begin; incrementalRecentEnd_ = end;
        if(ctx->tailBounds) {
            auto tail=reinterpret_cast<__gm__ int32_t*>(ctx->tailBounds)+(query/4)*2;
            incrementalTailBegin_=tail[0]; incrementalTailEnd_=tail[1];
        }
        uint32_t ends[4] = {runtimeShape_.history,runtimeShape_.history,runtimeShape_.history,runtimeShape_.history};
        if(ctx->queryEnds) for(uint32_t q=0;q<4;++q)
            ends[q]=reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)[query/4*4+q];
        auto reverse = kvip_cold_scan::Bounds(runtimeShape_.resident,runtimeShape_.history,ends);
        incrementalReverseBegin_ = reverse.begin; incrementalReverseEnd_ = reverse.end;
        incrementalReady_ = true;
    }

    __aicore__ inline void MergeIncremental(uint32_t bank) {
        if (!incrementalCount_) return;
        // A prior tile may still be copying the progress table from scoreOut.
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        PipeBarrier<PIPE_V>();
        const uint32_t size = QLICommon::Align(2048 + incrementalCount_, uint32_t(256));
#if KVIP_INCREMENTAL_WINDOW
        // Tiles write fixed positions after the retained K. Every position is
        // written once, including masked zeros. Consume before RotateFetch,
        // which reuses this scratch only at the same completed window boundary.
        DataCopy(mrgValueLocal_, incrementalScores_, 2048);
        PipeBarrier<PIPE_V>();
        if (size > 2048 + incrementalCount_) {
            kvip_incremental::ZeroTail(reinterpret_cast<__ubuf__ uint16_t*>(mrgValueLocal_.GetPhyAddr()),
                                      2048+incrementalCount_,size);
        }
#else
        // Source and output banks are disjoint. Padding is less than every
        // selected nonzero key; real zero-score ties precede padding.
        Duplicate(mrgValueLocal_, uint16_t(0), size);
        PipeBarrier<PIPE_V>();
        DataCopy(mrgValueLocal_, incrementalScores_, 2048);
        DataCopy(mrgValueLocal_[2048], incrementalScores_[2048],
                 QLICommon::Align(incrementalCount_, uint32_t(16)));
        PipeBarrier<PIPE_V>();
        if (size > 2048 + incrementalCount_) {
            // DataCopy rounds to a 32-byte block; mask its trailing values.
            kvip_incremental::ZeroTail(reinterpret_cast<__ubuf__ uint16_t*>(mrgValueLocal_.GetPhyAddr()),
                                      2048+incrementalCount_,size);
        }
#endif
        topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, size, 0, 1, true);
        PipeBarrier<PIPE_V>();
        Simt::VF_CALL<kvip_incremental::Remap>(Simt::Dim3{256,1,1},
            reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(incrementalIds_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(incrementalOffsets_.GetPhyAddr()),
            incrementalBegin_,incrementalReverseBegin_,incrementalReverseEnd_,0,0,incrementalTailBegin_,incrementalTailEnd_);
        SaveIncremental(bank);
        incrementalCount_ = 0;
    }

    __aicore__ inline void PublishIncremental(uint32_t query, uint32_t epoch) {
        auto ctx = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
        auto flags = mrgValueLocal_.template ReinterpretCast<uint32_t>();
        kvip_incremental::CompactIds(
            reinterpret_cast<__ubuf__ uint32_t*>(incrementalIds_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()));
        const uint32_t count = GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t);
        auto header = flags[2048];
        Duplicate(header,uint32_t(0),32);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        header.SetValue(0,count);
        // Reuse this already-transferred header for untimed prefix validation.
        header.SetValue(1,prefixThreshold_.GetValue((query%2)*128));
        header.SetValue(2,epoch+1);
        GlobalTensor<uint32_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->incrementalLists) +
            uint64_t(query*runtimeShape_.rounds+epoch)*kvip_incremental::kListWords);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if(count)DataCopy(output[kvip_incremental::kHeader],indicesOutLocal_,(count+7)/8*8);
        DataCopy(output,header,32);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        header.SetValue(0,epoch+1);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        // DMA only this 32-byte stripe; scalar cache-line writeback could
        // overwrite the adjacent normal progress stripe with a stale value.
        DataCopy(rotationProgressGm_[query*32+kvip_incremental::kReadyWord],header,8);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
#endif
