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
#if KVIP_INCREMENTAL_DIRECT
    uint32_t incrementalPending_ = 0;
    bool incrementalPacked_ = false;
#endif
#if KVIP_INCREMENTAL_TIES
    uint32_t incrementalStrict_ = 0;
#endif

#if KVIP_INCREMENTAL_DIRECT
    template<bool ties>
    __aicore__ inline void MaterializeIncremental(__ubuf__ uint16_t* idx,
                                                 __ubuf__ uint16_t* offsets) {
        auto values = reinterpret_cast<__ubuf__ uint16_t*>(mrgValueLocal_.GetPhyAddr());
        auto previous = reinterpret_cast<__ubuf__ uint32_t*>(incrementalIds_.GetPhyAddr());
        auto scores = reinterpret_cast<__ubuf__ uint16_t*>(scoreOutLocal_.GetPhyAddr());
        auto ids = reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr());
        auto pending = reinterpret_cast<__ubuf__ uint32_t*>(topkSharedTmpLocal_[32].GetPhyAddr());
        uint32_t strict = 0;
#if KVIP_INCREMENTAL_TIES
        if constexpr (ties) strict = incrementalStrict_;
#endif
#if KVIP_INCREMENTAL_LIGHT
        if (offsets) {
            auto fresh = reinterpret_cast<__ubuf__ uint32_t*>(
                topkSharedTmpLocal_[kvip_incremental::kMappedWord].GetPhyAddr());
            if constexpr (ties) {
                idx = reinterpret_cast<__ubuf__ uint16_t*>(topkOp_.IncrementalOrdinals().GetPhyAddr());
                kvip_incremental::TieOrdinals(idx, uint16_t(strict), uint16_t(incrementalCount_));
                PipeBarrier<PIPE_V>();
            }
            kvip_incremental::MapNewIds(offsets, fresh, incrementalCount_, incrementalBegin_,
                incrementalReverseBegin_, incrementalReverseEnd_, incrementalTailBegin_, incrementalTailEnd_);
            topkb16gather::FindValueOutputVFImpl(scores, values, idx, uint16_t(2048 / 128));
            PipeBarrier<PIPE_V>();
            kvip_incremental::GatherIds(idx, previous, fresh, ids, pending);
            return;
        }
#endif
        // Dense windows retain every original position; no compact-ID buffer
        // is large enough for them. Keep the exact general remapping path.
        kvip_incremental::Materialize<ties>(idx, values, previous, offsets, scores, ids, pending,
            strict, incrementalCount_, incrementalBegin_, incrementalReverseBegin_, incrementalReverseEnd_,
            incrementalTailBegin_, incrementalTailEnd_);
    }
#endif

    __aicore__ inline void SaveIncremental(uint32_t bank) {
        PipeBarrier<PIPE_V>();
        DataCopy(incrementalScores_, scoreOutLocal_, 2048);
        DataCopy(incrementalIds_, indicesOutLocal_, 2048);
        kvip_incremental::BroadcastThreshold(reinterpret_cast<__ubuf__ uint16_t*>(topkOp_.ThresholdResult().GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(prefixThreshold_[bank * 128].GetPhyAddr()));
        PipeBarrier<PIPE_V>();
        // scoreOut also backs the original progress table. Recreate that
        // table on its next publication, never publish TopK scores as progress.
#if !KVIP_INCREMENTAL_PROGRESS
        rotationProgressBase_ = UINT32_MAX;
#endif
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
#if KVIP_INCREMENTAL_TIES
        // Reuse the seed selector's GT count; do not rescan the selected K.
        incrementalStrict_ = topkOp_.IncrementalStrictCount();
#endif
#if KVIP_INCREMENTAL_DIRECT
        incrementalPacked_ = false;
#endif
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
#if KVIP_INCREMENTAL_DIRECT
        incrementalPacked_ = false;
#endif
        if (!incrementalCount_) return;
        // The original progress table aliases scoreOut. With an independent
        // bank, score/progress DMA sources are disjoint from selector outputs.
        // PublishIncremental still joins its own payload/header transfers.
#if !KVIP_INCREMENTAL_PROGRESS
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
#endif
        PipeBarrier<PIPE_V>();
        auto offsets = reinterpret_cast<__ubuf__ uint16_t*>(incrementalOffsets_.GetPhyAddr());
#if KVIP_INCREMENTAL_PREFILTER
        auto packed = topkSharedTmpLocal_.template ReinterpretCast<uint16_t>();
        auto window = reinterpret_cast<__ubuf__ uint16_t*>(mrgValueLocal_[2048].GetPhyAddr());
        kvip_incremental::PackWindow<false>(window,
            reinterpret_cast<__ubuf__ uint16_t*>(packed.GetPhyAddr()), incrementalCount_);
        const uint32_t admitted = GetSpr<SpecialPurposeReg::AR>() / sizeof(uint16_t);
        if (!admitted) {
            incrementalCount_ = 0;
            return; // K scores/IDs and threshold are already exact; retry list still publishes.
        }
        if (admitted <= kvip_incremental::kChunk) {
            PipeBarrier<PIPE_V>();
            kvip_incremental::PackWindow<true>(window, offsets, incrementalCount_);
            PipeBarrier<PIPE_V>();
            DataCopy(mrgValueLocal_[2048], packed, QLICommon::Align(admitted, uint32_t(16)));
            incrementalCount_ = admitted;
        } else {
            offsets = nullptr; // Preserve all fixed positions; never truncate a dense window.
        }
#endif
#if KVIP_INCREMENTAL_INPLACE
        // CANN 9.2 Model: short suffixes win; moving long tie runs regresses.
        if (incrementalStrict_ >= kvip_incremental::kTop - 256 && offsets && incrementalCount_ < kvip_incremental::kTop - incrementalStrict_) {
            auto fresh = reinterpret_cast<__ubuf__ uint32_t*>(
                topkSharedTmpLocal_[kvip_incremental::kMappedWord].GetPhyAddr());
            kvip_incremental::MapNewIds(offsets, fresh, incrementalCount_, incrementalBegin_,
                incrementalReverseBegin_, incrementalReverseEnd_, incrementalTailBegin_, incrementalTailEnd_);
            PipeBarrier<PIPE_V>();
            kvip_incremental::AppendBoundary(
                reinterpret_cast<__ubuf__ uint16_t*>(incrementalScores_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(incrementalIds_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(mrgValueLocal_[2048].GetPhyAddr()),
                fresh, incrementalStrict_, incrementalCount_);
            PipeBarrier<PIPE_V>();
            incrementalStrict_ += incrementalCount_;
            incrementalCount_ = 0;
            return; // Kth and strict prefix unchanged; publication compacts the updated IDs.
        }
#endif
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
#if KVIP_INCREMENTAL_DIRECT
#if KVIP_INCREMENTAL_TIES
        if (offsets && incrementalCount_ < kvip_incremental::kTop - incrementalStrict_) {
            // All new strict improvements fit in the old Kth tie group and
            // at least one old tie survives. Kth is therefore exactly unchanged.
            MaterializeIncremental<true>(nullptr, offsets);
            incrementalPending_ = GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t);
            incrementalPacked_ = true;
            PipeBarrier<PIPE_V>();
            DataCopy(incrementalScores_, scoreOutLocal_, 2048);
            DataCopy(incrementalIds_, indicesOutLocal_, 2048);
            PipeBarrier<PIPE_V>();
#if !KVIP_INCREMENTAL_PROGRESS
            rotationProgressBase_ = UINT32_MAX;
#endif
            incrementalStrict_ += incrementalCount_;
            incrementalCount_ = 0;
            return;
        }
#endif
        auto cutoff = topkOp_.ThresholdResult();
        auto ordinals = topkOp_.IncrementalOrdinals();
        topkOp_.Threshold(mrgValueLocal_, cutoff, size);
        PipeBarrier<PIPE_V>();
        auto idx = reinterpret_cast<__ubuf__ uint16_t*>(ordinals.GetPhyAddr());
        auto values = reinterpret_cast<__ubuf__ uint16_t*>(mrgValueLocal_.GetPhyAddr());
        auto kth = reinterpret_cast<__ubuf__ uint32_t*>(cutoff.GetPhyAddr());
        topkb16gather::FindIdxGTOutputVFImpl(idx, values, uint16_t(0), kth, uint16_t(size / 128));
#if KVIP_INCREMENTAL_TIES
        incrementalStrict_ = GetSpr<SpecialPurposeReg::AR>() / sizeof(uint16_t);
#endif
        topkb16gather::FindIdxEQOutputVFImpl(idx, values, uint16_t(0), kth, uint16_t(size / 128));
        PipeBarrier<PIPE_V>();
        MaterializeIncremental<false>(idx, offsets);
        incrementalPending_ = GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t);
        incrementalPacked_ = true;
#else
        topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, size, 0, 1, true);
        PipeBarrier<PIPE_V>();
        Simt::VF_CALL<kvip_incremental::Remap>(Simt::Dim3{256,1,1},
            reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(incrementalIds_.GetPhyAddr()),
            offsets,
            incrementalBegin_,incrementalReverseBegin_,incrementalReverseEnd_,0,0,incrementalTailBegin_,incrementalTailEnd_);
#endif
        SaveIncremental(bank);
        incrementalCount_ = 0;
    }

    __aicore__ inline void PublishIncremental(uint32_t query, uint32_t epoch) {
        auto ctx = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
#if KVIP_INCREMENTAL_DIRECT
        // Header+IDs share one contiguous staging area. The selected-ordinal
        // scratch starts later in this allocation and is never overwritten.
        if (!incrementalPacked_) {
            kvip_incremental::CompactIds(
                reinterpret_cast<__ubuf__ uint32_t*>(incrementalIds_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(topkSharedTmpLocal_[32].GetPhyAddr()));
            incrementalPending_ = GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t);
        }
        const uint32_t count = incrementalPending_;
        auto header = topkSharedTmpLocal_;
#else
        auto flags = mrgValueLocal_.template ReinterpretCast<uint32_t>();
        kvip_incremental::CompactIds(
            reinterpret_cast<__ubuf__ uint32_t*>(incrementalIds_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()));
        const uint32_t count = GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t);
        auto header = flags[2048];
#endif
#if KVIP_INCREMENTAL_LIGHT
        kvip_incremental::ListHeader(reinterpret_cast<__ubuf__ uint32_t*>(header.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(prefixThreshold_[(query%2)*128].GetPhyAddr()), count, epoch+1);
#else
        Duplicate(header,uint32_t(0),32);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        header.SetValue(0,count);
        // Reuse this already-transferred header for untimed prefix validation.
        header.SetValue(1,prefixThreshold_.GetValue((query%2)*128));
        header.SetValue(2,epoch+1);
#endif
        GlobalTensor<uint32_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->incrementalLists) +
            uint64_t(query*runtimeShape_.rounds+epoch)*kvip_incremental::kListWords);
#if !KVIP_INCREMENTAL_LIGHT
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
#endif
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
#if KVIP_INCREMENTAL_DIRECT
        DataCopy(output,header,32+(count+7)/8*8);
        incrementalPacked_ = false;
#else
        if(count)DataCopy(output[kvip_incremental::kHeader],indicesOutLocal_,(count+7)/8*8);
        DataCopy(output,header,32);
#endif
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
#if KVIP_INCREMENTAL_LIGHT
        DataCopy(rotationProgressGm_[query*32+kvip_incremental::kReadyWord],header[8],8);
#else
        header.SetValue(0,epoch+1);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        // DMA only this 32-byte stripe; scalar cache-line writeback could
        // overwrite the adjacent normal progress stripe with a stale value.
        DataCopy(rotationProgressGm_[query*32+kvip_incremental::kReadyWord],header,8);
#endif
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
#endif
