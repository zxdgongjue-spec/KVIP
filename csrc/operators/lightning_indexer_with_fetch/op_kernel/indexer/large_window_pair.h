// Included in QLIVector only for the one-request-per-cube large-batch kernel.
// A vector lane owns one or two disjoint 128-byte TopK headers and winner lists.
// The original single writer still consumes all four lists and updates covered.
template<uint32_t Queries>
__aicore__ inline void AdvanceLargeWindowQueries(uint32_t row, uint32_t epoch, uint32_t ready, uint32_t first) {
    static_assert(Queries == 1 || Queries == 2);
    auto ctx = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    auto events = reinterpret_cast<__gm__ rotating::Event*>(ctx->events);
    auto previous = events + (epoch - 1) * runtimeShape_.serviceLanes + row;
    const bool trace = streamingWorkspace_ &&
        (reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_)->prefetchCount & 0x80000000U);
    if (trace && !first) {
        events[epoch * runtimeShape_.serviceLanes + row].waitBegin = GetSystemCycle();
    }
    do {
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(previous), 128);
    } while (!reinterpret_cast<__gm__ volatile rotating::Event*>(previous)->complete);

    const uint64_t begin = trace ? GetSystemCycle() : 0;
    GlobalTensor<uint64_t> header;
    header.SetGlobalBuffer(IncrementalMeta(row * 4 + first));
    GlobalTensor<uint32_t> rows;
    rows.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rows) + row * sizeof(rotating::Row) / 4);
    DataCopy(incrementalOld_, rows, 16);
    DataCopy(incrementalMetadata_, header, Queries * 16);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID5);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID5);
    windowCoveredEmpty_ = incrementalOld_.GetValue(offsetof(rotating::Row, requests) / sizeof(uint32_t)) == 0;
    incrementalMetadataActive_ = true;
    exportWindowCold_ = true;
    rankedEpoch_ = epoch;
    rankTrace_ = trace;
    incrementalCalls_ = incrementalScanned_ = incrementalCandidates_ = 0;
    const uint32_t end = Min(ready, rotating::Trigger(runtimeShape_.history, runtimeShape_.resident,
                                                       runtimeShape_.rounds, epoch) * 128);
    rankHalf_ = 0;
    rankReady_ = ready;
    for (uint32_t i = 0; i < Queries; ++i) {
        incrementalMetadata_.SetValue(i * 16 + 4, ready);
        incrementalMetadataIndex_ = i;
        incrementalHasNext_ = incrementalPipeline_ && i + 1 < Queries;
        incrementalNextIndex_ = i + 1;
        incrementalNextQuery_ = row * 4 + first + i + 1;
        incrementalNextEnd0_ = end;
        incrementalNextEnd1_ = 0;
        AdvanceIncremental(row * 4 + first + i, end, 0, 0);
    }
    DrainWindowCopies();
    SetFlag<HardEvent::S_MTE3>(EVENT_ID5);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID5);
    DataCopy(header, incrementalMetadata_, Queries * 16);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID5);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    incrementalMetadataActive_ = false;
    exportWindowCold_ = false;
    if (trace && !first) {
        auto stamp = reinterpret_cast<__gm__ uint64_t*>(events[epoch * runtimeShape_.serviceLanes + row].pad);
        stamp[2] = begin;
    }
}

__aicore__ inline void AdvanceLargeWindowPair(uint32_t row, uint32_t epoch, uint32_t ready) {
    AdvanceLargeWindowQueries<2>(row, epoch, ready, (blockId_ % 2) * 2);
}

__aicore__ inline void PostLargeWindowRow(uint32_t row, uint32_t epoch) {
    // In the deferred kernel each physical team can serve two different rows.
    // The common posting path converts this logical block ID to its row lane.
    blockId_ = 2 * row;
    PostWindow(epoch, 0);
}

// The final window and its single posting owner have both published this row.
// Reuse the finalized positive-cutoff winners. Ineligible queries retain the
// full suffix/zero-cutoff selection in the four-owner terminal.
__aicore__ inline uint32_t StagePublishedLargeMisses(uint32_t query, uint32_t scan, uint32_t valid,
                                                  __gm__ uint32_t* destination) {
    auto meta = IncrementalMeta(query);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta), 128);
    if (!large_li::CanStagePublishedMisses(valid, scan, meta[0], uint32_t(meta[1]), uint16_t(meta[2]))) return 0;
    auto ctx = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    auto covered = reinterpret_cast<__gm__ uint32_t*>(kvip_prefetch_rank::CoveredAddress(
        ctx->incremental, runtimeShape_.batch, runtimeShape_.history, query / 4));
    auto compact = incrementalNew_.template ReinterpretCast<uint32_t>();
    LoadPublishedLargeIds(query);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
    auto count = kvip_prefetch_rank::CollectPositiveFinalMisses(indicesOutLocal_, compact, topkSharedTmpLocal_[128],
                                                                covered, valid);
    if (count) {
        GlobalTensor<uint32_t> ids;
        ids.SetGlobalBuffer(destination + query * 2048);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
        DataCopy(ids, compact, (count + 7) / 8 * 8);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID5);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    }
    // 1 already means the LI2 threshold is ready. Preserve zero misses as 2.
    return large_li::kStagedBase + count;
}
