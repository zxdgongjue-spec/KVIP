// Included in QLIVector only for the one-request-per-cube large-batch kernel.
// A vector lane owns two disjoint 128-byte TopK headers and two winner lists.
// The original single writer still consumes all four lists and updates covered.
__aicore__ inline void AdvanceLargeWindowPair(uint32_t row, uint32_t epoch, uint32_t ready
#if KVIP_TEST_SIBLING_TRACE
                                              , uint64_t scoreBegin, uint64_t scoreEnd
#elif KVIP_BS32_SHARE_FINAL && KVIP_TEST_SPLIT_ROLE_TRACE
                                              , uint64_t* rankBegin
#endif
) {
    auto ctx = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    const uint32_t first = (blockId_ % 2) * 2;
    auto events = reinterpret_cast<__gm__ rotating::Event*>(ctx->events);
    auto previous = events + (epoch - 1) * runtimeShape_.serviceLanes + row;
    const bool trace = streamingWorkspace_ &&
        (reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_)->prefetchCount & 0x80000000U);
#if KVIP_BS32_SHARE_FINAL && KVIP_TEST_SPLIT_ROLE_TRACE
    // The replay uses this separate image for all shared-final diagnostics.
    if (trace && first) {  // V1 owns the public timestamps and Post.
#else
    if (trace && !first) {
#endif
        events[epoch * runtimeShape_.serviceLanes + row].waitBegin = GetSystemCycle();
    }
#if KVIP_TEST_SIBLING_TRACE
    const uint64_t siblingArrival = first ? GetSystemCycle() : 0;
#endif
    do {
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(previous), 128);
    } while (!reinterpret_cast<__gm__ volatile rotating::Event*>(previous)->complete);

    const uint64_t begin = trace ? GetSystemCycle() : 0;
#if KVIP_BS32_SHARE_FINAL && KVIP_TEST_SPLIT_ROLE_TRACE
    *rankBegin = begin;
#endif
#if KVIP_TEST_SIBLING_TRACE
    if (trace) {
        // Two cold epochs, two separate cache lines. Store only after all
        // clocks so publishing a stamp cannot lengthen the measured wait.
        auto pipeline = reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_);
        auto line = reinterpret_cast<__gm__ uint64_t*>(pipeline->times)
                  + runtimeShape_.batch * 64 + (row * 4 + (first ? 3 : 0)) * 16;
        const uint32_t slot = 4 + 4 * (epoch - 1);
        line[slot] = scoreBegin;
        line[slot + 1] = scoreEnd;
        if (first) {
            line[slot + 2] = siblingArrival;
            line[slot + 3] = begin;
        }
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(line), 128);
    }
#endif
    GlobalTensor<uint64_t> header;
    header.SetGlobalBuffer(IncrementalMeta(row * 4 + first));
    GlobalTensor<uint32_t> rows;
    rows.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rows) + row * sizeof(rotating::Row) / 4);
#if defined(KVIP_TEST_WINDOW_RANK1)
    // The previous window publishes its per-row count before complete.
    // Load that count alongside requests, in the same DMA as the row header.
    static_assert(offsetof(rotating::Row, epochCount) / sizeof(uint32_t) + 1 < 24);
    DataCopy(incrementalOld_, rows, 24);
#else
    DataCopy(incrementalOld_, rows, 16);
#endif
    DataCopy(incrementalMetadata_, header, 32);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID5);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID5);
#if KVIP_LARGE_FOUR_WINDOW_CREDIT
    const uint32_t used = incrementalOld_.GetValue(offsetof(rotating::Row, requests) / sizeof(uint32_t));
    windowCoveredEmpty_ = used == 0;
    windowQueryGrant_ = large_li::WindowGrant(runtimeShape_.rounds, epoch, ctx->windowLimit,
                                               ctx->prefetchLimit, used) / 4;
#else
    windowCoveredEmpty_ = incrementalOld_.GetValue(offsetof(rotating::Row, requests) / sizeof(uint32_t)) == 0;
#endif
#if defined(KVIP_TEST_WINDOW_RANK1)
    if (epoch == 2) {
        const uint32_t previousPosts =
            incrementalOld_.GetValue(offsetof(rotating::Row, epochCount) / sizeof(uint32_t) + 1);
        // Let a busy Q nominate more if the preceding 256-token row grant
        // was at most half used. The posting owner still admits at most 256.
        widenThirdWindow_ = ctx->windowLimit == 256 && previousPosts <= 128;
    }
#endif
    incrementalMetadataActive_ = true;
    exportWindowCold_ = true;
    rankedEpoch_ = epoch;
    rankTrace_ = trace;
    incrementalCalls_ = incrementalScanned_ = incrementalCandidates_ = 0;
    const uint32_t end = Min(ready, rotating::Trigger(runtimeShape_.history, runtimeShape_.resident,
                                                       runtimeShape_.rounds, epoch) * 128);
    rankHalf_ = 0;
    rankReady_ = ready;
    for (uint32_t i = 0; i < 2; ++i) {
        incrementalMetadata_.SetValue(i * 16 + 4, ready);
        incrementalMetadataIndex_ = i;
        incrementalHasNext_ = incrementalPipeline_ && i == 0;
        incrementalNextIndex_ = i + 1;
        incrementalNextQuery_ = row * 4 + first + i + 1;
        incrementalNextEnd0_ = end;
        incrementalNextEnd1_ = 0;
        AdvanceIncremental(row * 4 + first + i, end, 0, 0);
    }
    DrainWindowCopies();
    SetFlag<HardEvent::S_MTE3>(EVENT_ID5);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID5);
    DataCopy(header, incrementalMetadata_, 32);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID5);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    incrementalMetadataActive_ = false;
    exportWindowCold_ = false;
#if defined(KVIP_TEST_WINDOW_RANK1)
    widenThirdWindow_ = false;
#endif
#if KVIP_BS32_SHARE_FINAL && KVIP_TEST_SPLIT_ROLE_TRACE
    if (trace && first) {
#else
    if (trace && !first) {
#endif
        auto stamp = reinterpret_cast<__gm__ uint64_t*>(events[epoch * runtimeShape_.serviceLanes + row].pad);
        stamp[2] = begin;
    }
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
