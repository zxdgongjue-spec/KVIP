// Included in QLIVector only for the one-request-per-cube large-batch kernel.
// A vector lane owns two disjoint 128-byte TopK headers and two winner lists.
// The original single writer still consumes all four lists and updates covered.
__aicore__ inline void AdvanceLargeWindowPair(uint32_t row, uint32_t epoch, uint32_t ready) {
    auto ctx = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    const uint32_t first = (blockId_ % 2) * 2;
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
    DataCopy(incrementalMetadata_, header, 32);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID5);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID5);
    windowCoveredEmpty_ = incrementalOld_.GetValue(offsetof(rotating::Row, requests) / sizeof(uint32_t)) == 0;
    incrementalMetadataActive_ = true;
    exportWindowCold_ = true;
    rankedEpoch_ = epoch;
    rankTrace_ = trace;
    incrementalCalls_ = incrementalScanned_ = incrementalCandidates_ = 0;
#if KVIP_LARGE_SPLIT_MIDDLE
    // The owner passed its row's physical tile end after joining both producers.
    const uint32_t end = ready;
#else
    const uint32_t end = Min(ready, rotating::Trigger(runtimeShape_.history, runtimeShape_.resident,
                                                       runtimeShape_.rounds, epoch) * 128);
#endif
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
    if (trace && !first) {
        auto stamp = reinterpret_cast<__gm__ uint64_t*>(events[epoch * runtimeShape_.serviceLanes + row].pad);
        stamp[2] = begin;
    }
}
