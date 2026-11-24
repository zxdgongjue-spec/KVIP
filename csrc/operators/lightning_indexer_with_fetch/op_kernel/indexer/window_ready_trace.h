// Each vector worker owns the second cache line of its diagnostic record.
bool readyTrace_ = false;
uint32_t coldEpoch_ = 0, coldBegin_ = 0, coldEnd_ = 0, coldMark_ = 0;
uint64_t coldClock_ = 0, firstPoll_ = 0;
uint64_t readyFreezeBegin_ = 0, readyFreezeEnd_ = 0, coldSamples_[12] = {};
__aicore__ inline void InitReadyTrace() {
    readyTrace_ = streamingWorkspace_ && runtimeShape_.rounds == 4 &&
        runtimeShape_.batch == runtimeShape_.serviceLanes &&
        uint32_t(blockId_) < runtimeShape_.batch * 4 &&
        (reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_)->prefetchCount & 0x80000000U);
    if (readyTrace_) coldClock_ = GetSystemCycle();
}
__aicore__ inline void BeginColdSegment(uint32_t epoch, uint32_t begin, uint32_t end) {
    if (!readyTrace_) return;
    coldEpoch_ = epoch;
    coldBegin_ = begin;
    coldEnd_ = end;
    coldMark_ = 0;
}
__aicore__ inline bool SamplePublication(uint32_t endTile) {
    return readyTrace_ && coldEpoch_ && coldMark_ < 4 &&
        endTile >= coldBegin_ + ((coldEnd_ - coldBegin_) * (coldMark_ + 1) + 3) / 4;
}
__aicore__ inline void FinishPublication(uint32_t endTile, uint64_t issued) {
    // Bracket the existing frontier DMA. Both score and hit writes precede
    // that DMA on MTE3. No extra frontier is published for the diagnostic.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID5);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    const uint64_t done = GetSystemCycle();
    const uint64_t offset = issued - coldClock_, latency = done - issued;
    uint64_t sample = ~uint64_t(0);
    if (endTile < 65536 && latency < 65536 && offset <= UINT32_MAX)
        sample = (uint64_t(endTile) << 48) | (latency << 32) | offset;
    do {
        coldSamples_[4 * (coldEpoch_ - 1) + coldMark_++] = sample;
    } while (SamplePublication(endTile));
}
__aicore__ inline void WriteReadyTrace() {
    if (!readyTrace_) return;
    auto ctx = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    const uint64_t base = runtimeShape_.batch * (512 + 4 * 2048 * 6 + 1024 +
        (runtimeShape_.history + 255) / 256 * 32);
    auto record = reinterpret_cast<__gm__ uint64_t*>(ctx->incremental + base) + blockId_ * 32 + 16;
    record[0] = coldClock_;
    record[1] = firstPoll_;
    record[2] = readyFreezeBegin_;
    record[3] = readyFreezeEnd_;
    for (uint32_t i = 0; i < 12; ++i) record[4 + i] = coldSamples_[i];
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(record), 128);
}
