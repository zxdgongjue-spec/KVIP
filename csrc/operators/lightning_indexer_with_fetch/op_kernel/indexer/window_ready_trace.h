// Diagnostic-only QLIVector members. Each AIV owns one 128-byte record in
// the second half of its rank-progress slot. The ordinary rank trace uses the
// first half. No publications, polling, cross-core barriers, or allocations.
bool readyTrace_ = false;
uint64_t readyScoreEnd_ = 0;
uint64_t readyFreezeBegin_ = 0, readyFreezeEnd_ = 0;
uint64_t readyQuery_[4][3] = {};
__aicore__ inline void InitReadyTrace() {
    readyTrace_ = streamingWorkspace_ && runtimeShape_.rounds == 4 &&
        runtimeShape_.batch == runtimeShape_.serviceLanes &&
        uint32_t(blockId_) < runtimeShape_.batch * 4 &&
        (reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_)->prefetchCount & 0x80000000U);
}
__aicore__ inline void MarkColdScoreEnd() {
    if (readyTrace_) readyScoreEnd_ = GetSystemCycle();
}
__aicore__ inline void WriteReadyTrace() {
    if (!readyTrace_) return;
    // Join the scalar clock to the final score DMA and vector operations.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID5);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID5);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID5);
    SetFlag<HardEvent::V_S>(EVENT_ID5);
    WaitFlag<HardEvent::V_S>(EVENT_ID5);
    const uint64_t safe = GetSystemCycle();
    auto ctx = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    const uint64_t base = runtimeShape_.batch * (512 + 4 * 2048 * 6 + 1024 +
        (runtimeShape_.history + 255) / 256 * 32);
    auto record = reinterpret_cast<__gm__ uint64_t*>(ctx->incremental + base) + blockId_ * 32 + 16;
    record[0] = readyScoreEnd_;
    record[1] = safe;
    record[2] = readyFreezeBegin_;
    record[3] = readyFreezeEnd_;
    for (uint32_t q = 0; q < 4; ++q) {
        record[4 + 3*q] = readyQuery_[q][0];
        record[5 + 3*q] = readyQuery_[q][1];
        record[6 + 3*q] = readyQuery_[q][2];
    }
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(record), 128);
}
