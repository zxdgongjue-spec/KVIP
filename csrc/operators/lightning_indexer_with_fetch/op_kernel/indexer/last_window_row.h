// SPDX-License-Identifier: Apache-2.0
// Members of QLIVector. One W0 scorer offers to its own service row.
// W3 decides the assignment. Its Q0/Q1 input and output remain private.
bool lastRowTeam_ = false;
enum : uint64_t { LastUndecided = 0, LastDelegated = 1, LastRetained = 2 };
struct alignas(64) LastRowJob {
    uint64_t decision, ready2, ready3, coveredEmpty, decisionQ, pad[3];
};
struct alignas(64) LastRowOffer {
    uint64_t available, complete, pad[6];
};
static_assert(sizeof(LastRowJob) == 64 && sizeof(LastRowOffer) == 64);

__aicore__ inline bool HasLastRowTeam() const {
    const auto& s = runtimeShape_;
    if (!streamingWorkspace_ || !s.splitQ || s.rounds != 4 || s.batch != s.serviceLanes ||
        s.serviceLanes != 2 * s.serviceAics || s.liCores != 4 * s.serviceAics ||
        s.liCores > s.aicCount) return false;
    const auto args = reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_);
    return !(args->prefetchCount & 0x80000000U);
}
// The upper half of each progress line is unused with four windows and tracing off.
__aicore__ inline __gm__ volatile LastRowJob* LastJob(uint32_t row) const {
    return reinterpret_cast<__gm__ volatile LastRowJob*>(RankProgress(row * 4 + 2) + 16);
}
__aicore__ inline __gm__ volatile LastRowOffer* LastOffer(uint32_t row) const {
    return reinterpret_cast<__gm__ volatile LastRowOffer*>(RankProgress(row * 4 + 3) + 16);
}
template<class T> __aicore__ inline void FlushLastLine(__gm__ volatile T* line) const {
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ T*>(line)),64);
}
__aicore__ inline void ResetLastRow(uint32_t row) {
    if (!HasLastRowTeam()) return;
    auto job = LastJob(row);
    auto offer = LastOffer(row);
    job->decision = LastUndecided;
    offer->available = 0;
    offer->complete = 0;
    FlushLastLine(job);
    FlushLastLine(offer);
}
// Exactly one owner writes the decision, and exactly one helper writes
// available/complete. They live on physically separate rank-progress lines.
// A Q cannot have two writers, even when the offer arrives at the deadline.
__aicore__ inline bool OfferLastPair(uint32_t row, uint32_t ownerQ, bool coveredEmpty) {
    auto offer = LastOffer(row);
    FlushLastLine(offer);
    auto job = LastJob(row);
    const uint64_t r2 = incrementalMetadata_.GetValue(2 * 16 + 4);
    const uint64_t r3 = incrementalMetadata_.GetValue(3 * 16 + 4);
    if (offer->available && uint32_t(r2) >= runtimeShape_.resident &&
        uint32_t(r3) >= runtimeShape_.resident && incrementalMetadata_.GetValue(2 * 16) &&
        incrementalMetadata_.GetValue(3 * 16)) {
        job->ready2 = r2;
        job->ready3 = r3;
        job->coveredEmpty = coveredEmpty;
        job->decisionQ = ownerQ;
        FlushLastLine(job);
        job->decision = LastDelegated;
        FlushLastLine(job);
        return true;
    }
    if (ownerQ == 1) {
        job->decisionQ = ownerQ;
        job->decision = LastRetained;
        FlushLastLine(job);
    }
    return false;
}
__aicore__ inline void JoinLastPair(uint32_t row) {
    auto offer = LastOffer(row);
    do { FlushLastLine(offer); } while (!offer->complete);
}
__aicore__ inline void HelpLastPair(uint32_t row, uint32_t half) {
    // Hand the scalar a fully drained score/bitmap/progress pipeline before
    // advertising the scratch banks. Other rows do not depend on this worker.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID5); WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID5); WaitFlag<HardEvent::MTE2_S>(EVENT_ID5);
    SetFlag<HardEvent::V_S>(EVENT_ID5); WaitFlag<HardEvent::V_S>(EVENT_ID5);
    auto job = LastJob(row);
    FlushLastLine(job);
    if (job->decision == LastRetained) return;
    auto offer = LastOffer(row);
    offer->available = 1;
    FlushLastLine(offer);
    uint64_t decision;
    for (;;) {
        const uint64_t lastRead = GetSystemCycle();
        FlushLastLine(job);
        decision = job->decision;
        if (decision != LastUndecided) break;
        // Give the scoring and earlier-window DMAs time to run. Recheck
        // after at least 1024 cycles from the previous read.
        while (GetSystemCycle() - lastRead < 1024U) {}
    }
    if (decision != LastDelegated) return;

    const uint64_t ready[2] = {job->ready2, job->ready3};
    const uint32_t epoch = runtimeShape_.rounds - 1;
    const uint32_t cap = rotating::SequenceWindowEnd(runtimeShape_.history, runtimeShape_.resident, half, epoch);
    GlobalTensor<uint64_t> headers;
    headers.SetGlobalBuffer(IncrementalMeta(row * 4 + 2));
    DataCopy(incrementalMetadata_, headers, 32);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID5); WaitFlag<HardEvent::MTE2_S>(EVENT_ID5);
    incrementalMetadataActive_ = true;
    exportWindowCold_ = true;
    rankTrace_ = false;
    rankedEpoch_ = epoch;
    rankHalf_ = half;
    windowCoveredEmpty_ = job->coveredEmpty;
    incrementalCalls_ = incrementalScanned_ = incrementalCandidates_ = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        incrementalMetadata_.SetValue(i * 16 + 4, ready[i]);
        incrementalMetadataIndex_ = i;
        rankReady_ = ready[i];
        incrementalHasNext_ = i == 0 && incrementalPipeline_;
        if (incrementalHasNext_) {
            incrementalNextQuery_ = row * 4 + 3;
            incrementalNextIndex_ = 1;
            incrementalNextEnd0_ = Min(uint32_t(ready[1]), cap);
            incrementalNextEnd1_ = Min(uint32_t(ready[1] >> 32), cap);
        }
        AdvanceIncremental(row * 4 + 2 + i, Min(uint32_t(ready[i]), cap),
                           Min(uint32_t(ready[i] >> 32), cap), half);
    }
    FinishWindowTransfers();
    SetFlag<HardEvent::S_MTE3>(EVENT_ID5); WaitFlag<HardEvent::S_MTE3>(EVENT_ID5);
    DataCopy(headers, incrementalMetadata_, 32);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID5); WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    incrementalMetadataActive_ = false;
    exportWindowCold_ = false;
    offer->complete = 1;
    FlushLastLine(offer);
}
