#pragma once
// One launch contains sixteen resident teams. Only the fourth member to leave
// a row assigns its successor. No team waits for a different row to retire.
using terminal_runtime::TerminalRowSchedule;

__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void PrepareTerminalTeams(
    __gm__ TerminalRowSchedule* schedule, __gm__ rotating::Context* ctx, uint32_t batch) {
    const uint32_t row = threadIdx.x;
    const auto ends = reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds);
    uint32_t length = 0;
    if (row < batch) {
        if (ends) {
            for (uint32_t q = 0; q < 4; ++q) {
                const uint32_t end = asc_ldcg(ends + row * 4 + q);
                if (end > length) length = end;
            }
        } else {
            length = ctx->shape.history;
        }
    }
    const uint32_t firstLength = asc_shfl(length, 0);
    const bool varied = asc_ballot(row < batch && length != firstLength) != 0;
    // Prefetch admissions already require Index updates. They are an observable
    // prefix of tail work before final TopK has computed the exact misses.
    uint32_t admitted = 0;
    if (varied && row < batch && length) {
        auto rows = reinterpret_cast<__gm__ rotating::Row*>(ctx->rows);
        admitted = asc_ldcg(&rows[row].slots);
    }
    uint32_t rank = row;
    if (varied) {
        rank = 0;
        for (uint32_t i = 0; i < batch; ++i) {
            const uint32_t n = asc_shfl(admitted, i);
            const uint32_t l = asc_shfl(length, i);
            if (n > admitted || (n == admitted && (l > length || (l == length && i < row)))) ++rank;
        }
    }
    if (row < batch) asc_stcg(schedule->order + rank, row);
    asc_stcg(schedule->arrivals + row, 0U);
    asc_stcg(schedule->successor + row, 0U);
    if (!row) {
        asc_stcg(&schedule->next, 16U);
        asc_stcg(&schedule->finishedTeams, 0U);
    }
    asc_threadfence();
    asc_syncthreads();
    if (!row) { asc_stcg(&schedule->ready, 1U); asc_threadfence(); }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void AssignTerminalSuccessor(
    __gm__ TerminalRowSchedule* schedule, uint32_t row, uint32_t batch) {
    if (!threadIdx.x) {
        asc_threadfence();
        if (asc_atomic_add(schedule->arrivals + row, 1U) == 3) {
            const uint32_t rank = asc_atomic_add(&schedule->next, 1U);
            const uint32_t next = rank < batch ? asc_ldcg(schedule->order + rank) : batch;
            asc_stcg(schedule->successor + row, next + 1);
            asc_threadfence();
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void RetireTerminalTeam(
    __gm__ TerminalRowSchedule* schedule, __gm__ pipeline_ab::Args* a, __gm__ rotating::Context* ctx) {
    uint32_t retired = 0;
    if (!threadIdx.x) retired = asc_atomic_add(&schedule->finishedTeams, 1U);
    if (asc_shfl(retired, 0) != 15) return;
    // The last team reaches here after all rows' four CQ and Index owners
    // have retired. Each saved event is copied and cleared exactly once.
    if (a->completedEvents) {
        const uint32_t words = ctx->shape.rounds * ctx->shape.serviceLanes * (sizeof(rotating::Event)/4);
        auto live = reinterpret_cast<__gm__ uint32_t*>(ctx->events);
        auto saved = reinterpret_cast<__gm__ uint32_t*>(a->completedEvents);
        for (uint32_t i = threadIdx.x; i < words; i += 32) {
            asc_stcg(saved + i, asc_ldcg(live + i));
            asc_stcg(live + i, 0U);
        }
    }
    asc_threadfence();
    asc_syncthreads();
    if (!threadIdx.x) { asc_stcg(&schedule->ready, 0U); asc_threadfence(); }
}

__aicore__ inline uint32_t LoadScheduleWord(__gm__ uint32_t* word) {
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(word), sizeof(*word));
    return *reinterpret_cast<__gm__ volatile uint32_t*>(word);
}

__aicore__ inline void RunTerminalTeams(GM_ADDR binding, uint32_t team, uint32_t q) {
    auto runtime = reinterpret_cast<__gm__ terminal_runtime::Binding*>(binding);
    auto schedule = reinterpret_cast<__gm__ TerminalRowSchedule*>(runtime->reserved[1]);
    auto a = reinterpret_cast<__gm__ pipeline_ab::Args*>(runtime->pipeline);
    auto ctx = reinterpret_cast<__gm__ rotating::Context*>(a->context);
    const uint32_t batch = ctx->shape.batch;
    if (!team && !q) {
        Simt::VF_CALL<PrepareTerminalTeams>(Simt::Dim3{32,1,1}, schedule, ctx, batch);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    while (!LoadScheduleWord(&schedule->ready)) {}
    uint32_t row = LoadScheduleWord(schedule->order + team);
    while (row < batch) {
        RunTerminalRow<true>(binding, row, q);
        Simt::VF_CALL<AssignTerminalSuccessor>(Simt::Dim3{32,1,1}, schedule, row, batch);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        uint32_t next;
        do { next = LoadScheduleWord(schedule->successor + row); } while (!next);
        row = next - 1;
    }
    if (!q) {
        Simt::VF_CALL<RetireTerminalTeam>(Simt::Dim3{32,1,1}, schedule, a, ctx);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
}
