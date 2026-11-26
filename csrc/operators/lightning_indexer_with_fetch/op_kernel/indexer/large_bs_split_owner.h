#if KVIP_TEST_SPLIT_ROLE_TRACE
// The three-kernel large-BS bridge never runs the former standalone streaming
// producer/consumer. Its first 64 pipeline timestamp words per row are free.
// Give V0 and V1 distinct, aligned cache lines, and flush once after service.
template <uint32_t N>
__aicore__ inline void SaveSplitRoleClocks(const Inputs& input, uint32_t row, uint32_t lane,
                                            const uint64_t (&values)[N]) {
    static_assert(N <= 16, "Each role has one 128-byte cache line");
    auto line = reinterpret_cast<__gm__ uint64_t*>(input.stream->times) + row * 64 + lane * 16;
    for (uint32_t i = 0; i < N; ++i) line[i] = values[i];
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(line), 128);
}
#endif

// The dedicated BS32 image assigns four queries to V0 for scoring and V1 for window service.
__aicore__ inline void ColdOwner(const Inputs& input, const RowPlan& plan) {
    JoinThreshold(input, plan.row);
    TPipe pipe;
    Indexer op;
    input.Init(op, pipe, plan.row, false);
    if ASCEND_IS_AIV {
        if (input.vecLane == 1) {
            if (input.Trace()) {
                uint64_t start = GetSystemCycle();
                for (uint32_t q = 0; q < 4; ++q) {
                    input.Stamp(plan.row, q, 2, start);
                }
            }
#if KVIP_TEST_SPLIT_ROLE_TRACE
            uint64_t serviceClocks[12]{};
#endif
            for (uint32_t epoch = 0; epoch < input.shape.rounds; ++epoch) {
                const uint32_t end = epoch ? large_li::Window(plan.start, plan.end, input.shape.rounds, epoch).end : plan.start;
#if KVIP_TEST_SPLIT_ROLE_TRACE
                serviceClocks[3 * epoch] = GetSystemCycle();
#endif
                if (epoch) {
                    input.Wait(plan.row, 5, epoch);
                }
#if KVIP_TEST_SPLIT_ROLE_TRACE
                serviceClocks[3 * epoch + 1] = epoch ? GetSystemCycle() : serviceClocks[0];
#endif
                PublishProgress(input, plan.row, 0, 4, end);
                op.vectorService.AllocEventID();
                op.vectorService.RotateFetch(epoch, 0);
                op.vectorService.FreeEventID();
#if KVIP_TEST_SPLIT_ROLE_TRACE
                serviceClocks[3 * epoch + 2] = GetSystemCycle();
#endif
            }
            if (input.stagedMisses) {
                // Probe once; a late row, an outstanding own Post or an ineligible Q
                // simply defers its exact classification to its original tail owner.
                auto last = reinterpret_cast<__gm__ rotating::Event*>(input.ctx->events) +
                            (input.shape.rounds - 1) * input.shape.serviceLanes;
                const uint32_t peer = (plan.row + input.shape.batch / 2) % input.shape.batch;
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(last + peer), 128);
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(last + plan.row), 128);
                auto events = reinterpret_cast<__gm__ volatile rotating::Event*>(last);
                if (!events[peer].complete && events[plan.row].complete) {
                    auto ends = reinterpret_cast<__gm__ const uint32_t*>(input.ctx->queryEnds);
                    const uint32_t scan = large_li::ScanHistory(input.shape, ends, plan.row);
                    op.vectorService.AllocEventID();
                    for (uint32_t q = 0; q < 4; ++q) {
                        const uint32_t query = plan.row * 4 + q;
                        const uint32_t valid = ends ? Min(ends[query], input.shape.history) : input.shape.history;
                        const uint32_t staged = op.vectorService.StagePublishedLargeMisses(
                            query, scan, valid, input.stagedMisses);
                        if (staged) input.Signal(plan.row, q, staged);
                    }
                    op.vectorService.FreeEventID();
                }
            }

            if (input.Trace()) {
                uint64_t end = GetSystemCycle();
                for (uint32_t q = 0; q < 4; ++q) {
                    input.Stamp(plan.row, q, 3, end);
                }
            }
#if KVIP_TEST_SPLIT_ROLE_TRACE
            if (input.Trace()) SaveSplitRoleClocks(input, plan.row, 1, serviceClocks);
#endif
            pipe.Destroy();
            return;
        }
    }
#if KVIP_TEST_SPLIT_ROLE_TRACE
    uint64_t scoreClocks[6]{};
#endif
    bool loaded = false;
    for (uint32_t epoch = 1; epoch < input.shape.rounds; ++epoch) {
        const auto span = large_li::Window(plan.start, plan.end, input.shape.rounds, epoch);
#if KVIP_TEST_SPLIT_ROLE_TRACE
        if ASCEND_IS_AIV { scoreClocks[2 * (epoch - 1)] = GetSystemCycle(); }
#endif
        if (span.end > span.begin) {
            op.ProcessLargeScore(span.begin, span.end, true, !loaded, false);
            loaded = true;
        }
        if ASCEND_IS_AIV {
#if KVIP_TEST_SPLIT_ROLE_TRACE
            scoreClocks[2 * (epoch - 1) + 1] = GetSystemCycle();
#endif
            input.Signal(plan.row, 5, epoch);
        }
    }
#if KVIP_TEST_SPLIT_ROLE_TRACE
    if ASCEND_IS_AIV { if (input.Trace()) SaveSplitRoleClocks(input, plan.row, 0, scoreClocks); }
#endif
    pipe.Destroy();
}
