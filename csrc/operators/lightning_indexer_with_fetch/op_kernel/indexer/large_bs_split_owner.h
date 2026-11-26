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
            for (uint32_t epoch = 0; epoch < input.shape.rounds; ++epoch) {
                const uint32_t end = epoch ? large_li::Window(plan.start, plan.end, input.shape.rounds, epoch).end : plan.start;
                if (epoch) {
                    input.Wait(plan.row, 5, epoch);
                }
                PublishProgress(input, plan.row, 0, 4, end);
                op.vectorService.AllocEventID();
                op.vectorService.RotateFetch(epoch, 0);
                op.vectorService.FreeEventID();
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
            pipe.Destroy();
            return;
        }
    }
    bool loaded = false;
    for (uint32_t epoch = 1; epoch < input.shape.rounds; ++epoch) {
        const auto span = large_li::Window(plan.start, plan.end, input.shape.rounds, epoch);
        if (span.end > span.begin) {
            op.ProcessLargeScore(span.begin, span.end, true, !loaded, false);
            loaded = true;
        }
        if ASCEND_IS_AIV {
            input.Signal(plan.row, 5, epoch);
        }
    }
    pipe.Destroy();
}
