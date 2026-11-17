// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace pipeline_ab {
namespace post_index_finalize {

// No prefetch was issued. Every low-plan change is already a final Get
// destination, including invalid-resident self-refills. Publish those exact
// destinations once; the legacy changes[0:2048] residency scan is redundant.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void PublishState(__gm__ rotating::Context* ctx, __gm__ uint64_t* update,
                         uint32_t batch, uint32_t resident, uint32_t postOffset,
                         __gm__ uint32_t* eventDestinations, uint32_t count,
                         uint32_t postError, uint64_t finalError) {
    if (!finalError && ctx->hotValid) {
        const uint32_t owner = OwnerRow(ctx, batch);
        auto valid = reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid) + uint64_t(owner) * resident;
        for (uint32_t i = threadIdx.x; i < count; i += 256)
            asc_stcg(valid + asc_ldcg(eventDestinations + i), 1U);
    }
    if (finalError) {
        // Match CompleteEarlyUpdate: failures invalidate the stable owner and
        // never make a partially written Get visible as a successful hot hit.
        const uint32_t owner = asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(update[9]) + batch);
        const uint32_t invalidResident = uint32_t(update[25]);
        auto valid = reinterpret_cast<__gm__ uint32_t*>(update[16]) + uint64_t(owner) * invalidResident;
        for (uint32_t i = threadIdx.x; i < invalidResident; i += 256) asc_stcg(valid + i, 0U);
    }
    if (threadIdx.x < 4) {
        auto post = reinterpret_cast<__gm__ FinalPost*>(ctx->owners + postOffset) + batch * 4 + threadIdx.x;
        // Exact planning publishes Q counts in rowStatus once. Materialize
        // the legacy FinalPost ABI here, avoiding four earlier scalar stores.
        // Early input rejection never prepared a plan; preserve its init state.
        auto info = reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus) + batch * kRowStatusWords;
        if (asc_ldcg(info + 9)) {
            const uint32_t count = asc_ldcg(info + 5 + threadIdx.x);
            asc_stcg(&post->count, count);
            asc_stcg(&post->nonconflicting, count);
            asc_stcg(&post->descriptor, batch * 8192 + threadIdx.x * 2048);
            asc_stcg(&post->head, uint64_t(0));
        }
        // Write the final state directly. No intermediate pending=count state
        // is published only to be reread and immediately cleared by Finalize.
        asc_stcg(&post->pending, 0U);
        asc_stcg(&post->deferred, 0U);
        asc_stcg(&post->error, postError);
    }
    if (!threadIdx.x)
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(update[6]) + batch,
                 postError | (finalError ? 4U : 0U));
    asc_threadfence();
}

// Called only after all queues were drained and the existing row completion
// reduction checked completed==rowStatus[1]. Transport errors remain in the
// per-core records; publish exactly the previous 32-bit FinalPost/status ABI.
__aicore__ inline void Finish(__gm__ Args* args, uint32_t batch, uint32_t total, uint64_t transportError) {
    auto ctx = reinterpret_cast<__gm__ rotating::Context*>(args->context);
    const auto shape = LoadRuntimeShape(ctx);
    auto row = reinterpret_cast<__gm__ Row*>(ctx->rows) + batch;
    auto info = reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus) + batch * kRowStatusWords;
    auto update = reinterpret_cast<__gm__ uint64_t*>(args->updateBindings);
    const bool profile = (args->prefetchCount & 0x80000000U) != 0;
    auto ticks = reinterpret_cast<__gm__ uint64_t*>(args->times) +
                 uint64_t(shape.batch) * 128 + uint64_t(batch) * 128 + 96;
    if (profile) ticks[0] = GetSystemCycle();
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info), 128);
    const uint32_t postError = uint32_t(transportError) ? uint32_t(transportError) :
                               (transportError ? kvaf_device::kInvalidArgs : 0U);
    // Counts were acquired and checked before Post and remain immutable.
    // Reuse their raw sum, including error cases, rather than read four post
    // records again after the CQ join.
    // Preserve the old final status on late structural errors: successful
    // transport completion was recorded before the pending/count checks.
    const uint32_t completed = transportError ? 0 : total;
    uint64_t error = row->error | uint64_t(postError);
    if (info[0] || row->pending || total != info[1]) error |= kvaf_device::kInvalidArgs;
    if (profile) ticks[1] = GetSystemCycle();
    Simt::VF_CALL<PublishState>(Simt::Dim3{256, 1, 1}, ctx, update, batch,
        shape.resident, shape.finalPostOffset,
        reinterpret_cast<__gm__ uint32_t*>(args->eventsDst) + uint64_t(batch) * args->maxEvents + kPrefetch,
        total, postError, error);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    if (profile) ticks[2] = GetSystemCycle();
    info[0] |= uint32_t(error);
    info[3] = 0;
    info[4] = completed;
    if (!error) row->finalPending = 0;
    auto status = reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(ctx->finalStatus) + batch * 2;
    kvaf_device::WriteStatus(status, error, total, completed, 0, 0, UINT64_MAX, 0);
    kvaf_device::WriteStatus(status + 1, 0, 0, 0, 0, 0, UINT64_MAX, 0);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status), 2 * sizeof(*status));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info), 128);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row) + 960, 64);
#if KVIP_AUTO_ROUTE
    if (ctx->autoRoute) {
        auto config = reinterpret_cast<__gm__ kvip_auto_route::Config*>(ctx->autoRoute);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(config), 128);
        if (kvip_auto_route::Valid(config)) {
            auto pending = reinterpret_cast<__gm__ kvip_auto_route::Pending*>(config->pending) + batch;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(pending), 128);
            pending->usefulPrefetch = 0;
            pending->necessary = info[1];
            if (pending->owner < config->owners) {
                auto feedback = reinterpret_cast<__gm__ uint8_t*>(config->feedback) + pending->owner * 128;
                kvaf_device::CleanCacheLines(feedback, 128);
                if (kvip_auto_route::Commit(config, batch, error | postError))
                    kvaf_device::CleanCacheLines(feedback, 128);
            }
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(pending), 128);
        }
    }
#endif
    if (profile) {
        ticks[3] = GetSystemCycle();
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks), 32 * sizeof(uint64_t));
    }
}

}  // namespace post_index_finalize
}  // namespace pipeline_ab
