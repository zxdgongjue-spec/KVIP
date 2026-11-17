// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace pipeline_ab {
namespace low_q4_outputs_mte {

// Selected IDs and slots are contiguous within a query. Only the logical-ID
// lookup is indirect; retain that gather in SIMT, with all other words in UB.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void GatherLogical(
    __gm__ int64_t* logical, uint64_t logicalBase,
    uint32_t history, uint32_t slotStride,
    __ubuf__ uint32_t* selected, __ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* output, __ubuf__ uint32_t* error) {
    if (!threadIdx.x) error[0] = 0;
    asc_syncthreads();
    for (uint32_t i = threadIdx.x; i < rotating::kTop; i += 256) {
        const uint32_t token = selected[i];
        uint32_t id = UINT32_MAX;
        if (token == UINT32_MAX) {
            slots[i] = UINT32_MAX;
        } else if (token >= history || slots[i] >= slotStride) {
            slots[i] = UINT32_MAX;
            asc_atomic_or(error, 1U);
        } else {
            id = uint32_t(asc_ldcg(logical + logicalBase + token));
        }
        output[i] = id;
    }
    asc_syncthreads();
}

// One AIV owns one TND output row. This runs before the existing pre-Post
// collective join, so logical IDs cannot change while an output is gathered.
// No additional global synchronization or CPU work is introduced.
__aicore__ inline uint32_t Publish(
    __gm__ rotating::Context* ctx, __gm__ Args* args, uint32_t block) {
    const uint32_t batch = ctx->shape.batch;
    if (block >= batch * 4) return 0;
    auto update = reinterpret_cast<__gm__ uint64_t*>(args->updateBindings);
    if (!update || !update[20] || !update[21] || !update[22] || !update[23]) return 0;
    auto valid = reinterpret_cast<__gm__ int64_t*>(update[21]);
    const bool active = valid[block] != 0;
    uint64_t source = 0, logicalBase = 0;
    auto logical = reinterpret_cast<__gm__ int64_t*>(update[3]);
    if (active) {
        auto unpack = reinterpret_cast<__gm__ int64_t*>(update[20]);
        source = uint64_t(unpack[block]);
        auto owners = reinterpret_cast<__gm__ uint32_t*>(update[9]);
        if (source >= uint64_t(batch) * 4 || !logical || !owners || !update[24]) return 1;
        logicalBase = uint64_t(owners[source / 4]) * uint32_t(update[24]);
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 3 * rotating::kTop * sizeof(uint32_t) + 64);
    auto local = storage.Get<uint32_t>();
    auto selected = local;
    auto slots = local[rotating::kTop];
    auto output = local[2 * rotating::kTop];
    auto error = local[3 * rotating::kTop];
    uint32_t code = 0;
    if (active) {
        AscendC::GlobalTensor<uint32_t> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected) + source * rotating::kTop);
        AscendC::DataCopy(selected, input, rotating::kTop);
        input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations) + source * rotating::kTop);
        AscendC::DataCopy(slots, input, rotating::kTop);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        Simt::VF_CALL<GatherLogical>(Simt::Dim3{256, 1, 1}, logical, logicalBase,
            ctx->shape.history, ctx->outputStride ? uint32_t(ctx->outputStride) : ctx->shape.resident,
            reinterpret_cast<__ubuf__ uint32_t*>(selected.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(slots.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(output.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(error.GetPhyAddr()));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        code = error.GetValue(0);
    } else {
        // An inactive output never reads its unspecified unpack entry.
        AscendC::Duplicate(output, UINT32_MAX, rotating::kTop);
        slots = output;
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::GlobalTensor<uint32_t> destination;
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(update[22]) + uint64_t(block) * rotating::kTop);
    AscendC::DataCopy(destination, output, rotating::kTop);
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(update[23]) + uint64_t(block) * rotating::kTop);
    AscendC::DataCopy(destination, slots, rotating::kTop);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    pipe.Destroy();
    return code;
}

}  // namespace low_q4_outputs_mte
}  // namespace pipeline_ab
