#include "../op_kernel/pipeline_types.h"
#include "api.h"
#include "context.h"
#include "cold_scan.h"
#include "rolling_threshold.h"
extern "C" uint32_t kvip_incremental_window() { return KVIP_INCREMENTAL_WINDOW; }
extern "C" uint32_t kvip_incremental_inplace() { return KVIP_INCREMENTAL_INPLACE; }
extern "C" uint32_t kvip_window_probe() { return KVIP_WINDOW_PROBE; }
extern "C" uint32_t kvip_trunk_pruned() { return KVIP_TRUNK_PRUNED; }
extern "C" uint32_t kvip_native_trunk() { return KVIP_NATIVE_TRUNK; }
extern "C" uint32_t kvip_seed_parallel() { return KVIP_SEED_PARALLEL; }
extern "C" uint32_t kvip_seed_mask() { return KVIP_SEED_MASK; }
extern "C" uint32_t kvip_consumed_prefix() { return KVIP_CONSUMED_PREFIX; }
extern "C" uint32_t kvip_incremental_segments() { return KVIP_INCREMENTAL_SEGMENTS; }
extern "C" uint32_t kvip_incremental_progress() { return KVIP_INCREMENTAL_PROGRESS; }
extern "C" uint32_t kvip_incremental_light() { return KVIP_INCREMENTAL_LIGHT; }
extern "C" uint32_t kvip_incremental_direct() { return KVIP_INCREMENTAL_DIRECT; }
extern "C" uint32_t kvip_incremental_ties() { return KVIP_INCREMENTAL_TIES; }
extern "C" uint32_t kvip_incremental_prefilter() { return KVIP_INCREMENTAL_PREFILTER; }
extern "C" uint32_t kvip_incremental_register() { return KVIP_INCREMENTAL_REGISTER; }
extern "C" uint32_t kvip_incremental_topk() { return KVIP_INCREMENTAL_TOPK; }
extern "C" uint32_t kvip_rolling_boundary() { return KVIP_ROLLING_BOUNDARY; }
extern "C" uint32_t kvip_rolling_sparse() { return KVIP_ROLLING_SPARSE; }
extern "C" uint32_t kvip_rolling_fast_path() { return KVIP_ROLLING_FAST_PATH; }
extern "C" uint32_t kvip_rolling_reuse_seed() { return KVIP_ROLLING_REUSE_SEED; }
extern "C" uint32_t kvip_compact_out_of_place() { return KVIP_COMPACT_OUT_OF_PLACE; }
extern "C" uint32_t kvip_refilter_safe() { return KVIP_REFILTER_SAFE; }
extern "C" uint32_t kvip_refilter_refill() { return KVIP_REFILTER_REFILL; }
extern "C" uint32_t kvip_rolling_compact() { return KVIP_ROLLING_COMPACT; }
extern "C" uint32_t kvip_rolling_first_window() { return KVIP_ROLLING_FIRST_WINDOW; }
extern "C" uint32_t kvip_rolling_windowed() { return KVIP_ROLLING_WINDOWED; }
extern "C" uint32_t kvip_rolling_batched() { return KVIP_ROLLING_BATCHED; }
extern "C" uint32_t kvip_rolling_threshold() { return KVIP_ROLLING_THRESHOLD; }
extern "C" uint32_t kvip_window_refilter() { return KVIP_WINDOW_REFILTER; }
extern "C" uint32_t kvip_reverse_cold() { return KVIP_REVERSE_COLD; }
extern "C" uint32_t kvip_prepare_shape(rotating::RuntimeShape* shape) { return rotating::PrepareShape(shape); }
#define ENTRY(name, blocks) \
extern "C" uint32_t aclrtlaunch_v9_##name##_kernel(uint32_t,void*,void*); \
extern "C" uint32_t v9_##name(void* bindings,void* stream,const rotating::RuntimeShape* shape) { \
    if (!bindings || !stream || !shape) return 100000; \
    return aclrtlaunch_v9_##name##_kernel(blocks,stream,bindings); }
ENTRY(init,shape->batch * (1 + rotating::SeedInitHelpers(*shape)))

extern "C" uint32_t aclrtlaunch_v9_li_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_1(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_2(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_3(uint32_t,void*,void*);
extern "C" uint32_t v9_li(void* bindings, void* stream, const rotating::RuntimeShape* shape) {
#if KVIP_COMPACT_OUT_OF_PLACE
    // cfg[1536:2016] fits the existing split-Q/16K work bank. Other
    // geometries need their own scratch proof before enabling this option.
    if (!shape || !shape->splitQ || shape->batch > 16 || shape->indexHeads != 32 ||
        rotating::UseSequenceSplit(*shape)) return 100000;
#endif
#if KVIP_ROLLING_THRESHOLD
    // The persistent histogram belongs to one Q producer. Do not silently
    // use an unseeded state when a different kernel owns each sequence shard.
    // The refilter's per-row threshold snapshot uses the existing Q2/16K
    // scratch layout. Larger non-split-Q batches have a smaller scratch bank.
    if (!shape || !shape->splitQ || shape->batch > 16 || rotating::UseSequenceSplit(*shape)) return 100000;
#if KVIP_ROLLING_WINDOWED
    if (shape->indexHeads != 32 || shape->rounds < 2 || shape->history < shape->resident ||
        shape->resident % 128 || shape->history % 128 ||
        ((shape->history - shape->resident) / 128 + shape->rounds - 2) / (shape->rounds - 1) * 128 >
            kvip_rolling::kWindowScores) return 100000;
#endif
#endif
#if KVIP_INCREMENTAL_TOPK
    if (!shape || !shape->splitQ || shape->batch > 16 || shape->indexHeads != 32 ||
        rotating::UseSequenceSplit(*shape) || shape->resident != 16384 || shape->rounds != 9 ||
        shape->history <= shape->resident ||
        ((shape->history - shape->resident)/128+7)/8*128 > 65535) return 100001;
#if KVIP_INCREMENTAL_WINDOW
    // Split-Q/H32 uses a 16K trunk. The dead merge scratch is K+16K u16.
    if (shape->history % 128 ||
        ((shape->history-shape->resident)/128+7)/8*128 > 16384) return 100001;
#endif
#endif
#define LI_LAUNCH_0 aclrtlaunch_v9_li_kernel
#define LI_LAUNCH_1 aclrtlaunch_v9_li_kernel_1
#define LI_LAUNCH_2 aclrtlaunch_v9_li_kernel_2
#define LI_LAUNCH_3 aclrtlaunch_v9_li_kernel_3
#include "../../common/lifetch/li_launch.h"
#undef LI_LAUNCH_0
#undef LI_LAUNCH_1
#undef LI_LAUNCH_2
#undef LI_LAUNCH_3
}
extern "C" uint32_t v9_li_cores(const rotating::RuntimeShape* shape) { return shape->liCores; }
extern "C" uint32_t v9_batch(const rotating::RuntimeShape* shape) { return shape->batch; }
extern "C" uint32_t v9_history(const rotating::RuntimeShape* shape) { return shape->history; }
extern "C" uint32_t v9_resident(const rotating::RuntimeShape* shape) { return shape->resident; }
extern "C" uint32_t v9_row_bytes(const rotating::RuntimeShape* shape) { return sizeof(rotating::Row); }
extern "C" uint32_t v9_context_bytes(const rotating::RuntimeShape* shape) { return sizeof(rotating::Context); }
extern "C" uint32_t v9_rounds(const rotating::RuntimeShape* shape) { return shape->rounds; }
extern "C" uint32_t v9_simulation(const rotating::RuntimeShape* shape) { return 0; }
extern "C" uint32_t v9_abi_version(const rotating::RuntimeShape* shape) { return 12 + KVIP_INCREMENTAL_TOPK + 2*KVIP_WINDOW_PROBE + 4*KVIP_SEED_MASK; }
extern "C" uint32_t v9_aic_count(const rotating::RuntimeShape* shape) { return shape->aicCount; }
extern "C" uint32_t v9_service_lanes(const rotating::RuntimeShape* shape) { return shape->serviceLanes; }
extern "C" uint32_t v9_final_workers(const rotating::RuntimeShape* shape) { return shape->finalWorkers; }
extern "C" uint32_t v9_final_owner_bytes(const rotating::RuntimeShape* shape) { return shape->finalOwnerBytes; }
extern "C" uint32_t v9_final_replicated(const rotating::RuntimeShape* shape) { return 1; }
extern "C" uint32_t v9_token_bytes(const rotating::RuntimeShape* shape) { return shape->tokenBytes; }

#include "api.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include "batch_get_abi.h"

// Native header is the source of truth for structure sizes and padding.
extern "C" uint32_t refactor_fetch_args_bytes() { return sizeof(KvafExecuteBatchGetKernelArgs); }
extern "C" uint32_t refactor_fetch_status_bytes() { return sizeof(KvafBatchGetCoreStatus); }
extern "C" uint32_t refactor_fetch_entity_bytes() { return sizeof(KvafBatchGetDirectUrmaEntity); }
extern "C" uint32_t refactor_fetch_queue_bytes() { return sizeof(KvafBatchGetQueueState); }

extern "C" uint32_t refactor_fetch_pack_args(void *output, uint32_t capacity,
                                             const uint64_t *fields, uint32_t count)
{
    if (!output || !fields || (count != 17 && count != 18) || capacity != sizeof(KvafExecuteBatchGetKernelArgs)) return 1;
    for (uint32_t i = 10; i < count; ++i)
        if (fields[i] > std::numeric_limits<uint32_t>::max()) return 2;
    KvafExecuteBatchGetKernelArgs args{};
    args.address_table = fields[0]; args.valid_table = fields[1];
    args.indices = fields[2]; args.offsets = fields[3]; args.miss_count = fields[4];
    args.output_indices = fields[5]; args.output = fields[6]; args.core_status = fields[7];
    args.channel_entity = fields[8]; args.queue_state = fields[9];
    args.table_capacity = fields[10]; args.request_capacity = fields[11];
    args.output_rows = fields[12]; args.token_bytes = fields[13];
    args.tokens_per_block = fields[14]; args.block_count = fields[15];
    args.jettys_per_core = fields[16];
    if (count == 18) args.batch_get_max_inflight_total = fields[17];
    std::memcpy(output, &args, sizeof(args));
    return 0;
}

extern "C" uint32_t aclrtlaunch_v9_streaming_resolve_update(uint32_t,void*,void*);
extern "C" uint32_t kvip_lifetch_streaming_forward(void* bindings,void* update,void* workspace,void* stream,const rotating::RuntimeShape* shape,uint32_t workers) {
    if(!bindings||!update||!workspace||!stream||!shape||shape->resident>131072||shape->batch<1||shape->batch>16||shape->finalWorkers!=4)return 100000;
    if((workers!=1 && workers!=2 && workers!=4) || shape->batch*pipeline_ab::kResolveCores+4*workers>shape->aicCount*2)return 100000;
    uint32_t result=v9_init(bindings,stream,shape);
    if(!result)result=v9_li(bindings,stream,shape);
    if(!result)result=aclrtlaunch_v9_streaming_resolve_update(shape->batch*pipeline_ab::kResolveCores+4*workers,stream,workspace);
    return result;
}

extern "C" uint32_t aclrtlaunch_kvip_transport_snapshot_kernel(uint32_t,void*,void*,void*,uint32_t);
extern "C" uint32_t kvip_transport_snapshot(void* source, void* destination, uint32_t bytes, void* stream) {
    if (!source || !destination || !stream || !bytes || bytes % 32 ||
        reinterpret_cast<uintptr_t>(source) % 32 || bytes > 64U * 1024U * 1024U) return 100000;
    return aclrtlaunch_kvip_transport_snapshot_kernel(1,stream,source,destination,bytes);
}

extern "C" uint32_t aclrtlaunch_v9_counter_calibration(uint32_t,void*,void*);
extern "C" uint32_t kvip_counter_calibration(void* output,void* stream) {
    if(!output || !stream)return 100000;
    return aclrtlaunch_v9_counter_calibration(1,stream,output);
}

extern "C" uint32_t aclrtlaunch_v9_query_fetch_only(uint32_t,void*,void*,uint32_t);
extern "C" uint32_t kvip_query_fetch_only(void* workspace,void* stream,uint32_t query) {
    const uint32_t selected=query & 255U;
    if(!workspace||!stream||selected>4||(query & ~511U))return 100000;
    return aclrtlaunch_v9_query_fetch_only(selected==4?8:2,stream,workspace,query);
}

extern "C" uint32_t aclrtlaunch_v9_native_fetch_only(uint32_t,void*,void*,void*,uint32_t);
extern "C" uint32_t kvip_native_fetch_only(void* workspace,void* counts,void* stream,uint32_t query) {
    if(!workspace||!counts||!stream||query>4)return 100000;
    return aclrtlaunch_v9_native_fetch_only(query==4?8:2,stream,workspace,counts,query);
}

extern "C" uint32_t kvip_streaming_resolver_cores(){return pipeline_ab::kResolveCores;}

extern "C" uint32_t kvip_streaming_flag_count(){return pipeline_ab::kFlagCount;}
