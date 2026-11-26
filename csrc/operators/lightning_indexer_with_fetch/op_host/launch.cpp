#include "../op_kernel/pipeline_types.h"
#include "api.h"
#include "context.h"
#if KVIP_STAGE_TRACE
#include <acl/acl_prof.h>
extern "C" uint32_t kvip_stage_trace_supported() { return 1; }
#endif
extern "C" uint32_t kvip_prepare_shape(rotating::RuntimeShape* shape) { return rotating::PrepareShape(shape); }
extern "C" uint32_t aclrtlaunch_v9_init_kernel(uint32_t,void*,void*,uint32_t,void*,
    uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
static uint32_t LaunchInit(void* bindings,void* stream,const rotating::RuntimeShape* shape,void* workspace,uint32_t fullReset) {
    return aclrtlaunch_v9_init_kernel(shape->batch*4,stream,bindings,fullReset,workspace,
        shape->batch, shape->history, shape->resident, shape->rounds,
        shape->serviceLanes, shape->finalWorkers, shape->finalPostOffset, shape->tokenBytes);
}
extern "C" uint32_t v9_init(void* bindings,void* stream,const rotating::RuntimeShape* shape) {
    if (!bindings || !stream || !shape) return 100000;
    return LaunchInit(bindings,stream,shape,nullptr,1);
}

extern "C" uint32_t aclrtlaunch_v9_li_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_1(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_2(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_3(uint32_t,void*,void*);
static uint32_t LaunchLi(void* bindings, void* stream, const rotating::RuntimeShape* shape) {
#define LI_LAUNCH_0(blocks,stream,bindings) aclrtlaunch_v9_li_kernel(blocks,stream,bindings)
#define LI_LAUNCH_1 aclrtlaunch_v9_li_kernel_1
#define LI_LAUNCH_2 aclrtlaunch_v9_li_kernel_2
#define LI_LAUNCH_3 aclrtlaunch_v9_li_kernel_3
#include "../../common/lifetch/li_launch.h"
#undef LI_LAUNCH_0
#undef LI_LAUNCH_1
#undef LI_LAUNCH_2
#undef LI_LAUNCH_3
}
extern "C" uint32_t v9_li(void* bindings,void* stream,const rotating::RuntimeShape* shape) {
    return LaunchLi(bindings,stream,shape);
}
extern "C" uint32_t v9_li_cores(const rotating::RuntimeShape* shape) { return shape->liCores; }
extern "C" uint32_t v9_batch(const rotating::RuntimeShape* shape) { return shape->batch; }
extern "C" uint32_t v9_history(const rotating::RuntimeShape* shape) { return shape->history; }
extern "C" uint32_t v9_resident(const rotating::RuntimeShape* shape) { return shape->resident; }
extern "C" uint32_t v9_row_bytes(const rotating::RuntimeShape* shape) { return sizeof(rotating::Row); }
extern "C" uint32_t v9_context_bytes(const rotating::RuntimeShape* shape) { return sizeof(rotating::Context); }
extern "C" uint32_t v9_rounds(const rotating::RuntimeShape* shape) { return shape->rounds; }
extern "C" uint32_t v9_simulation(const rotating::RuntimeShape* shape) { return 0; }
extern "C" uint32_t v9_abi_version(const rotating::RuntimeShape* shape) { return 12; }
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
    uint32_t result=LaunchLi(bindings,stream,shape);
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

extern "C" uint32_t aclrtlaunch_v9_parallel_terminal(uint32_t,void*,void*);
extern "C" uint32_t kvip_parallel_tail_supported(const rotating::RuntimeShape* shape) {
    return shape && shape->batch && shape->batch<=16 && shape->finalWorkers==4 &&
        shape->batch*4<=shape->aicCount*2 && rotating::UseSequenceSplit(*shape);
}
// Keep the threshold, Cold/Service and Terminal launches in one shared path.
#if KVIP_HOT_SCORE_PARTITION
#include "../../common/lifetch/hot_score_partition.h"
#include "../../common/lifetch/cold_score_partition.h"
#include "../../common/lifetch/dedicated_cold_schedule.h"
#include "../op_kernel/p2_w1_schedule.h"
#include "../op_kernel/cold_gm_schedule.h"
extern "C" uint32_t aclrtlaunch_v9_li_kernel_hot_partitioned(uint32_t,void*,void*,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_threshold_cold_partitioned(uint32_t,void*,void*,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_cold_partitioned(uint32_t,void*,void*,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_cold_ready_partitioned(uint32_t,void*,void*,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_threshold_retired_service(uint32_t,void*,void*,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_cold_retired_service(uint32_t,void*,void*,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_resident_topk_high(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_score_t2(uint32_t,void*,void*,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_cold_t2(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_threshold_dedicated_service(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_cold_dedicated_service(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_hot_ready_service(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_cold_static_ready_service(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_dynamic_service(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_threshold_p2_w1(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_cold_p2_w1(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_hot_cold_gm(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_cold_gm_batch(uint32_t,void*,void*,uint32_t);
#endif
enum class ParallelTailStage { Hot, Threshold, ColdService, Terminal, ResidentHigh };

template<bool Profile, typename Launch>
static uint32_t LaunchTailStage(void* stream, ParallelTailStage stage, Launch launch) {
#if KVIP_STAGE_TRACE
    if constexpr (Profile) {
        static constexpr const char* markers[][2] = {
            {"KVIP_LIFETCH_HOT_BEGIN", "KVIP_LIFETCH_HOT_END"},
            {"KVIP_LIFETCH_THRESHOLD_BEGIN", "KVIP_LIFETCH_THRESHOLD_END"},
            {"KVIP_LIFETCH_COLD_SERVICE_BEGIN", "KVIP_LIFETCH_COLD_SERVICE_END"},
            {"KVIP_LIFETCH_TERMINAL_BEGIN", "KVIP_LIFETCH_TERMINAL_END"},
            {"KVIP_LIFETCH_RESIDENT_HIGH_BEGIN", "KVIP_LIFETCH_RESIDENT_HIGH_END"},
        };
        const auto& [begin, end] = markers[static_cast<uint32_t>(stage)];
        // The markers are submitted to the same stream as the kernel so
        // msprof can show the device boundaries as well as host submission.
        uint32_t result = aclprofMarkEx(begin, std::strlen(begin), stream);
        if (result) return result;
        result = launch();
        if (result) return result;
        return aclprofMarkEx(end, std::strlen(end), stream);
    }
#endif
    return launch();
}

template<bool Profile = false>
static uint32_t LaunchParallelTail(void* bindings,void* terminal,
    void* stream,const rotating::RuntimeShape* shape,uint32_t hotParts,uint32_t coldParts=2,bool earlyW1=false,
    bool readyCold=false,bool residentHigh=false,bool retiredService=false) {
    if(!bindings || !terminal || !stream || !kvip_parallel_tail_supported(shape))return 100000;
    uint32_t result;
#if KVIP_HOT_SCORE_PARTITION
    if(hotParts>2)result=aclrtlaunch_v9_li_kernel_hot_partitioned(shape->batch*hotParts,stream,bindings,hotParts);
    else
#endif
    result=LaunchTailStage<Profile>(stream,ParallelTailStage::Hot,[&] {
        return aclrtlaunch_v9_li_kernel(shape->liCores,stream,bindings);
    });
#if KVIP_HOT_SCORE_PARTITION
    if (residentHigh && !result) result=LaunchTailStage<Profile>(stream,ParallelTailStage::ResidentHigh,[&] {
        return aclrtlaunch_v9_li_kernel_resident_topk_high(shape->batch*8,stream,bindings);
    });
    if (earlyW1) {
        if (!result) result = LaunchTailStage<Profile>(stream, ParallelTailStage::Threshold, [&] {
            return aclrtlaunch_v9_li_kernel_threshold_p2_w1(shape->liCores, stream, bindings);
        });
        if (!result) result = LaunchTailStage<Profile>(stream, ParallelTailStage::ColdService, [&] {
            return aclrtlaunch_v9_li_kernel_cold_p2_w1(shape->liCores, stream, bindings);
        });
    } else if(coldParts>2){
        if(!result)result=LaunchTailStage<Profile>(stream,ParallelTailStage::Threshold,[&] {
            return retiredService
                ? aclrtlaunch_v9_li_kernel_threshold_retired_service(shape->liCores,stream,bindings,coldParts)
                : aclrtlaunch_v9_li_kernel_threshold_cold_partitioned(shape->liCores,stream,bindings,coldParts);
        });
        if(!result)result=LaunchTailStage<Profile>(stream,ParallelTailStage::ColdService,[&] {
            if (retiredService) {
                return aclrtlaunch_v9_li_kernel_cold_retired_service(shape->batch*coldParts,stream,bindings,coldParts);
            }
            return readyCold
                ? aclrtlaunch_v9_li_kernel_cold_ready_partitioned(shape->batch*coldParts,stream,bindings,coldParts)
                : aclrtlaunch_v9_li_kernel_cold_partitioned(shape->batch*coldParts,stream,bindings,coldParts);
        });
    } else
#endif
    {
        if(!result)result=LaunchTailStage<Profile>(stream,ParallelTailStage::Threshold,[&] {
            return aclrtlaunch_v9_li_kernel_1(shape->liCores,stream,bindings);
        });
        if(!result)result=LaunchTailStage<Profile>(stream,ParallelTailStage::ColdService,[&] {
            return aclrtlaunch_v9_li_kernel_2(shape->liCores,stream,bindings);
        });
    }
    if(!result)result=LaunchTailStage<Profile>(stream,ParallelTailStage::Terminal,[&] {
        return aclrtlaunch_v9_parallel_terminal(shape->batch*4,stream,terminal);
    });
    return result;
}
extern "C" uint32_t kvip_parallel_tail_forward(void* bindings,void* terminal,
    void* stream,const rotating::RuntimeShape* shape) {
    return LaunchParallelTail(bindings,terminal,stream,shape,2);
}

#if KVIP_STAGE_TRACE
extern "C" uint32_t kvip_parallel_tail_profile_forward(void* bindings,void* terminal,
    void* stream,const rotating::RuntimeShape* shape) {
    return LaunchParallelTail<true>(bindings,terminal,stream,shape,2);
}
#endif

#if KVIP_HOT_SCORE_PARTITION
extern "C" uint32_t kvip_p2_w1_supported(const rotating::RuntimeShape* shape) {
    return kvip_parallel_tail_supported(shape) && kvip_p2_w1::Supports(*shape);
}

extern "C" uint32_t kvip_parallel_tail_p2_w1_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    if (!kvip_p2_w1_supported(shape)) return 100000;
    return LaunchParallelTail(bindings, terminal, stream, shape, 2, 2, true);
}

extern "C" uint32_t kvip_cold_gm_supported(const rotating::RuntimeShape* shape) {
    return kvip_parallel_tail_supported(shape) && kvip_cold_gm::Supports(*shape);
}

static uint32_t LaunchColdGm(void* bindings, void* terminal, void* stream,
    const rotating::RuntimeShape* shape, bool consumeReadyRows) {
    if (!bindings || !terminal || !stream || !kvip_cold_gm_supported(shape)) return 100000;
    uint32_t result = aclrtlaunch_v9_li_kernel_hot_cold_gm(shape->liCores, stream, bindings);
    if (!result) result = aclrtlaunch_v9_li_kernel_cold_gm_batch(
        shape->liCores, stream, bindings, uint32_t(consumeReadyRows));
    if (!result && !consumeReadyRows) {
        result = aclrtlaunch_v9_li_kernel_cold_t2(shape->liCores, stream, bindings);
    }
    if (!result) result = aclrtlaunch_v9_parallel_terminal(shape->batch * 4, stream, terminal);
    return result;
}

extern "C" uint32_t kvip_parallel_tail_cold_gm_fenced_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    return LaunchColdGm(bindings, terminal, stream, shape, false);
}

extern "C" uint32_t kvip_parallel_tail_cold_gm_pipeline_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    return LaunchColdGm(bindings, terminal, stream, shape, true);
}

extern "C" uint32_t kvip_dedicated_cold_supported(const rotating::RuntimeShape* shape) {
    using namespace kvip_dedicated_cold;
    return kvip_parallel_tail_supported(shape) && shape->batch == kBatch && shape->rounds == kRounds &&
        shape->history == kHistory && shape->resident == kResident && shape->indexHeads == 32 &&
        shape->aicCount == kMixGroups && shape->liCores == kMixGroups && shape->serviceLanes == kServiceLanes;
}

template<bool Profile = false>
static uint32_t LaunchDedicatedService(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    if (!bindings || !terminal || !stream || !kvip_dedicated_cold_supported(shape)) return 100000;
    const uint32_t groups = shape->liCores;
    uint32_t result = LaunchTailStage<Profile>(stream, ParallelTailStage::Hot, [&] {
        return aclrtlaunch_v9_li_kernel(groups, stream, bindings);
    });
    if (!result) result = LaunchTailStage<Profile>(stream, ParallelTailStage::Threshold, [&] {
        return aclrtlaunch_v9_li_kernel_threshold_dedicated_service(groups, stream, bindings);
    });
    if (!result) result = LaunchTailStage<Profile>(stream, ParallelTailStage::ColdService, [&] {
        return aclrtlaunch_v9_li_kernel_cold_dedicated_service(groups, stream, bindings);
    });
    if (!result) result = LaunchTailStage<Profile>(stream, ParallelTailStage::Terminal, [&] {
        return aclrtlaunch_v9_parallel_terminal(shape->batch * 4, stream, terminal);
    });
    return result;
}

extern "C" uint32_t kvip_parallel_tail_dedicated_service_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    return LaunchDedicatedService(bindings, terminal, stream, shape);
}

#if KVIP_STAGE_TRACE
extern "C" uint32_t kvip_parallel_tail_dedicated_service_profile_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    return LaunchDedicatedService<true>(bindings, terminal, stream, shape);
}
#endif

extern "C" uint32_t kvip_dynamic_service_supported(const rotating::RuntimeShape* shape) {
#if KVIP_SERVICE_LOCAL_COUNTS || KVIP_SERVICE_UB_CANDIDATES
    return 0;
#else
    return kvip_dedicated_cold_supported(shape);
#endif
}

template<bool Dynamic, bool Profile = false>
static uint32_t LaunchReadyService(void* bindings, void* terminal, void* stream,
                                   const rotating::RuntimeShape* shape) {
    if (!bindings || !terminal || !stream || !kvip_dynamic_service_supported(shape)) return 100000;
    const uint32_t groups = shape->liCores;
    uint32_t result = LaunchTailStage<Profile>(stream, ParallelTailStage::Hot, [&] {
        return aclrtlaunch_v9_li_kernel_hot_ready_service(groups, stream, bindings);
    });
    if constexpr (!Dynamic) {
        if (!result) result = LaunchTailStage<Profile>(stream, ParallelTailStage::Threshold, [&] {
            return aclrtlaunch_v9_li_kernel_threshold_dedicated_service(groups, stream, bindings);
        });
    }
    if (!result) result = LaunchTailStage<Profile>(stream, ParallelTailStage::ColdService, [&] {
        if constexpr (Dynamic) return aclrtlaunch_v9_li_kernel_dynamic_service(groups, stream, bindings);
        return aclrtlaunch_v9_li_kernel_cold_static_ready_service(groups, stream, bindings);
    });
    if (!result) result = LaunchTailStage<Profile>(stream, ParallelTailStage::Terminal, [&] {
        return aclrtlaunch_v9_parallel_terminal(shape->batch * 4, stream, terminal);
    });
    return result;
}

extern "C" uint32_t kvip_parallel_tail_static_ready_service_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    return LaunchReadyService<false>(bindings, terminal, stream, shape);
}

extern "C" uint32_t kvip_parallel_tail_dynamic_service_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    return LaunchReadyService<true>(bindings, terminal, stream, shape);
}

#if KVIP_STAGE_TRACE
extern "C" uint32_t kvip_parallel_tail_static_ready_service_profile_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    return LaunchReadyService<false, true>(bindings, terminal, stream, shape);
}

extern "C" uint32_t kvip_parallel_tail_dynamic_service_profile_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    return LaunchReadyService<true, true>(bindings, terminal, stream, shape);
}
#endif

extern "C" uint32_t kvip_partitioned_hot_score_parts(const rotating::RuntimeShape* shape) {
    return kvip_parallel_tail_supported(shape) ? kvip_hot_score::Parts(shape->batch,shape->resident,shape->aicCount) : 0;
}
extern "C" uint32_t kvip_parallel_tail_hot_partitioned_forward(void* bindings,void* terminal,
    void* stream,const rotating::RuntimeShape* shape) {
    const auto parts=kvip_partitioned_hot_score_parts(shape);
    if(parts<2)return 100000;
    return LaunchParallelTail(bindings,terminal,stream,shape,parts);
}
extern "C" uint32_t kvip_partitioned_cold_score_parts(const rotating::RuntimeShape* shape) {
    if(!kvip_parallel_tail_supported(shape) || shape->liCores!=2*shape->batch)return 0;
    return kvip_cold_score::Parts(shape->batch,shape->history,shape->resident,
        rotating::SequenceHalfWindow(*shape),shape->rounds,shape->aicCount);
}
extern "C" uint32_t kvip_small_batch_cold_parts(const rotating::RuntimeShape* shape) {
    if(!kvip_parallel_tail_supported(shape) || shape->liCores!=2*shape->batch)return 0;
    return kvip_cold_score::ReadyParts(shape->batch,shape->history,shape->resident,
        rotating::SequenceHalfWindow(*shape),shape->rounds,shape->aicCount);
}

template<bool Profile>
static uint32_t LaunchSmallBatchCold(void* bindings,void* terminal,void* stream,const rotating::RuntimeShape* shape) {
    const auto parts=kvip_small_batch_cold_parts(shape);
    if(parts==2)return LaunchParallelTail<Profile>(bindings,terminal,stream,shape,2);
    if(parts!=4)return 100000;
    return LaunchParallelTail<Profile>(bindings,terminal,stream,shape,2,parts,false,true);
}

extern "C" uint32_t kvip_small_batch_cold_p8_parts(const rotating::RuntimeShape* shape) {
    constexpr uint32_t parts = 8;
    if (!kvip_parallel_tail_supported(shape) || shape->batch != 1 || shape->liCores != 2 ||
        shape->rounds != 2 || shape->serviceAics != 1 ||
        shape->aicCount < parts || shape->aicCount > kvip_cold_score::kMaxParts)
        return 0;
    const auto half = rotating::SequenceHalfWindow(*shape);
    return kvip_cold_score::CompleteWindows(shape->history, shape->resident, half, shape->rounds, parts) ? parts : 0;
}

extern "C" uint32_t kvip_small_batch_cold_full_parts(const rotating::RuntimeShape* shape) {
    constexpr uint32_t groups = 32;
    if (!kvip_parallel_tail_supported(shape) || shape->aicCount != groups ||
        (shape->batch != 1 && shape->batch != 2 && shape->batch != 8) ||
        shape->liCores != 2 * shape->batch || shape->history != 131072 ||
        shape->resident != 16384 || shape->indexHeads != 32)
        return 0;
    const auto parts = groups / shape->batch;
    const auto half = rotating::SequenceHalfWindow(*shape);
    return kvip_cold_score::CompleteWindows(shape->history, shape->resident, half, shape->rounds, parts) ? parts : 0;
}

template<bool Profile, bool RetiredService = false>
static uint32_t LaunchSmallBatchColdFull(void* bindings, void* terminal, void* stream,
                                        const rotating::RuntimeShape* shape) {
    const auto parts = kvip_small_batch_cold_full_parts(shape);
    if (!parts) return 100000;
    if constexpr (RetiredService) {
        if (shape->serviceLanes < shape->batch || shape->serviceLanes > shape->batch + 1) return 100000;
    }
    return LaunchParallelTail<Profile>(bindings, terminal, stream, shape, 2, parts, false, true, false, RetiredService);
}

template<bool Profile>
static uint32_t LaunchSmallBatchColdP8(void* bindings, void* terminal, void* stream,
                                      const rotating::RuntimeShape* shape) {
    const auto parts = kvip_small_batch_cold_p8_parts(shape);
    if (!parts) return 100000;
    return LaunchParallelTail<Profile>(bindings, terminal, stream, shape, 2, parts, false, true);
}

template<bool Profile, uint32_t Parts>
static uint32_t LaunchSmallBatchColdRetired(void* bindings, void* terminal, void* stream,
                                            const rotating::RuntimeShape* shape) {
    const auto supported = Parts == 4 ? kvip_small_batch_cold_parts(shape) : kvip_small_batch_cold_p8_parts(shape);
    if (supported != Parts || shape->serviceLanes < shape->batch ||
        shape->serviceLanes > shape->batch + 1) return 100000;
    return LaunchParallelTail<Profile>(bindings, terminal, stream, shape, 2, Parts, false, true, false, true);
}

template<bool Profile>
static uint32_t LaunchSmallBatchTopK(void* bindings,void* terminal,void* stream,const rotating::RuntimeShape* shape) {
    if(!kvip_parallel_tail_supported(shape) || (shape->batch!=1 && shape->batch!=8) ||
        shape->history!=131072 || shape->resident!=16384 || shape->indexHeads!=32 || shape->liCores!=2*shape->batch ||
        shape->batch*8>2*shape->aicCount)return 100000;
    return LaunchParallelTail<Profile>(bindings,terminal,stream,shape,2,2,false,false,true);
}

extern "C" uint32_t kvip_small_batch_cold_forward(void* bindings,void* terminal,void* stream,
    const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchCold<false>(bindings,terminal,stream,shape);
}

extern "C" uint32_t kvip_small_batch_cold_p8_forward(void* bindings, void* terminal, void* stream,
                                                  const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchColdP8<false>(bindings, terminal, stream, shape);
}

extern "C" uint32_t kvip_small_batch_cold_full_forward(void* bindings, void* terminal, void* stream,
                                                       const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchColdFull<false>(bindings, terminal, stream, shape);
}

extern "C" uint32_t kvip_small_batch_cold_retired_forward(void* bindings, void* terminal, void* stream,
                                                          const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchColdFull<false, true>(bindings, terminal, stream, shape);
}

extern "C" uint32_t kvip_small_batch_cold_retired_p4_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchColdRetired<false, 4>(bindings, terminal, stream, shape);
}

extern "C" uint32_t kvip_small_batch_cold_retired_p8_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchColdRetired<false, 8>(bindings, terminal, stream, shape);
}

extern "C" uint32_t kvip_small_batch_topk_forward(void* bindings,void* terminal,void* stream,
    const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchTopK<false>(bindings,terminal,stream,shape);
}

#if KVIP_STAGE_TRACE
extern "C" uint32_t kvip_small_batch_cold_profile_forward(void* bindings,void* terminal,void* stream,
    const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchCold<true>(bindings,terminal,stream,shape);
}

extern "C" uint32_t kvip_small_batch_cold_p8_profile_forward(void* bindings, void* terminal, void* stream,
                                                          const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchColdP8<true>(bindings, terminal, stream, shape);
}

extern "C" uint32_t kvip_small_batch_cold_full_profile_forward(void* bindings, void* terminal, void* stream,
                                                               const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchColdFull<true>(bindings, terminal, stream, shape);
}

extern "C" uint32_t kvip_small_batch_cold_retired_profile_forward(void* bindings, void* terminal, void* stream,
                                                                  const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchColdFull<true, true>(bindings, terminal, stream, shape);
}

extern "C" uint32_t kvip_small_batch_cold_retired_p4_profile_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchColdRetired<true, 4>(bindings, terminal, stream, shape);
}

extern "C" uint32_t kvip_small_batch_cold_retired_p8_profile_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchColdRetired<true, 8>(bindings, terminal, stream, shape);
}

extern "C" uint32_t kvip_small_batch_topk_profile_forward(void* bindings,void* terminal,void* stream,
    const rotating::RuntimeShape* shape) {
    return LaunchSmallBatchTopK<true>(bindings,terminal,stream,shape);
}
#endif

extern "C" uint32_t kvip_parallel_tail_cold_partitioned_forward(void* bindings,void* terminal,
    void* stream,const rotating::RuntimeShape* shape) {
    const auto cold=kvip_partitioned_cold_score_parts(shape);
    if(cold<2)return 100000;
    return LaunchParallelTail(bindings,terminal,stream,shape,2,cold);
}
#if KVIP_STAGE_TRACE
extern "C" uint32_t kvip_parallel_tail_cold_partitioned_profile_forward(void* bindings,void* terminal,
    void* stream,const rotating::RuntimeShape* shape) {
    const auto cold=kvip_partitioned_cold_score_parts(shape);
    if(cold<=2)return 100000;
    return LaunchParallelTail<true>(bindings,terminal,stream,shape,2,cold);
}
#endif
extern "C" uint32_t kvip_parallel_tail_history_partitioned_forward(void* bindings,void* terminal,
    void* stream,const rotating::RuntimeShape* shape) {
    const auto hot=kvip_partitioned_hot_score_parts(shape);
    const auto cold=kvip_partitioned_cold_score_parts(shape);
    if(hot<2 || cold<2)return 100000;
    return LaunchParallelTail(bindings,terminal,stream,shape,hot,cold);
}

extern "C" uint32_t kvip_unified_score_parts(const rotating::RuntimeShape* shape) {
    if (!kvip_parallel_tail_supported(shape) || shape->liCores != 2 * shape->batch ||
        shape->history % 256 || shape->rounds > 4) return 0;
    const uint32_t budget = shape->aicCount / shape->batch;
    const uint32_t parts = budget < 16 ? budget : 16;
    return parts >= 2 && shape->history / 256 >= parts ? parts : 0;
}

template<bool Profile = false>
static uint32_t LaunchUnifiedScore(void* bindings, void* terminal, void* stream,
    const rotating::RuntimeShape* shape, uint32_t parts) {
    if (!bindings || !terminal || !stream || parts < 2) return 100000;
    uint32_t result = LaunchTailStage<Profile>(stream, ParallelTailStage::Hot, [&] {
        return aclrtlaunch_v9_li_kernel_score_t2(shape->batch * parts, stream, bindings, parts);
    });
    if (!result) result = LaunchTailStage<Profile>(stream, ParallelTailStage::Threshold, [&] {
        return aclrtlaunch_v9_li_kernel_1(shape->liCores, stream, bindings);
    });
    if (!result) result = LaunchTailStage<Profile>(stream, ParallelTailStage::ColdService, [&] {
        return aclrtlaunch_v9_li_kernel_cold_t2(shape->liCores, stream, bindings);
    });
    if (!result) result = LaunchTailStage<Profile>(stream, ParallelTailStage::Terminal, [&] {
        return aclrtlaunch_v9_parallel_terminal(shape->batch * 4, stream, terminal);
    });
    return result;
}

extern "C" uint32_t kvip_parallel_tail_unified_p2_forward(void* bindings, void* terminal,
    void* stream, const rotating::RuntimeShape* shape) {
    return LaunchUnifiedScore(bindings, terminal, stream, shape, kvip_unified_score_parts(shape) ? 2 : 0);
}

extern "C" uint32_t kvip_parallel_tail_unified_partitioned_forward(void* bindings, void* terminal,
    void* stream, const rotating::RuntimeShape* shape) {
    return LaunchUnifiedScore(bindings, terminal, stream, shape, kvip_unified_score_parts(shape));
}
#if KVIP_STAGE_TRACE
extern "C" uint32_t kvip_parallel_tail_unified_p2_profile_forward(void* bindings, void* terminal,
    void* stream, const rotating::RuntimeShape* shape) {
    return LaunchUnifiedScore<true>(bindings, terminal, stream, shape, kvip_unified_score_parts(shape) ? 2 : 0);
}

extern "C" uint32_t kvip_parallel_tail_unified_partitioned_profile_forward(void* bindings, void* terminal,
    void* stream, const rotating::RuntimeShape* shape) {
    return LaunchUnifiedScore<true>(bindings, terminal, stream, shape, kvip_unified_score_parts(shape));
}
#endif
#endif
