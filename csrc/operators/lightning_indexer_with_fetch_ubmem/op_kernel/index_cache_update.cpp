// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
  #include "simt_api/asc_simt.h"
  #include "simt_api/device_functions.h"
  #include "simt_api/device_atomic_functions.h"
  #include "simt_api/device_sync_functions.h"
  #include "simt_api/device_warp_functions.h"
  #include "index_cache_update.h"
  #include "context.h"
  #include "completion.h"
  #include "../../common/lifetch/staged_outputs.h"
using namespace AscendC;
namespace {
constexpr uint32_t kCandidates = kExtendedCandidates, kThreads = 512;
template<uint32_t Threads>
__simt_callee__ inline void PublishOutputRow(__gm__ uint64_t* a,uint32_t row,
    uint64_t sourceRow,uint32_t owner,bool active) {
  auto topk=reinterpret_cast<__gm__ int32_t*>(a[18]);
  auto slots=reinterpret_cast<__gm__ int32_t*>(a[19]);
  auto logical=reinterpret_cast<__gm__ int64_t*>(a[3]);
  auto logicalOut=reinterpret_cast<__gm__ int32_t*>(a[22]);
  auto slotsOut=reinterpret_cast<__gm__ int32_t*>(a[23]);
  for(uint32_t k=threadIdx.x;k<2048;k+=Threads) {
    int32_t token=active?asc_ldcg(topk+sourceRow*2048+k):-1;
    int32_t id=token>=0 && uint32_t(token)<uint32_t(a[24])?
        static_cast<int32_t>(asc_ldcg(logical+uint64_t(owner)*uint32_t(a[24])+token)):-1;
    int32_t slot=active?asc_ldcg(slots+sourceRow*2048+k):-1;
    asc_stcg(logicalOut+uint64_t(row)*2048+k,id);
    asc_stcg(slotsOut+uint64_t(row)*2048+k,slot);
  }
}
template<uint32_t Threads>
__simt_callee__ inline void PublishOwnedOutputs(__gm__ uint64_t* a,uint32_t batch,
    uint32_t owner,uint32_t planError) {
  auto unpack=reinterpret_cast<__gm__ int64_t*>(a[20]);
  auto valid=reinterpret_cast<__gm__ int64_t*>(a[21]);
  for(uint32_t row=0;row<uint32_t(SIM_BATCH)*4;++row) {
    bool active=asc_ldcg(valid+row)!=0;
    uint64_t sourceRow=active?asc_ldcg(unpack+row):0;
    active=active && sourceRow<uint64_t(SIM_BATCH)*4;
    // Valid TND rows belong to the packed producer, never the output row's
    // batch. Padding/invalid selectors still have exactly one clearing owner.
    uint32_t writer=active?uint32_t(sourceRow/4):row/4;
    if(writer==batch)PublishOutputRow<Threads>(a,row,sourceRow,owner,active && !planError);
  }
}
// keys, scales, addresses, logical IDs, LRU slots, producer changes,
// status, address row stride, producer rows, hot validity. No KV payload input.
template<bool Extended,bool WithOutputs=false>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void Update(__gm__ uint64_t* a, uint32_t b, uint32_t owner,
                                                            bool checkTransport, __ubuf__ uint32_t* work) {
  uint32_t tid = threadIdx.x;
  auto status = work + (Extended ? 2*kExtendedHash : kPlanWords);
  auto rows = reinterpret_cast<__gm__ uint32_t*>(a[8]) + b * 256;
  auto lru = reinterpret_cast<__gm__ uint32_t*>(a[4]) + b * kCandidates;
  auto changes = reinterpret_cast<__gm__ uint32_t*>(a[5]) + b * kCandidates;
  // Do not publish Index ownership after a failed/incomplete producer fetch.
  // These are the existing Row error, pending and finalPending ABI fields.
  if (!tid) status[0] = (asc_ldcg(rows + 11) || asc_ldcg(rows + 211) || asc_ldcg(rows + 248)) ? 2U : 0U;
  if (!tid && checkTransport) {
    auto rowStatus = reinterpret_cast<__gm__ uint32_t*>(a[10]) + b * 32;
    if (asc_ldcg(rowStatus)) status[0] |= 4U;
    auto posts = reinterpret_cast<__gm__ uint32_t*>(a[11]) + uint64_t(b) * a[14] * 32;
    for (uint32_t worker = 0; worker < a[14]; ++worker)
      for (uint32_t field = 1; field < 4; ++field)
        if (asc_ldcg(posts + worker * 32 + field)) status[0] |= 4U;
    auto completed=reinterpret_cast<__gm__ uint64_t*>(a[12])+uint64_t(b)*rotating::kUbmemStatusWords;
    auto region=reinterpret_cast<__gm__ uint64_t*>(a[13]);
    if(asc_ldcg(region)!=rotating::kUbmemMagic || asc_ldcg(region+1)!=rotating::kUbmemVersion ||
       !asc_ldcg(region+5) || asc_ldcg(completed) ||
       asc_ldcg(completed+1)!=asc_ldcg(completed+2) ||
       asc_ldcg(completed+3)!=asc_ldcg(completed+4) ||
       asc_ldcg(completed+6)!=asc_ldcg(region+5))status[0]|=4U;
  }
  asc_syncthreads();
  if (!status[0]) {
    if constexpr (!Extended) BuildUpdatePlan<kThreads,V9_BITMAP_INDEX!=0>(lru,changes,uint32_t(a[24]),uint32_t(a[25]),work,status);
    else BuildExtendedPlan<kThreads,V9_BITMAP_INDEX!=0>(lru,changes,
        asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(a[26])+b),uint32_t(a[24]),uint32_t(a[25]),work,status,
        reinterpret_cast<__gm__ uint32_t*>(a[27])+uint64_t(b)*kExtendedWorkWords);
  }
  asc_syncthreads();
  if (!tid) asc_stcg(reinterpret_cast<__gm__ uint32_t*>(a[6]) + b, status[0]);
  auto plan = reinterpret_cast<__gm__ uint32_t*>(a[checkTransport ? 17 : 10]) + uint64_t(b) * kExtendedTransferWords;
  if (status[0]) {
    if (!tid) asc_stcg(plan, uint32_t(0));
  } else {
    if constexpr (!Extended) PublishUpdatePlan<kThreads>(work,status,plan,uint32_t(a[25]));
    else PublishExtendedPlan<kThreads>(reinterpret_cast<__gm__ uint32_t*>(a[27])+uint64_t(b)*kExtendedWorkWords,
                                       status,plan,uint32_t(a[25]));
  }
  // The optional publisher consumes this VF's local error. No other batch's
  // status or plan is read, and Apply remains a later kernel boundary.
  if constexpr(WithOutputs)PublishOwnedOutputs<kThreads>(a,b,owner,status[0]);
  asc_threadfence();
}
// Service output adaptation runs before Apply changes logical Index ownership.
// Distribute complete TND query rows across the AIV budget. Each output row
// owns 8 KiB in each output tensor, so no two AIVs share a write sector. Keep
// this separate from Apply: another row may still read the old logical Index.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishOutputs(__gm__ uint64_t* a, uint32_t core) {
  auto unpack = reinterpret_cast<__gm__ int64_t*>(a[20]);
  auto valid = reinterpret_cast<__gm__ int64_t*>(a[21]);
  auto owners = reinterpret_cast<__gm__ uint32_t*>(a[9]);
  for (uint32_t row = core; row < uint32_t(SIM_BATCH)*4; row += rotating::kOutputCores) {
    bool active = asc_ldcg(valid + row) != 0;
    uint64_t sourceRow = active ? asc_ldcg(unpack + row) : 0;
    // Plan statuses are complete on every packed row before this separate
    // launch. A TND row need not come from the same packed batch.
    active = active && sourceRow<uint64_t(SIM_BATCH)*4 &&
        asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(a[6])+sourceRow/4)==0;
    uint64_t owner = active ? asc_ldcg(owners + sourceRow / 4) : 0;
    PublishOutputRow<kThreads>(a,row,sourceRow,uint32_t(owner),active);
  }
}
template <bool Owned>
__aicore__ inline void Apply(GM_ADDR bindings) {
  uint32_t b = GetBlockIdx() / kCopyWorkers, group = GetBlockIdx() % kCopyWorkers;
  auto a = reinterpret_cast<__gm__ uint64_t*>(bindings);
  auto input = reinterpret_cast<__gm__ uint32_t*>(a[Owned ? 17 : 10]) + uint64_t(b) * kExtendedTransferWords;
  uint32_t owner = Owned ? reinterpret_cast<__gm__ uint32_t*>(a[9])[b] : b;
  TPipe pipe;
  TBuf<TPosition::VECCALC> storage;
  pipe.InitBuffer(storage, kMoveScratchBytes);
  auto local = storage.Get<uint32_t>();
  GlobalTensor<uint32_t> planGm;
  planGm.SetGlobalBuffer(input);
  DataCopy(local, planGm, kTransferHeaderWords);
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  uint32_t count = local.GetValue(0);
  if (!count) return;
  if(count>kPlanRecords) {
    if(count>kExtendedRecords)return;
    Simt::VF_CALL<MoveExtended<kThreads, V9_UBMEM_BALANCED_INDEX != 0>>(Simt::Dim3{kThreads,1,1},a,owner,group,Owned,input,
      reinterpret_cast<__ubuf__ uint64_t*>(local.GetPhyAddr()),count);
    return;
  }
  DataCopy(local[kTransferHeaderWords], planGm[kTransferHeaderWords], ((count * 2 + 7) / 8) * 8);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  Simt::VF_CALL<Move<kThreads, V9_UBMEM_BALANCED_INDEX != 0>>(Simt::Dim3{kThreads, 1, 1}, a, owner, group, Owned,
                                reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ uint64_t*>(local[kTransferPlanWords].GetPhyAddr()), count);
}

template<bool ReleaseScratch=false>
__aicore__ inline void PlanOwned(GM_ADDR bindings,bool withOutputs=false) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> scratch;
    pipe.InitBuffer(scratch, kExtendedScratchBytes);
    auto local = scratch.Get<uint32_t>();
    auto args = reinterpret_cast<__gm__ uint64_t*>(bindings);
    auto owners = reinterpret_cast<__gm__ uint32_t*>(args[9]);
    uint32_t batch = GetBlockIdx();
    auto counts=reinterpret_cast<__gm__ uint32_t*>(args[26]);
    auto work=reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
    if(withOutputs) {
      if(counts[batch]<=kPlanCandidates)
        Simt::VF_CALL<Update<false,true>>(Simt::Dim3{kThreads,1,1},args,batch,owners[batch],true,work);
      else Simt::VF_CALL<Update<true,true>>(Simt::Dim3{kThreads,1,1},args,batch,owners[batch],true,work);
    } else {
      if(counts[batch]<=kPlanCandidates)
        Simt::VF_CALL<Update<false>>(Simt::Dim3{kThreads,1,1},args,batch,owners[batch],true,work);
      else Simt::VF_CALL<Update<true>>(Simt::Dim3{kThreads,1,1},args,batch,owners[batch],true,work);
    }
    if constexpr(ReleaseScratch) {
      // Update releases its GM status/plan writes before returning. Join the
      // VF and destroy its UB/events before another TPipe occupies this AIV.
      SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
      pipe.Destroy();
    }
}

__aicore__ inline void CompleteAndPlan(GM_ADDR bindings,GM_ADDR update,bool withOutputs) {
    uint32_t batch=GetBlockIdx();if(batch>=SIM_BATCH)return;
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]);
    rotating::CompleteBatch(ctx,batch);
    // Scalar completion/cache publication must precede SIMT status reads.
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
#if V9_UBMEM_MERGE_OUTPUTS && V9_STAGED_OUTPUTS
    auto args=reinterpret_cast<__gm__ uint64_t*>(update);
    if(withOutputs && kvip_output::CanStageOutputs(uint32_t(args[25]),SIM_BATCH)) {
      PlanOwned<true>(update);
      // This publisher reads only this packed row's completed status. It
      // owns all TND rows that select this producer, even across batches.
      // Apply remains a later kernel, after every old logical ID was read.
      kvip_output::PublishStagedOutputs<true>(args,batch,SIM_BATCH);
      return;
    }
#endif
    PlanOwned(update,withOutputs);
}

}  // namespace
#endif


// Same update body, with stable request-owner storage instead of compact batch
// storage. The test interface also carries its hot-valid pointer.
extern "C" __global__ __aicore__ void kvip_ubmem_direct_update_owned_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    PlanOwned(bindings);
  }
#endif
}
// The preceding resolve launch completed every final worker. Completion and
// planning are row-local; the separate outputs launch remains the cross-row
// barrier needed by TND unpack, before Apply changes logical Index ownership.
extern "C" __global__ __aicore__ void kvip_ubmem_complete_plan_kernel(GM_ADDR bindings,GM_ADDR update) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    CompleteAndPlan(bindings,update,false);
  }
#endif
}
// Packed ownership removes the cross-batch status dependency of TND output
// publication. All old logical IDs are read before the following Apply launch.
extern "C" __global__ __aicore__ void kvip_ubmem_complete_plan_outputs_kernel(GM_ADDR bindings,GM_ADDR update) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {CompleteAndPlan(bindings,update,true);}
#endif
}
extern "C" __global__ __aicore__ void kvip_ubmem_direct_update_outputs_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    auto args = reinterpret_cast<__gm__ uint64_t*>(bindings);
    uint32_t core = GetBlockIdx();
#if V9_STAGED_OUTPUTS
    if(kvip_output::CanStageOutputs(uint32_t(args[25]),SIM_BATCH)) {
      if(core<SIM_BATCH)kvip_output::PublishStagedOutputs<true>(args,core,SIM_BATCH);
      return;
    }
#endif
    if(core>=rotating::kOutputCores)return;
    Simt::VF_CALL<PublishOutputs>(Simt::Dim3{kThreads, 1, 1}, args, core);
  }
#endif
}


extern "C" __global__ __aicore__ void kvip_ubmem_direct_apply_owned_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    Apply<true>(bindings);
  }
#endif
}
