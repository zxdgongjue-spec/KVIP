// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifndef V9_STAGED_OUTPUTS
#define V9_STAGED_OUTPUTS 0
#endif

#if V9_STAGED_OUTPUTS && defined(__CCE_AICORE__)
namespace kvip_output {
using namespace AscendC;
constexpr uint32_t kTop=2048,kQueries=4,kPacked=kTop*kQueries;
constexpr uint32_t kMaxResident=16384,kMaxBatch=32,kMaxQueries=kQueries*kMaxBatch;
constexpr uint32_t kTokens=kMaxResident*2,kSlots=kTokens+kPacked,kInvalid=kSlots+kPacked;
constexpr uint32_t kUnpack=kInvalid+kTop,kValid=kUnpack+kMaxQueries*2;
constexpr uint32_t kControl=kValid+kMaxQueries*2,kScratchBytes=(kControl+16)*4;
static_assert(kScratchBytes<=223232,"Staged output metadata exceeds AIV UB capacity");

__aicore__ inline bool CanStageOutputs(uint32_t resident,uint32_t batches) {
    return resident && resident<=kMaxResident && !(resident%4) && batches && batches<=kMaxBatch;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ConvertStagedOutputs(__ubuf__ int32_t* tokens,__ubuf__ int64_t* hot,
    __ubuf__ int32_t* invalid,__gm__ int64_t* logical,uint32_t history,uint32_t resident,
    uint32_t queryMask,bool hasInvalid) {
    if(queryMask)for(uint32_t i=threadIdx.x;i<kPacked;i+=512) {
        if(!(queryMask&(1U<<(i/kTop))))continue;
        int32_t token=tokens[i],value=-1;
        if(token>=0 && uint32_t(token)<history)
            value=static_cast<int32_t>(uint32_t(token)<resident?hot[token]:asc_ldcg(logical+token));
        tokens[i]=value;
    }
    if(hasInvalid)for(uint32_t i=threadIdx.x;i<kTop;i+=512)invalid[i]=-1;
}

// One AIV owns a packed request's Q4 input. Gather the small TND map once,
// reuse its resident logical-ID table across all four queries, and publish
// whole 8 KiB output rows by MTE3. Apply must remain a later kernel: these
// logical IDs describe the old Index permutation.
// CheckStatus is true only after UBmem's preceding complete/plan kernel.
// URMA's original early output contract intentionally precedes its plan.
template<bool CheckStatus>
__aicore__ inline void PublishStagedOutputs(__gm__ uint64_t* a,uint32_t b,uint32_t batches) {
    const uint32_t history=uint32_t(a[24]),resident=uint32_t(a[25]),queries=batches*kQueries;
    TPipe pipe;TBuf<TPosition::VECCALC> storage;pipe.InitBuffer(storage,kScratchBytes);
    auto local=storage.Get<uint32_t>();
    auto hot=local.ReinterpretCast<int64_t>();
    auto tokens=local[kTokens].ReinterpretCast<int32_t>();
    auto slots=local[kSlots].ReinterpretCast<int32_t>();
    auto invalid=local[kInvalid].ReinterpretCast<int32_t>();
    auto unpack=local[kUnpack].ReinterpretCast<int64_t>();
    auto valid=local[kValid].ReinterpretCast<int64_t>();
    GlobalTensor<int64_t> metadata;metadata.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(a[20]));
    DataCopy(unpack,metadata,queries);
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(a[21]));DataCopy(valid,metadata,queries);
    GlobalTensor<uint32_t> control;control.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a[9])+b);
    // A single owner/status word may be the entire B1 tensor. Pad the UB
    // destination rather than reading a complete GM sector past that tensor.
    DataCopyExtParams one{1,4,0,0,0};DataCopyPadExtParams<uint32_t> padding{false,0,0,0};
    DataCopyPad(local[kControl],control,one,padding);
    if constexpr(CheckStatus) {
        control.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a[6])+b);
        DataCopyPad(local[kControl+8],control,one,padding);
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    const uint32_t owner=local.GetValue(kControl);
    const bool healthy=!CheckStatus || !local.GetValue(kControl+8);
    uint32_t queryMask=0;bool hasInvalid=false;
    for(uint32_t row=0;row<queries;++row) {
        uint64_t source=uint64_t(unpack.GetValue(row));
        bool mapped=valid.GetValue(row)!=0 && source<queries;
        uint32_t writer=mapped?uint32_t(source/kQueries):row/kQueries;
        if(writer==b) {
            if(mapped && healthy)queryMask|=1U<<uint32_t(source%kQueries);
            hasInvalid|=!mapped || !healthy;
        }
    }
    if(!queryMask && !hasInvalid){pipe.Destroy();return;}
    auto logical=reinterpret_cast<__gm__ int64_t*>(a[3])+uint64_t(owner)*history;
    if(queryMask) {
        GlobalTensor<int64_t> sourceLogical;sourceLogical.SetGlobalBuffer(logical);
        DataCopy(hot,sourceLogical,resident);
        GlobalTensor<int32_t> source;source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[18])+uint64_t(b)*kPacked);
        DataCopy(tokens,source,kPacked);
        source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[19])+uint64_t(b)*kPacked);
        DataCopy(slots,source,kPacked);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    Simt::VF_CALL<ConvertStagedOutputs>(Simt::Dim3{512,1,1},
        reinterpret_cast<__ubuf__ int32_t*>(tokens.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int64_t*>(hot.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(invalid.GetPhyAddr()),logical,history,resident,queryMask,hasInvalid);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    GlobalTensor<int32_t> logicalOut,slotsOut;
    logicalOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[22]));
    slotsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[23]));
    for(uint32_t row=0;row<queries;++row) {
        uint64_t source=uint64_t(unpack.GetValue(row));
        bool mapped=valid.GetValue(row)!=0 && source<queries;
        uint32_t writer=mapped?uint32_t(source/kQueries):row/kQueries;
        if(writer!=b)continue;
        if(mapped && healthy) {
            uint32_t offset=uint32_t(source%kQueries)*kTop;
            DataCopy(logicalOut[row*kTop],tokens[offset],kTop);
            DataCopy(slotsOut[row*kTop],slots[offset],kTop);
        } else {
            DataCopy(logicalOut[row*kTop],invalid,kTop);
            DataCopy(slotsOut[row*kTop],invalid,kTop);
        }
    }
    // The URMA caller reuses UB for the Index planner in this same kernel.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    pipe.Destroy();
}
}
#endif
