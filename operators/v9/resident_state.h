#pragma once
#include "contract_device.h"
#include "contract_parallel.h"
#include "owned_h0.h"

namespace rotating {
// Status rows are densely packed; scalar cache-line writeback could overwrite
// a different request's SIMT status. All status writes therefore use GM stores.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void FailContractPreflight(__gm__ Context* ctx, uint32_t b, int32_t error) {
    if (threadIdx.x) return;
    auto c = contract_simt::ContractOf(ctx);
    asc_stcg(contract_simt::ContractPtr<int32_t>(c->status) + b, error);
    asc_stcg(contract_simt::ContractPtr<int32_t>(c->error_detail) + b * 4, error);
    asc_atomic_cas(contract_simt::ContractPtr<int32_t>(c->sticky), 0, error);
    asc_threadfence();
}
template<int Stage = -1>
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void InitResidentState(__gm__ Context* ctx, uint32_t b, __ubuf__ uint32_t* scratch) {
    auto c = contract_simt::ContractOf(ctx);
    uint32_t tid = threadIdx.x;
    auto status = contract_simt::ContractPtr<int32_t>(c->status);
    auto detail = contract_simt::ContractPtr<int32_t>(c->error_detail) + b * 4;
    auto row = contract_simt::ContractPtr<uint32_t>(ctx->rows) + b * 256;
    auto hot = contract_simt::ContractPtr<uint32_t>(c->hot_bitmap) + b * ((c->candidate_capacity + 31) / 32);
    if constexpr (Stage != 1) {
    // Only ReserveContract writes mapping. Its completed reservation list is
    // the exact dirty set, including discarded prefetches. A fresh launcher
    // initializes mapping to -1 once; batch-row reuse clears the OLD row here.
    uint32_t previousSlots = asc_ldcg(row + 8);
    asc_syncthreads();
    for(uint32_t i=tid;i<previousSlots && i<2048;i+=256) {
        int32_t t=contract_simt::ContractPtr<int32_t>(c->reserved_tokens)[b*2048+i];
        if(t>=0 && uint32_t(t)<c->candidate_capacity)
            asc_stcg(contract_simt::ContractPtr<int32_t>(ctx->mapping)+uint64_t(b)*c->candidate_capacity+t,-1);
    }
    asc_threadfence(); asc_syncthreads();
    for (uint32_t i = tid; i < 256; i += 256) asc_stcg(row + i, 0U);
    for (uint32_t i = tid; i < kFinalWorkers * 32; i += 256)
        asc_stcg(contract_simt::ContractPtr<uint32_t>(ctx->owners + kFinalPostOffset) + b * kFinalWorkers * 32 + i, 0U);
    for (uint32_t i = tid; i < kRowStatusWords; i += 256)
        asc_stcg(contract_simt::ContractPtr<uint32_t>(ctx->rowStatus) + b * kRowStatusWords + i, 0U);
    for (uint32_t i = tid; i < 128; i += 256)
        asc_stcg(contract_simt::ContractPtr<uint32_t>(ctx->progress) + b * 128 + i, 0U);
    for (uint32_t i = b * 256 + tid; i < kRounds * kServiceLanes * 32; i += SIM_BATCH * 256)
        asc_stcg(contract_simt::ContractPtr<uint32_t>(ctx->events) + i, 0U);
    for (uint32_t i = tid; i < c->candidate_capacity / 4; i += 256)
        asc_stcg(contract_simt::ContractPtr<uint32_t>(ctx->hits) + b * c->candidate_capacity / 4 + i, 0U);
    for (uint32_t i = tid; i < 512; i += 256)
        asc_stcg(contract_simt::ContractPtr<uint32_t>(c->touched) + b * 512 + i, 0U);
    for (uint32_t i = tid; i < kSlots; i += 256) {
        if (i < c->queries * kTop) {
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->topk) + b * c->queries * kTop + i, -1);
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->destination) + b * c->queries * kTop + i, -1);
        }
    }
    for(uint32_t i=tid;i<2048;i+=256) {
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->extra_prefetch_slots)+b*2048+i,-1);
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->eligible_victims)+b*2048+i,-1);
    }
    if(tid<32)asc_stcg(contract_simt::ContractPtr<int32_t>(c->hot_ready)+b*32+tid,0);
    if (tid < 4) {
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->publish_meta)+b*4+tid,0);
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->hot_thresholds)+b*4+tid,0);
        asc_stcg(detail + tid, 0);
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->transfers) + b * 4 + tid, 0);
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->threshold_valid) + b * 4 + tid, 0);
    }
    if (!tid) {
        asc_stcg(status + b, int32_t(kPending));
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->extra_prefetch_counts)+b,0);
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->eligible_counts)+b,0);
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->resident_lengths) + b, 0);
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->miss_counts) + b, 0);
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->plan_count) + b, 0);
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->logical_misses) + b, 0);
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->hot_counts) + b, 0);
        if (!b) {
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->miss_counts) + c->batch, 0);
            asc_stcg(contract_simt::ContractPtr<uint32_t>(ctx->count), 0U);
        }
    }
    asc_threadfence(); asc_syncthreads();
    if(!tid) {
        int32_t error=0;
        bool dummy=contract_simt::ContractPtr<int32_t>(c->profile)[0]==2;
        if(contract_simt::ContractPtr<int32_t>(c->sticky)[0])error=kPoisoned;
        if(!error && (dummy || contract_simt::ContractPtr<int32_t>(c->row_modes)[0]==0)) {
            for(uint32_t r=0;r<c->batch;++r)
                if(!dummy && contract_simt::ContractPtr<int32_t>(c->row_modes)[r]!=0)error=kInvalid;
            if(!error)error=kSkipped;
        }
        asc_stcg(detail,error);
    }
    asc_threadfence();asc_syncthreads();
    int32_t error=asc_ldcg(detail);
    asc_syncthreads();
    if(error==kSkipped) {
        if(!tid){asc_stcg(detail,0);asc_stcg(status+b,int32_t(kSkipped));}
        return;
    }
    }
    if constexpr (Stage != 0) {
    if(asc_ldcg(status+b)==kSkipped) return;
    int32_t error=asc_ldcg(detail);
    int32_t length = contract_simt::ContractPtr<int32_t>(c->candidate_lens)[b];
    int32_t logical = contract_simt::ContractPtr<int32_t>(c->logical_lens)[b];
    int32_t budget = contract_simt::ContractPtr<int32_t>(c->budgets)[b];
    int32_t p = contract_simt::ContractPtr<int32_t>(c->pool_rows)[b];
    int32_t dirtyLo=contract_simt::ContractPtr<int32_t>(c->index_dirty)[b*2];
    int32_t dirtyHi=contract_simt::ContractPtr<int32_t>(c->index_dirty)[b*2+1];
    uint64_t indexCache=contract_simt::IndexBase(c,b);
    if(!indexCache || indexCache%128 || dirtyLo<0 || dirtyHi<dirtyLo || uint32_t(dirtyHi)>c->candidate_capacity)
        error=kInvalid;
    if (c->batch != SIM_BATCH || c->queries != SIM_QUERIES || c->candidate_capacity != SIM_HISTORY ||
        c->resident_capacity != SIM_RESIDENT || c->score_stride != SIM_HISTORY+128 || p < 0 || uint32_t(p) >= c->max_requests ||
        length <= 0 || uint32_t(length) > c->candidate_capacity || logical < length ||
        budget < 2048 || uint32_t(budget) > c->resident_capacity ||
        uint64_t(budget) + logical - length > uint64_t(c->resident_columns) * 128 ||
        ctx->prefetchLimit > 2048) error = kInvalid;
    for (uint32_t r = tid; r < c->batch; r += 256) {
        if (contract_simt::ContractPtr<int32_t>(c->cumulative_query)[r] != int32_t((r + 1) * c->queries) ||
            contract_simt::ContractPtr<int32_t>(c->candidate_lens)[r] != length ||
            contract_simt::ContractPtr<int32_t>(c->logical_lens)[r] != logical ||
            contract_simt::ContractPtr<int32_t>(c->budgets)[r] != budget ||
            contract_simt::ContractPtr<int32_t>(c->row_modes)[r] != 2 ||
            (r != b && contract_simt::ContractPtr<int32_t>(c->pool_rows)[r] == p)) error = kInvalid;
    }
    if(error)asc_atomic_cas(detail,0,error);
    asc_threadfence();asc_syncthreads();error=asc_ldcg(detail);asc_syncthreads();
    if (!error) {
        // Victim validation is local to this VF; do not atomically round-trip GM.
        auto membership = scratch;
        for(uint32_t word=tid;word<512;word+=256) membership[word]=0;
        asc_syncthreads();
        for(uint32_t i=tid;i<2048;i+=256) {
            int32_t slot=contract_simt::ContractPtr<int32_t>(c->victims)[b*2048+i];
            if(slot<0 || slot>=budget){error=kInvalid;continue;}
            uint32_t mask=1U<<(uint32_t(slot)%32);
            if(asc_atomic_or(membership+slot/32,mask)&mask)error=kInvalid;
        }
        asc_threadfence();asc_syncthreads();
        for(uint32_t word=tid;word<512;word+=256) {
            if(membership[word]!=contract_simt::ContractPtr<uint32_t>(c->victim_bitmap)[b*512+word])
                error=kInvalid;

        }
        uint32_t pages = (uint32_t(length) + 127) / 128;
        for (uint32_t page = tid; page < pages && !error; page+=256) {
            int32_t block = contract_simt::ContractPtr<int32_t>(c->index_table)[b * c->index_columns + page];
            if (block < 0 || uint32_t(block) >= c->index_blocks) { error = kInvalid; break; }
            uint64_t d = (uint64_t(p) * c->num_layers + c->layer_id) * c->qcap + page;
            if (d >= c->table_capacity || !contract_simt::ContractPtr<uint8_t>(c->valid_table)[d]) {
                error = kDirectory; break;
            }
            uint64_t base = contract_simt::ContractPtr<uint64_t>(c->address_table)[d];
            if (!base || base > UINT64_MAX - 128 * 656) error = kDirectory;
        }
        pages = (uint32_t(budget + logical - length) + 127) / 128;
        for (uint32_t page = tid; page < pages && !error; page+=256) {
            int32_t block = contract_simt::ContractPtr<int32_t>(c->resident_table)[b * c->resident_columns + page];
            if (block < 0 || uint32_t(block) >= c->resident_blocks) error = kInvalid;
        }
    }
    if(error)asc_atomic_cas(detail,0,error);
    asc_threadfence();asc_syncthreads();
    error=asc_ldcg(detail);
    if(!error) {
        auto forward=contract_simt::ContractPtr<int32_t>(c->forward)+uint64_t(p)*c->candidate_capacity;
        auto reverse=contract_simt::ContractPtr<int32_t>(c->reverse)+uint64_t(p)*c->resident_capacity;
        bool owned=contract_simt::IndexOwned(c,b);
        uint32_t words=(c->candidate_capacity+31)/32;
        // Owned scans use physical hot rows, never the derived hot_tokens list.
        // The bounded UB bitmap covers up to 1M candidate tokens. Larger
        // shapes retain GM bitmap atomics without increasing UB consumption.
        // Exact snapshot comparison catches external restore/reorder/mapping
        // writes even when candidate length and request identity are unchanged.
        if(owned) {
            if(!tid)asc_stcg(contract_simt::ContractPtr<int32_t>(indexCache)+27,0);
            asc_threadfence();asc_syncthreads();
            if(ReuseOwnedH0(ctx,b,scratch))return;
        }
        if(!owned || words>32768) {
            for(uint32_t w=tid;w<words;w+=256)asc_stcg(hot+w,0U);
            asc_threadfence();asc_syncthreads();
        }
        if(owned) {
            bool localBits=words<=32768;
            auto bits=scratch+8;
            if(localBits)for(uint32_t w=tid;w<words;w+=256)bits[w]=0;
            asc_syncthreads();
            uint32_t count=0,mismatch=0;
            auto state=contract_simt::ContractPtr<int32_t>(indexCache);
            auto positions=contract_simt::IndexPositions(c,b);
            auto ids=contract_simt::IndexIds(c,b,true);
            int32_t nh=asc_ldcg(state+2);
            // Stage the small hot directory once. The inverse check then reads
            // UB rather than chasing a third dependent random GM address.
            auto localIds=bits+words;
            bool stageIds=nh>=0 && uint32_t(nh)<=c->resident_capacity &&
                8+words+uint32_t(nh)<=32784;
            if(stageIds)for(uint32_t i=tid;i<uint32_t(nh);i+=256)localIds[i]=uint32_t(asc_ldcg(ids+i));
            asc_syncthreads();
            for(uint32_t slot=tid;slot<uint32_t(budget);slot+=256) {
                int32_t t=asc_ldcg(reverse+slot);
                if(t<0 || t>=length || asc_ldcg(forward+t)!=int32_t(slot))continue;
                ++count;
                if(localBits)asc_atomic_or(bits+t/32,1U<<(t%32));
                else asc_atomic_or(hot+t/32,1U<<(t%32));
                int32_t at=asc_ldcg(positions+t);
                mismatch+=nh<0 || uint32_t(nh)>c->resident_capacity || at>-2 || int64_t(-int64_t(at)-2)>=nh ||
                    (stageIds?int32_t(localIds[-int64_t(at)-2]):asc_ldcg(ids+(-int64_t(at)-2)))!=t;
            }
            asc_syncthreads();
            if(localBits)for(uint32_t w=tid;w<words;w+=256)asc_stcg(hot+w,bits[w]);
            // Both sums are bounded by 16384, so pack them into one scan.
            auto total=ContractPrefixUB(count+(mismatch<<16),scratch);
            if(!tid)asc_stcg(state+19,(total.total>>16)?2:1);
            total.total&=0xffffU;
            if(!total.total)for(uint32_t i=tid;i<2048;i+=256)
                asc_stcg(contract_simt::ContractPtr<int32_t>(c->eligible_victims)+b*2048+i,
                    contract_simt::ContractPtr<int32_t>(c->victims)[b*2048+i]);
            asc_threadfence();asc_syncthreads();
            if(!tid) {
                asc_stcg(contract_simt::ContractPtr<int32_t>(c->hot_counts)+b,int32_t(total.total));
                if(!total.total) {
                    asc_stcg(contract_simt::ContractPtr<int32_t>(c->eligible_counts)+b,2048);
                    asc_stcg(contract_simt::ContractPtr<int32_t>(c->hot_ready)+b*32+1,1);
                }
            }
        } else {
        uint32_t chunk=(uint32_t(budget)+255)/256,lo=tid*chunk,hi=lo+chunk,count=0;
        if(hi>uint32_t(budget))hi=uint32_t(budget);
        // Coalesced GM/UB staging; retain slot-order compaction below.
        for(uint32_t slot=tid;slot<uint32_t(budget);slot+=256) {
            int32_t t=asc_ldcg(reverse+slot);
            bool valid=t>=0 && t<length && asc_ldcg(forward+t)==int32_t(slot);
            scratch[8+slot]=valid?uint32_t(t):UINT32_MAX;
        }
        asc_syncthreads();
        for(uint32_t slot=lo;slot<hi;++slot) count+=scratch[8+slot]!=UINT32_MAX;
        auto prefix=ContractPrefixUB(count,scratch);
        uint32_t at=prefix.begin;
        for(uint32_t slot=lo;slot<hi;++slot) {
            int32_t t=int32_t(scratch[8+slot]);
            if(t>=0) {
                asc_atomic_or(hot+t/32,1U<<(t%32));
                asc_stcg(contract_simt::ContractPtr<int32_t>(c->hot_tokens)+b*c->resident_capacity+at++,t);
            }
        }
        if(!prefix.total)for(uint32_t i=tid;i<2048;i+=256)
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->eligible_victims)+b*2048+i,
                contract_simt::ContractPtr<int32_t>(c->victims)[b*2048+i]);
        asc_threadfence();asc_syncthreads();
        if(!tid) {
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->hot_counts)+b,int32_t(prefix.total));
            if(!prefix.total) {
                asc_stcg(contract_simt::ContractPtr<int32_t>(c->eligible_counts)+b,2048);
                asc_stcg(contract_simt::ContractPtr<int32_t>(c->hot_ready)+b*32+1,1);
            }
        }
        }
    } else if(!tid) {
        asc_stcg(status+b,error);
        asc_atomic_cas(contract_simt::ContractPtr<int32_t>(c->sticky),0,error);
    }
    asc_threadfence();
    }
}

// Registered-host-region checks run on all lanes, rather than a scalar
// per-page loop on the critical init path. Index/directory validity was already
// checked by InitResidentState before this address is dereferenced.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ValidateContractRemote(__gm__ Context* ctx,uint32_t b,uint64_t low,uint64_t size) {
    using namespace contract_simt;
    auto c=ContractOf(ctx);
    if(asc_ldcg(ContractPtr<int32_t>(c->status)+b)!=kPending)return;
    uint32_t length=ContractLength(c,b);
    for(uint32_t page=threadIdx.x;page<(length+127)/128;page+=256) {
        uint64_t address=ContractPtr<uint64_t>(c->address_table)[ContractDirectory(c,b,page*128)];
        uint64_t bytes=uint64_t(length-page*128<128?length-page*128:128)*kTokenBytes;
        if(address<low || address-low>size || bytes>size-(address-low)) {
            asc_stcg(ContractPtr<int32_t>(c->status)+b,int32_t(kInvalid));
            asc_stcg(ContractPtr<int32_t>(c->error_detail)+b*4,int32_t(kInvalid));
            asc_atomic_cas(ContractPtr<int32_t>(c->sticky),0,int32_t(kInvalid));
        }
    }
    asc_threadfence();
}

// CQ completion is checked by the caller before invoking this single request
// publisher. SIMT barriers separate all clears from all installs.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void PublishResidentState(__gm__ Context* ctx, uint32_t b, int32_t transportError,
                                 uint32_t submittedFinal, __ubuf__ uint32_t* scratch) {
    auto c = contract_simt::ContractOf(ctx); uint32_t tid = threadIdx.x;
    auto row = contract_simt::ContractPtr<Row>(ctx->rows) + b;
    auto detail = contract_simt::ContractPtr<int32_t>(c->error_detail) + b * 4;
    int32_t prior = contract_simt::ContractPtr<int32_t>(c->status)[b];
    if (prior == kSkipped) return;
    int32_t error = prior < 0 ? prior : (transportError ? kCommunication : 0);
    if (!error && contract_simt::ContractPtr<int32_t>(c->sticky)[0] && !contract_simt::ContractPtr<int32_t>(c->plan_count)[b])
        error = kPoisoned;
    if (!tid) {
        auto transfer = contract_simt::ContractPtr<int32_t>(c->transfers) + b * 4;
        asc_stcg(transfer, int32_t(row->requests));
        asc_stcg(transfer + 1, int32_t(submittedFinal));
        asc_stcg(transfer + 2, int32_t(row->requests + submittedFinal));
        asc_stcg(detail + 1, int32_t(row->requests || submittedFinal));
    }
    if (error) {
        if (!tid) {
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->status) + b, error);
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->miss_counts) + b, -1);
            asc_stcg(detail, error);
            asc_atomic_cas(contract_simt::ContractPtr<int32_t>(c->sticky), 0, error);
        }
        return;
    }
    uint32_t p = contract_simt::ContractPool(c, b), budget = contract_simt::ContractBudget(c, b);
    auto forward = contract_simt::ContractPtr<int32_t>(c->forward) + uint64_t(p) * c->candidate_capacity;
    auto reverse = contract_simt::ContractPtr<int32_t>(c->reverse) + uint64_t(p) * c->resident_capacity;
    auto touched = contract_simt::ContractPtr<uint32_t>(c->touched) + b * 512;
    auto state=contract_simt::ContractPtr<int32_t>(contract_simt::IndexBase(c,b));
    bool saveH0=contract_simt::IndexOwned(c,b) && asc_ldcg(state+20)==1;
    auto snapshot=contract_simt::IndexDirty(c,b,true);
    auto savedBits=reinterpret_cast<__gm__ uint32_t*>(contract_simt::IndexDirty(c,b,false));
    auto meta=contract_simt::ContractPtr<int32_t>(c->publish_meta)+b*4;
    bool planned=asc_ldcg(meta+3)==1;
    uint32_t changed=planned?uint32_t(asc_ldcg(meta+1)):budget;
    for (uint32_t i = tid; i < changed; i += 256) {
        uint32_t s=planned?uint32_t(asc_ldcg(contract_simt::ContractPtr<int32_t>(c->changed_slots)+b*c->resident_capacity+i)):i;
        if (!planned && !((touched[s / 32] >> (s % 32)) & 1U)) continue;
        int32_t old = asc_ldcg(reverse + s);
        if (old >= 0 && uint32_t(old) < c->candidate_capacity && asc_ldcg(forward + old) == int32_t(s))
            asc_stcg(forward + old, -1);
        asc_stcg(reverse + s, -1);
        if(saveH0) {
            int32_t priorToken=asc_ldcg(snapshot+s);
            if(priorToken>=0)asc_atomic_and(savedBits+priorToken/32,~(1U<<(priorToken%32)));
            asc_stcg(snapshot+s,-1);
        }
    }
    asc_threadfence(); asc_syncthreads();
    uint32_t misses = uint32_t(contract_simt::ContractPtr<int32_t>(c->plan_count)[b]);
    auto tokens = contract_simt::ContractPtr<int32_t>(c->planned_tokens) + b * kSlots;
    auto slots = contract_simt::ContractPtr<int32_t>(c->planned_slots) + b * kSlots;
    for (uint32_t i = tid; i < misses; i += 256) {
        asc_stcg(forward + tokens[i], slots[i]);
        asc_stcg(reverse + slots[i], tokens[i]);
        if(saveH0) {
            asc_stcg(snapshot+slots[i],tokens[i]);
            asc_atomic_or(savedBits+tokens[i]/32,1U<<(tokens[i]%32));
        }
    }
    for(uint32_t i=tid;i<row->slots;i+=256) {
        int32_t s=contract_simt::ContractPtr<int32_t>(c->extra_prefetch_slots)[b*2048+i];
        if(s>=0) {
            int32_t t=contract_simt::ContractPtr<int32_t>(c->reserved_tokens)[b*2048+i];
            asc_stcg(forward+t,s); asc_stcg(reverse+s,t);
            if(saveH0) {
                asc_stcg(snapshot+s,t);asc_atomic_or(savedBits+t/32,1U<<(t%32));
            }
        }
    }
    asc_threadfence(); asc_syncthreads();
    uint32_t reused=planned?uint32_t(asc_ldcg(meta+2)):0;
    if(asc_ldcg(meta)==15) {
        auto staged=contract_simt::ContractPtr<int32_t>(c->staged_destination)+b*8192;
        auto destination=contract_simt::ContractPtr<int32_t>(c->destination)+b*8192;
        for(uint32_t i=tid;i<8192;i+=256)asc_stcg(destination+i,asc_ldcg(staged+i));
    } else {
    for(uint32_t q=0;q<4;++q) {
        auto selected=contract_simt::ContractPtr<int32_t>(c->topk)+(b*4+q)*kTop;
        auto output=contract_simt::ContractPtr<int32_t>(c->destination)+(b*4+q)*kTop;
        auto loc=contract_simt::ContractPtr<int32_t>(ctx->locations)+b*kSlots+q*kTop;
        uint32_t offset=0;
        for(uint32_t pass=0;pass<2;++pass) {
            uint32_t mask=0,count=0;
            for(uint32_t j=0;j<8;++j) {
                int32_t t=asc_ldcg(selected+tid*8+j);
                if(t>=0 && uint32_t(contract_simt::ContractHot(c,b,uint32_t(t)))==pass){mask|=1U<<j;++count;}
            }
            auto prefix=ContractPrefixUB(count,scratch);uint32_t at=offset+prefix.begin;
            for(uint32_t j=0;j<8;++j)if(mask&(1U<<j))asc_stcg(output+at++,asc_ldcg(loc+tid*8+j));
            offset+=prefix.total;
        }
    }
    }
    if(!planned) {
        for(uint32_t i=tid;i<misses;i+=256)
            reused+=contract_simt::ContractPtr<int32_t>(ctx->mapping)[uint64_t(b)*c->candidate_capacity+tokens[i]]>=0;
        reused=ContractPrefixUB(reused,scratch).total;
    }
    asc_threadfence();asc_syncthreads();
    if(!tid) {
        uint32_t adopted=uint32_t(contract_simt::ContractPtr<int32_t>(c->extra_prefetch_counts)[b])+reused;
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->transfers) + b * 4 + 3, int32_t(row->requests - adopted));
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->cursors) + p, contract_simt::ContractPtr<int32_t>(c->proposed_cursor)[b]);
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->miss_counts) + b, int32_t(misses));
        asc_atomic_add(contract_simt::ContractPtr<int32_t>(c->miss_counts) + c->batch, int32_t(misses));
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->resident_lengths) + b,
                 int32_t(budget) + contract_simt::ContractPtr<int32_t>(c->logical_lens)[b] - int32_t(contract_simt::ContractLength(c, b)));
        asc_threadfence();
        if(saveH0)asc_stcg(state+26,asc_ldcg(state+2));
        asc_threadfence();
        asc_stcg(state,1);
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->status) + b, int32_t(kOk));
    }
    asc_threadfence();
}
}
