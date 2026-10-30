#pragma once
#include "contract_device.h"
#include "contract_parallel.h"
namespace rotating {
// Keep surviving rows at their existing positions. Only holes and truncated
// tails are reassigned. Metadata scans are parallel; Key copies are dirty-only.
__simt_callee__ inline void RebuildIndexCache(__gm__ Context* ctx, uint32_t b, bool tail,
                                             __ubuf__ uint32_t* scratch) {
    using namespace contract_simt;
    auto c=ContractOf(ctx);
    if(asc_ldcg(ContractPtr<int32_t>(c->status)+b)!=kPending)return;
    uint32_t tid=threadIdx.x,length=ContractLength(c,b),budget=ContractBudget(c,b),pool=ContractPool(c,b);
    auto state=ContractPtr<int32_t>(IndexBase(c,b));
    auto reverse=ContractPtr<int32_t>(c->reverse)+uint64_t(pool)*c->resident_capacity;
    auto forward=ContractPtr<int32_t>(c->forward)+uint64_t(pool)*c->candidate_capacity;
    auto target=ContractPtr<int32_t>(c->next_reverse)+b*c->resident_capacity;
    auto bits=ContractPtr<uint32_t>(c->scores)+uint64_t(b)*4*c->score_stride/2;
    auto desired=ContractPtr<int32_t>(c->hot_tokens)+b*c->resident_capacity;
    auto hot=IndexIds(c,b,true),cold=IndexIds(c,b,false);
    auto hd=IndexDirty(c,b,true),cd=IndexDirty(c,b,false),pos=IndexPositions(c,b);
    auto jobs=pos+c->candidate_capacity;
    bool reset=!asc_ldcg(state) || asc_ldcg(state+4)!=int32_t(pool) ||
        asc_ldcg(state+5)!=int32_t(c->layer_id) || asc_ldcg(state+6)!=int32_t(budget);
    uint32_t oldh=reset?0:uint32_t(asc_ldcg(state+2)),oldc=reset?0:uint32_t(asc_ldcg(state+3));
    uint32_t oldlen=reset?0:uint32_t(asc_ldcg(state+1));
    for(uint32_t i=tid;i<(c->candidate_capacity+31)/32;i+=256)asc_stcg(bits+i,0U);
    for(uint32_t s=tid;s<budget;s+=256) {
        int32_t t=asc_ldcg(reverse+s);
        bool valid=t>=0 && uint32_t(t)<length && asc_ldcg(forward+t)==int32_t(s);
        asc_stcg(target+s,valid?t:-1);
    }
    for(uint32_t i=tid;i<c->resident_capacity+c->candidate_capacity;i+=256)asc_stcg(hd+i,0);
    if(reset)for(uint32_t i=tid;i<c->candidate_capacity;i+=256)asc_stcg(pos+i,-1);
    asc_threadfence(); asc_syncthreads();
    if(tail) {
        uint32_t reserved=ContractPtr<Row>(ctx->rows)[b].slots;
        for(uint32_t i=tid;i<reserved;i+=256)
            asc_stcg(target+ContractPtr<int32_t>(c->reserved_slots)[b*2048+i],-1);
        asc_threadfence(); asc_syncthreads();
        uint32_t n=ContractPtr<int32_t>(c->plan_count)[b];
        for(uint32_t i=tid;i<n;i+=256)
            asc_stcg(target+ContractPtr<int32_t>(c->planned_slots)[b*kSlots+i],
                ContractPtr<int32_t>(c->planned_tokens)[b*kSlots+i]);
        uint32_t npre=ContractPtr<Row>(ctx->rows)[b].slots;
        for(uint32_t i=tid;i<npre;i+=256) {
            int32_t s=ContractPtr<int32_t>(c->extra_prefetch_slots)[b*2048+i];
            if(s>=0)asc_stcg(target+s,ContractPtr<int32_t>(c->reserved_tokens)[b*2048+i]);
        }
    }
    asc_threadfence(); asc_syncthreads();
    uint32_t chunk=(budget+255)/256,begin=tid*chunk,end=begin+chunk;
    if(end>budget)end=budget;
    uint32_t count=0;
    for(uint32_t s=begin;s<end;++s)count+=asc_ldcg(target+s)>=0;
    auto prefix=ContractPrefixUB(count,scratch);
    uint32_t nh=prefix.total,nc=length-nh,rank=prefix.begin;
    for(uint32_t s=begin;s<end;++s) {
        int32_t t=asc_ldcg(target+s);
        if(t>=0) { asc_stcg(desired+rank++,t); asc_atomic_or(bits+t/32,1U<<(t%32)); }
    }
    asc_threadfence(); asc_syncthreads();
    // Gather incoming hot rows in stable resident-slot order, before any bank
    // is overwritten. Surviving hot entries below the new extent never move.
    chunk=(nh+255)/256; begin=tid*chunk; end=begin+chunk; if(end>nh)end=nh;
    count=0;
    for(uint32_t i=begin;i<end;++i) {
        int32_t t=asc_ldcg(desired+i),at=asc_ldcg(pos+t);
        count+=!(at<=-2 && uint32_t(-at-2)<nh);
    }
    prefix=ContractPrefixUB(count,scratch); rank=prefix.begin;
    for(uint32_t i=begin;i<end;++i) {
        int32_t t=asc_ldcg(desired+i),at=asc_ldcg(pos+t);
        if(!(at<=-2 && uint32_t(-at-2)<nh))asc_stcg(jobs+rank++,t);
    }
    // Cold incoming rows use logical order. This scans compact metadata, not
    // Key data; all 256 lanes cooperate, including first-bank construction.
    chunk=(length+255)/256; begin=tid*chunk; end=begin+chunk; if(end>length)end=length;
    count=0;
    for(uint32_t t=begin;t<end;++t) {
        int32_t at=asc_ldcg(pos+t);
        if(!((asc_ldcg(bits+t/32)>>(t%32))&1U) && !(at>=0 && uint32_t(at)<nc))++count;
    }
    prefix=ContractPrefixUB(count,scratch); rank=prefix.begin;
    for(uint32_t t=begin;t<end;++t) {
        int32_t at=asc_ldcg(pos+t);
        if(!((asc_ldcg(bits+t/32)>>(t%32))&1U) && !(at>=0 && uint32_t(at)<nc))
            asc_stcg(jobs+c->resident_capacity+rank++,int32_t(t));
    }
    asc_threadfence(); asc_syncthreads();
    // Freeze both sets of holes before modifying IDs or reverse positions.
    for(uint32_t bank=0;bank<2;++bank) {
        uint32_t n=bank?nc:nh,old=bank?oldc:oldh;
        auto ids=bank?cold:hot; auto dirty=bank?cd:hd;
        chunk=(n+255)/256; begin=tid*chunk; end=begin+chunk; if(end>n)end=n;
        count=0;
        for(uint32_t i=begin;i<end;++i) {
            int32_t t=i<old?asc_ldcg(ids+i):-1;
            bool member=t>=0 && uint32_t(t)<length &&
                bool((asc_ldcg(bits+t/32)>>(t%32))&1U)==!bank;
            asc_stcg(dirty+i,member?0:1); count+=!member;
        }
        prefix=ContractPrefixUB(count,scratch); rank=prefix.begin;
        for(uint32_t i=begin;i<end;++i)if(asc_ldcg(dirty+i))asc_stcg(dirty+i,int32_t(++rank));
    }
    asc_threadfence(); asc_syncthreads();
    int32_t lo=tail?0:ContractPtr<int32_t>(c->index_dirty)[b*2];
    int32_t hi=tail?0:ContractPtr<int32_t>(c->index_dirty)[b*2+1];
    for(uint32_t bank=0;bank<2;++bank) {
        uint32_t n=bank?nc:nh,offset=bank?c->resident_capacity:0;
        auto ids=bank?cold:hot; auto dirty=bank?cd:hd;
        for(uint32_t i=tid;i<n;i+=256) {
            int32_t move=asc_ldcg(dirty+i);
            int32_t t=move?asc_ldcg(jobs+offset+move-1):asc_ldcg(ids+i);
            if(move) { asc_stcg(ids+i,t); asc_stcg(pos+t,bank?int32_t(i):-int32_t(i)-2); }
            asc_stcg(dirty+i,int32_t(move || (t>=lo && t<hi)));
        }
    }
    for(uint32_t t=length+tid;t<oldlen;t+=256)asc_stcg(pos+t,-1);
    asc_threadfence(); asc_syncthreads();
    // Reuse the incoming lists for one compact, deterministic dirty-copy list.
    uint32_t extent=c->resident_capacity+c->candidate_capacity;
    chunk=(extent+255)/256; begin=tid*chunk; end=begin+chunk; if(end>extent)end=extent;
    count=0;
    for(uint32_t i=begin;i<end;++i)count+=asc_ldcg(hd+i)!=0;
    prefix=ContractPrefixUB(count,scratch); rank=prefix.begin;
    for(uint32_t i=begin;i<end;++i)if(asc_ldcg(hd+i))asc_stcg(jobs+rank++,int32_t(i));
    uint32_t jobsCount=prefix.total;
    asc_threadfence(); asc_syncthreads();
    auto keys=ContractPtr<uint32_t>(IndexKey(c,b,true)),scales=ContractPtr<uint32_t>(IndexScale(c,b,true));
    for(uint32_t j=tid;j<jobsCount*32;j+=256) {
        uint32_t index=uint32_t(asc_ldcg(jobs+j/32)),word=j%32;
        bool ishot=index<c->resident_capacity;
        uint32_t i=ishot?index:index-c->resident_capacity,t=uint32_t(asc_ldcg((ishot?hot:cold)+i));
        uint32_t page=ContractPtr<int32_t>(c->index_table)[b*c->index_columns+t/128],source=page*128+t%128;
        asc_stcg(keys+uint64_t(index)*32+word,asc_ldcg(ContractPtr<uint32_t>(c->key)+uint64_t(source)*32+word));
        if(!word)asc_stcg(scales+index,asc_ldcg(ContractPtr<uint32_t>(c->key_scale)+source));
    }
    asc_threadfence(); asc_syncthreads();
    for(uint32_t j=tid;j<jobsCount;j+=256)asc_stcg(hd+asc_ldcg(jobs+j),0);
    if(!tid) {
        asc_stcg(state+1,int32_t(length)); asc_stcg(state+2,int32_t(nh)); asc_stcg(state+3,int32_t(nc));
        asc_stcg(state+4,int32_t(pool)); asc_stcg(state+5,int32_t(c->layer_id)); asc_stcg(state+6,int32_t(budget));
        asc_stcg(state+7,int32_t(jobsCount));
        if(!tail)asc_stcg(ContractPtr<int32_t>(c->cold_counts)+b,int32_t(nc));
        asc_stcg(state,tail?0:1); // Tail publication still belongs to CQ completion.
    }
    asc_threadfence();
}
}

#include "index_cache_incremental.h"
