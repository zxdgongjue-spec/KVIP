// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "auto_route.h"

#if defined(__CCE_AICORE__)
#include "simt_api/asc_simt.h"
#include "simt_api/device_functions.h"
#include "simt_api/device_sync_functions.h"
#define KVIP_ROUTE_SIMT_FN __simt_callee__
#define KVIP_ROUTE_SIMT_UB __ubuf__
#define KVIP_ROUTE_SIMT_GM __gm__
#else
#define KVIP_ROUTE_SIMT_FN
#define KVIP_ROUTE_SIMT_UB
#define KVIP_ROUTE_SIMT_GM
#endif

namespace kvip_auto_route {
template<class T>
KVIP_ROUTE_SIMT_FN inline T ParallelLoad(const KVIP_ROUTE_SIMT_GM T* p) {
#if defined(__CCE_AICORE__)
    // CANN's load overloads take mutable GM pointers even for read-only loads.
    return asc_ldcg(const_cast<KVIP_ROUTE_SIMT_GM T*>(p));
#else
    return *p;
#endif
}
// One lane evaluates one request. The same immutable input is evaluated by
// every init block; only the block's own Pending/control cache lines are written.
struct ParallelRow {
    Pending pending;
    uint64_t bytes, work, active, unknown, wasHigh, firstReason;
};

KVIP_ROUTE_SIMT_FN inline bool ParallelValid(const KVIP_ROUTE_SIMT_UB Config* c) {
    return c->version == kVersion && c->owners && c->owners <= 65536 &&
        c->requests && c->feedback && c->pending && c->lowBudget &&
        c->lowBudget < c->highBudget && c->highBudget <= 65535 &&
        c->maxStepGap && c->maxStepGap <= 1024 && c->learning <= 1 &&
        c->fullTotal && c->fullTotal <= 2048 && c->fullWindow &&
        c->fullWindow <= 480 && c->fullSteps <= 65535 && c->fullCarry <= 1;
}

KVIP_ROUTE_SIMT_FN inline void EvaluateParallelRow(
    const KVIP_ROUTE_SIMT_UB Config* c, uint32_t batch, uint32_t history,
    uint32_t tokenBytes, uint32_t b, const KVIP_ROUTE_SIMT_UB Request* requests,
    uint32_t owner, const KVIP_ROUTE_SIMT_UB uint32_t* ends,
    const KVIP_ROUTE_SIMT_GM Feedback* feedback, KVIP_ROUTE_SIMT_UB ParallelRow* out) {
    auto words=reinterpret_cast<KVIP_ROUTE_SIMT_UB uint64_t*>(out);
    for(uint32_t i=0;i<sizeof(ParallelRow)/8;++i)words[i]=0;
    out->firstReason=UINT64_MAX;
    if(b>=batch)return;
    auto r=requests+b;auto p=&out->pending;
    p->owner=r->owner;p->generation=r->generation;p->step=r->step;
    p->queries=r->queries;p->tokenBytes=tokenBytes;
    uint64_t queries=0;
    for(uint32_t i=0;i<4;++i){
        uint64_t end=ends[i];if(end)++queries;p->scoreWork+=end;
        if(end>history)queries=100;
    }
    if(!r->queries && !queries)return;
    out->active=1;out->work=p->scoreWork;
    bool identity=r->owner<c->owners && r->owner==owner && r->generation &&
        r->step && r->queries>=1 && r->queries<=4 && queries==r->queries;
    for(uint32_t j=0;j<batch;++j)
        if(j!=b && requests[j].queries && requests[j].owner==r->owner)identity=false;
    uint64_t reason=Predicted;
    if(!identity)reason=InvalidIdentity;
    else if(!c->learning)reason=LearningDisabled;
    else {
        auto f=feedback+r->owner;
        // Cache-bypassing loads see the preceding successful graph's feedback
        // without each scalar block invalidating every owner's cache line.
        uint64_t generation=ParallelLoad(&f->generation),step=ParallelLoad(&f->step);
        uint64_t previousQueries=ParallelLoad(&f->queries),necessary=ParallelLoad(&f->necessary);
        uint64_t previousMode=ParallelLoad(&f->mode),previousBytes=ParallelLoad(&f->tokenBytes);
        bool valid=ParallelLoad(&f->valid)==kValid && generation && step &&
            previousQueries>=1 && previousQueries<=4 && necessary<=8192 && previousMode<=1;
        bool learnable=!valid || r->generation>generation ||
            (r->generation==generation && r->step>step);
        p->previousStep=valid?step:0;p->learnable=learnable;
        if(!learnable)reason=StaleStep;
        else if(!valid || r->generation!=generation)reason=Unknown;
        else if(r->step-step>c->maxStepGap)reason=StaleStep;
        else if(previousBytes!=tokenBytes)reason=ShapeChange;
        else {
            uint32_t numerator=uint32_t(necessary)*uint32_t(r->queries)+uint32_t(previousQueries)-1;
            uint32_t predicted=previousQueries==1?numerator:previousQueries==2?numerator>>1:
                previousQueries==3?numerator/3:numerator>>2;
            out->bytes=uint64_t(predicted)*tokenBytes;out->wasHigh=previousMode!=0;
        }
    }
    if(reason!=Predicted){out->unknown=1;out->firstReason=uint64_t(b)*8+reason;}
}

KVIP_ROUTE_SIMT_FN inline void CombineParallelRows(
    KVIP_ROUTE_SIMT_UB ParallelRow* a,const KVIP_ROUTE_SIMT_UB ParallelRow* b) {
    a->bytes+=b->bytes;a->work+=b->work;a->active+=b->active;
    a->unknown|=b->unknown;a->wasHigh|=b->wasHigh;
    if(b->firstReason<a->firstReason)a->firstReason=b->firstReason;
}

KVIP_ROUTE_SIMT_FN inline void FinishParallelRow(
    const KVIP_ROUTE_SIMT_UB Config* c,const KVIP_ROUTE_SIMT_UB ParallelRow* total,
    KVIP_ROUTE_SIMT_UB Pending* pending) {
    uint64_t budget=total->wasHigh?c->lowBudget:c->highBudget;
    pending->mode=!total->unknown && total->active && total->work &&
        total->bytes*1024<=budget*total->work?0:1;
    pending->reason=!total->active?Unknown:total->unknown?total->firstReason%8:Predicted;
}

#if defined(__CCE_AICORE__)
struct ParallelScratch {
    uint32_t noPrefetch, valid;
    Config config;
    Request requests[32];
    uint32_t owners[32], ends[32][4];
    ParallelRow rows[32];
};
static_assert(sizeof(ParallelScratch)<=16384,"Route scratch exceeds existing init buffer");

template<class Context>
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void EvaluateParallel(
    KVIP_ROUTE_SIMT_GM Context* ctx,uint32_t publishRow,KVIP_ROUTE_SIMT_UB uint32_t* words) {
    auto scratch=reinterpret_cast<KVIP_ROUTE_SIMT_UB ParallelScratch*>(words);
    uint32_t lane=threadIdx.x,batch=ctx->shape.batch;
    auto c=&scratch->config;
    if(!lane){scratch->noPrefetch=0;scratch->valid=0;}
    if(lane<16)reinterpret_cast<KVIP_ROUTE_SIMT_UB uint64_t*>(c)[lane]=
        ParallelLoad(reinterpret_cast<const KVIP_ROUTE_SIMT_GM uint64_t*>(ctx->autoRoute)+lane);
    asc_syncthreads();
    if(!lane)scratch->valid=ParallelValid(c) && batch && batch<=16 && publishRow<batch &&
        c->pending%128==0 && ctx->admissionControls && ctx->admissionControls%128==0;
    asc_syncthreads();
    if(!scratch->valid)return;
    scratch->owners[lane]=0;
    for(uint32_t i=0;i<4;++i)scratch->ends[lane][i]=0;
    if(lane<batch){
        auto r=reinterpret_cast<const KVIP_ROUTE_SIMT_GM Request*>(c->requests)+lane;
        // UB-resident objects cannot invoke the implicit local-memory
        // assignment operator in the device compiler.
        scratch->requests[lane].owner=ParallelLoad(&r->owner);
        scratch->requests[lane].generation=ParallelLoad(&r->generation);
        scratch->requests[lane].step=ParallelLoad(&r->step);
        scratch->requests[lane].queries=ParallelLoad(&r->queries);
        scratch->owners[lane]=ctx->rowOwners?
            ParallelLoad(reinterpret_cast<const KVIP_ROUTE_SIMT_GM uint32_t*>(ctx->rowOwners)+lane):lane;
        for(uint32_t i=0;i<4;++i)scratch->ends[lane][i]=ctx->queryEnds?
            ParallelLoad(reinterpret_cast<const KVIP_ROUTE_SIMT_GM uint32_t*>(ctx->queryEnds)+lane*4+i):
            (i<scratch->requests[lane].queries?ctx->shape.history:0);
    }
    asc_syncthreads();
    EvaluateParallelRow(c,batch,ctx->shape.history,ctx->shape.tokenBytes,lane,scratch->requests,
        scratch->owners[lane],scratch->ends[lane],
        reinterpret_cast<const KVIP_ROUTE_SIMT_GM Feedback*>(c->feedback),scratch->rows+lane);
    asc_syncthreads();
    // Only aggregate fields change. Every lane's Pending record remains intact.
    for(uint32_t distance=16;distance;distance>>=1){
        if(lane<distance)CombineParallelRows(scratch->rows+lane,scratch->rows+lane+distance);
        asc_syncthreads();
    }
    auto p=&scratch->rows[publishRow].pending;
    if(!lane){FinishParallelRow(c,scratch->rows,p);scratch->noPrefetch=p->mode==0;}
    asc_syncthreads();
    if(lane<16)asc_stcg(reinterpret_cast<KVIP_ROUTE_SIMT_GM uint64_t*>(c->pending)+publishRow*16+lane,
        reinterpret_cast<KVIP_ROUTE_SIMT_UB uint64_t*>(p)[lane]);
    if(publishRow%8==0){
        uint32_t rows=batch-publishRow;if(rows>8)rows=8;
        if(lane<rows*4){
            uint32_t field=lane%4;
            int32_t value=field==0?(p->mode?int32_t(c->fullTotal):0):
                field==1?int32_t(c->fullWindow):field==2?int32_t(c->fullSteps):int32_t(c->fullCarry);
            asc_stcg(reinterpret_cast<KVIP_ROUTE_SIMT_GM int32_t*>(ctx->admissionControls)+publishRow*4+lane,value);
        }
    }
    asc_threadfence();asc_syncthreads();
}
#endif
}  // namespace kvip_auto_route
#undef KVIP_ROUTE_SIMT_FN
#undef KVIP_ROUTE_SIMT_UB
#undef KVIP_ROUTE_SIMT_GM
