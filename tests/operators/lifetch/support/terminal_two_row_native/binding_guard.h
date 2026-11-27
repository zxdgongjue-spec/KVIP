// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace joint_tail {
// Experimental launch only. Validate live owners/routes every invocation. A checked capture
// supplies stable, nonoverlapping lease/tensor extents; dynamic output remaps use the original tail.
__aicore__ inline bool DirectRows(__gm__ pipeline_ab::Args* a,__gm__ rotating::Context* ctx) {
    auto u=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
    if(!u || !u[9] || !u[20] || !u[21] || !u[22] || !u[23] ||
       (ctx->rowOwners && ctx->rowOwners!=u[9]) || ctx->shape.batch!=32 ||
       ctx->outputStride<ctx->shape.resident || u[7]<ctx->shape.history) return false;
    const uint64_t size=uint64_t(32)*4*2048*sizeof(uint32_t);
    if(!Disjoint({u[22],size},{u[23],size}) ||
       (u[22]!=ctx->selected && !Disjoint({u[22],size},{ctx->selected,size})) ||
       (u[23]!=ctx->locations && !Disjoint({u[23],size},{ctx->locations,size})))return false;
    auto owner=reinterpret_cast<__gm__ volatile int32_t*>(u[9]);
    auto unpack=reinterpret_cast<__gm__ volatile int64_t*>(u[20]);
    auto valid=reinterpret_cast<__gm__ volatile int64_t*>(u[21]);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(u[9]),32*4);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(u[20]),128*8);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(u[21]),128*8);
    for(uint32_t r=0;r<32;++r) {
        if(owner[r]!=int32_t(r)) return false;
        for(uint32_t q=0;q<4;++q)if(!valid[r*4+q] || unpack[r*4+q]!=int64_t(r*4+q)) return false;
    }
    return true;
}
}
