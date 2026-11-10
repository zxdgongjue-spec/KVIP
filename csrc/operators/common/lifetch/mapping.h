// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace rotating {
// Dense callers use the identity defaults. The UBmem opt-in masks stale
// generations before any consumer interprets a virtual prefetch/ABC slot.
__simt_callee__ inline uint32_t DecodeMappingSlot(uint32_t value,uint32_t tag=0,uint32_t mask=UINT32_MAX) {
    if(mask==UINT32_MAX)return value;
    return value!=UINT32_MAX && (value&~mask)==tag ? value&mask : UINT32_MAX;
}
__simt_callee__ inline uint32_t EncodeMappingSlot(uint32_t slot,uint32_t tag,uint32_t mask) {
    // Reserve the all-ones slot in tagged mode: at the last epoch it would
    // otherwise collide with the invalid UINT32_MAX word.
    return mask==UINT32_MAX?slot:(slot<mask ? tag|slot : UINT32_MAX);
}
__simt_callee__ inline uint32_t ReadMappingSlot(__gm__ uint32_t* address,uint32_t tag,uint32_t mask) {
    return DecodeMappingSlot(asc_ldcg(address),tag,mask);
}
}
