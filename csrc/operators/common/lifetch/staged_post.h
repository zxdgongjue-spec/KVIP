#pragma once
#include <stdint.h>
#if defined(__CCE_AICORE__)
#define KVIP_POST_INLINE __aicore__ inline
#define KVIP_PACK_INLINE __simt_callee__ inline
#else
#define KVIP_POST_INLINE inline
#define KVIP_PACK_INLINE inline
#endif
// Host/device share only integer layout and exact sortable-key predicate logic.
// No owner/query state is published here; launch boundaries still join epochs.
namespace kvip_staged_post {
KVIP_POST_INLINE uint32_t FirstQuery(uint32_t group,uint32_t lane) { return lane*(group/2); }
KVIP_POST_INLINE uint32_t QueryCount(uint32_t group) { return group/2; }
KVIP_POST_INLINE uint64_t CanonicalOffset(uint32_t owner,uint32_t q,uint32_t history,uint32_t token) {
    return (uint64_t(owner)*4+q)*history+token;
}
KVIP_POST_INLINE uint64_t BitmapOffset(uint32_t owner,uint32_t q,uint32_t history,uint32_t token) {
    return (uint64_t(owner)*4+q)*(history/4)+token/4;
}
KVIP_POST_INLINE uint64_t ThresholdOffset(uint32_t owner,uint32_t q) {
    return uint64_t(owner)*(4*2048*3+512)+4*2048*3+q*128;
}
KVIP_PACK_INLINE uint8_t Pack4(uint16_t a,uint16_t b,uint16_t c,uint16_t d,
                             uint16_t theta,uint32_t token,uint32_t validEnd) {
    uint8_t value=0;
    if(token<validEnd && a>theta)value|=1;
    if(token+1<validEnd && b>theta)value|=4;
    if(token+2<validEnd && c>theta)value|=16;
    if(token+3<validEnd && d>theta)value|=64;
    return value;
}
}
#undef KVIP_POST_INLINE
#undef KVIP_PACK_INLINE
