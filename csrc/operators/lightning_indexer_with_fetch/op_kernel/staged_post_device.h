#pragma once
#include "../../common/lifetch/staged_post.h"
namespace kvip_staged_device {
// Four sortable uint16 keys form one byte. Each SIMT thread owns whole bytes.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void PackColdUB(
    __ubuf__ uint16_t* scores,__ubuf__ uint8_t* bits,uint32_t count,
    uint32_t scoreStride,uint32_t bitStride,uint32_t queries,uint32_t token,
    uint16_t theta0,uint16_t theta1,uint32_t valid0,uint32_t valid1) {
    const uint32_t bytes=count/4;
    for(uint32_t i=threadIdx.x;i<queries*bytes;i+=256) {
        uint32_t q=i/bytes,x=(i%bytes)*4;
        auto s=scores+q*scoreStride+x;
        bits[q*bitStride+x/4]=kvip_staged_post::Pack4(s[0],s[1],s[2],s[3],
            q?theta1:theta0,token+x,q?valid1:valid0);
    }
}
// Reduction output uses 512B/query; predicate output uses 256B/query.
// Read theta inside the VF after the existing MTE2 -> V dependency.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void PackColdThresholdUB(
    __ubuf__ uint16_t* scores,__ubuf__ uint8_t* bits,__ubuf__ uint16_t* theta,
    uint32_t queries,uint32_t token,uint32_t valid0,uint32_t valid1) {
    for(uint32_t i=threadIdx.x;i<queries*32;i+=256) {
        uint32_t q=i/32,x=(i%32)*4;auto s=scores+q*256+x;
        bits[q*256+x/4]=kvip_staged_post::Pack4(s[0],s[1],s[2],s[3],
            theta[q*128],token+x,q?valid1:valid0);
    }
}
}
