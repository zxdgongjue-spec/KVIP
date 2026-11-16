#include "kernel_operator.h"
#if defined(__CCE_AICORE__) && !V9_UBMEM
#include "context.h"
struct QLITilingData {
    uint32_t bSize,tSize,gSize,s1Size,s2Size,sparseCount,usedCoreNum,blockSize,maxBlockNumPerBatch,sparseMode;
    uint32_t keyStride0=16384,keyDequantScaleStride0=128;
    uint32_t resident,splitQ,aicCount;
};
#include "indexer/indexer.h"
#define KVIP_STAGED_VECTOR_ENTRY 1
#include "staged_indexer.h"
#endif
