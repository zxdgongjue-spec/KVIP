#pragma once
#include <stdint.h>
namespace li_serial {
struct Inputs {
    uint64_t query, weights, scales, pages;
    uint64_t packedQuery, packedWeights, packedScales, packedPages, tasks;
    uint32_t heads, history, stride, rows;
};
static_assert(sizeof(Inputs)==88,"Serial pack ABI");
}
