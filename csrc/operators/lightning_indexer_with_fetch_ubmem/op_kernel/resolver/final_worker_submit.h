#pragma once
#include "ubmem/fused_batch_get_device.h"
namespace rotating {
__aicore__ inline __ubuf__ uint32_t* FinalPtr(LocalTensor<uint32_t> x) {return reinterpret_cast<__ubuf__ uint32_t*>(x.GetPhyAddr());}
#define LIFETCH_CLEAN_CACHE_LINES ubmem_device::CleanCacheLines
#include "../../../common/lifetch/final_post.h"
#undef LIFETCH_CLEAN_CACHE_LINES
}
