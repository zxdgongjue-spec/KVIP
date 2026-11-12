#pragma once
#include "prefetch.h"
#include "../../lidu/op_kernel/hot_slot_selection.h"

#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"

namespace rotating {
__simt_callee__ inline uint32_t SparseHash(uint32_t token) { return (token*0x9e3779b9U)>>19; }
}

namespace rotating {
__aicore__ inline __ubuf__ uint32_t* FinalPtr(LocalTensor<uint32_t> x) {return reinterpret_cast<__ubuf__ uint32_t*>(x.GetPhyAddr());}

#define LIFETCH_CLEAN_CACHE_LINES kvaf_device::CleanCacheLines
#include "../../common/lifetch/final_post.h"
#undef LIFETCH_CLEAN_CACHE_LINES
}
