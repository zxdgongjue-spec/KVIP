#pragma once
#ifndef V9_UBMEM_BALANCED_INDEX
#define V9_UBMEM_BALANCED_INDEX 0
#endif
#if V9_UBMEM_BALANCED_INDEX != 0 && V9_UBMEM_BALANCED_INDEX != 1
#error "V9_UBMEM_BALANCED_INDEX must be 0 or 1"
#endif
#include "../../../common/lifetch/index_cache_update.h"
