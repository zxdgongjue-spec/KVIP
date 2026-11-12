#pragma once
extern "C" uint32_t aclrtlaunch_v9_li_pool_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_split_topk_local(uint32_t,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t,void*);
extern "C" uint32_t aclrtlaunch_split_topk_merge(uint32_t,void*,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,void*);
extern "C" uint32_t kvop_direct_update_outputs_early(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_fetch_update_kernel(uint32_t,void*,void*,void*);
