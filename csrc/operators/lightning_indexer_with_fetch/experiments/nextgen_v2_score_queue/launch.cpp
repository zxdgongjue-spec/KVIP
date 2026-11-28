#include <stdint.h>
extern "C" uint32_t aclrtlaunch_v9_b16_score_clip_kernel(uint32_t,void*,void*);
extern "C" uint32_t v9_b16_score_clip(void* bindings,void* stream) {
    if(!bindings || !stream) return 100000;
    return aclrtlaunch_v9_b16_score_clip_kernel(32,stream,bindings);
}
