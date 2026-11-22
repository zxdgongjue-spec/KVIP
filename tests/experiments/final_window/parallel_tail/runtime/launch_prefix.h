// Append to the isolated candidate's op_host/launch.cpp only. The production
// entry and device kernels remain unchanged. A non-split LI already executes
// its final TopK internally, so reject it BEFORE launching any work.
extern "C" uint32_t kvip_parallel_tail_prefix(
    void* bindings, void* stream, const rotating::RuntimeShape* shape) {
    if (!bindings || !stream || !shape || !shape->batch ||
        shape->batch > 16 || shape->finalWorkers != 4 ||
        shape->batch * 4 > shape->aicCount * 2 ||
        !rotating::UseSequenceSplit(*shape)) return 100000;
#define LI_LAUNCH_0 aclrtlaunch_v9_li_kernel
#define LI_LAUNCH_1 aclrtlaunch_v9_li_kernel_1
#define LI_LAUNCH_2 aclrtlaunch_v9_li_kernel_2
#define LI_LAUNCH_3(blocks, stream, bindings) 0U
#include "../../common/lifetch/li_launch.h"
#undef LI_LAUNCH_0
#undef LI_LAUNCH_1
#undef LI_LAUNCH_2
#undef LI_LAUNCH_3
}
