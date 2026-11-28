#include "context.h"
#include "../op_kernel/pipeline_types.h"
#include <cstdint>

extern "C" uint32_t aclrtlaunch_v9_nextgen_finalize_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_streaming_resolve_update(uint32_t,void*,void*);

// V4-B1 consumes Global TopK already written to bindings[9]. Score/Local/Global
// are launched by the staged caller before this function on the same stream.
// This entry never launches legacy LI or waits on native Rotation event 8.
extern "C" uint32_t kvip_nextgen_finalize_and_resolve(
    void* bindings, void* update, void* workspace, void* stream,
    const rotating::RuntimeShape* shape, uint32_t workers,
    uint32_t prefetch_limit, uint32_t upstream_validated) {
    if (!bindings || !update || !workspace || !stream || !shape || upstream_validated != 1) return 100000;
    // Local/Global TopK do not yet publish device-side completion status.
    // Until V4-B2 adds that protocol, require a synchronized host validation.
    // Deliberately narrow V4-B1 admission; expand only after exact end-to-end
    // TopK, payload, CQ and ownership checks on the target device.
    if (shape->batch != 16 || shape->history != 131072 ||
        shape->resident != 16384 || shape->indexHeads != 32 ||
        shape->aicCount != 32 || shape->liCores != 32 ||
        shape->splitQ != 1 || shape->finalWorkers != 4 ||
        shape->rounds == 0 || shape->serviceLanes == 0 ||
        (workers != 1 && workers != 2 && workers != 4) ||
        prefetch_limit != 0 ||
        shape->batch * pipeline_ab::kResolveCores + 4 * workers >
            shape->aicCount * 2) {
        return 100000;
    }
    uint32_t result=aclrtlaunch_v9_nextgen_finalize_kernel(
        shape->liCores,stream,bindings);
    if (!result) {
        result=aclrtlaunch_v9_streaming_resolve_update(
            shape->batch*pipeline_ab::kResolveCores+4*workers,stream,workspace);
    }
    return result;
}
