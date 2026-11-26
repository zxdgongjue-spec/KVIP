#include <cstdint>

#include "runtime_shape.h"
#if KVIP_TEST_SIBLING_TRACE
#define aclrtlaunch_v9_li_large_two_cold aclrtlaunch_v9_li_large_two_cold_trace
#endif
extern "C" uint32_t aclrtlaunch_v9_li_large_two_cold(uint32_t, void*, void*, void*);
using Launch = uint32_t (*)(uint32_t, void*, void*, void*);
using TailLaunch = uint32_t (*)(uint32_t, void*, void*);

// Hot and final kernels come directly from the loaded candidate library.
extern "C" uint32_t kvip_test_two_cold_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape, Launch hot, TailLaunch tail) {
    if (!bindings || !terminal || !stream || !shape || !hot || !tail ||
        !rotating::NativeLargeBatch(shape->batch, shape->aicCount) || shape->serviceLanes != shape->batch ||
        shape->rounds != 3 || shape->liCores != 32 || shape->finalWorkers != 4) {
        return 100000;
    }
    uint32_t rc = hot(shape->liCores, stream, bindings, terminal);
    if (!rc) rc = aclrtlaunch_v9_li_large_two_cold(shape->liCores, stream, bindings, terminal);
    if (!rc) rc = tail(64, stream, terminal);
    return rc;
}
