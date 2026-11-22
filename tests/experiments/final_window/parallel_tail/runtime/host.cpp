#include <cstdint>
#include "runtime_shape.h"
#include "terminal_runtime_abi.h"

extern "C" uint32_t aclrtlaunch_v9_parallel_terminal(uint32_t,void*,void*);
using Prefix = uint32_t (*)(void*,void*,const rotating::RuntimeShape*);

// A single host call captures unchanged LI 0/1/2 followed by the private tail.
// Prefix is resolved once from the isolated baseline bridge, never from GM.
extern "C" uint32_t kvip_parallel_tail_forward(
    Prefix prefix,void* bindings,void* terminal,void* stream,
    const rotating::RuntimeShape* shape) {
    if(!prefix || !bindings || !terminal || !stream || !shape ||
       !shape->batch || shape->batch>16 || shape->indexHeads!=32 ||
       shape->finalWorkers!=4 || shape->batch*4>shape->aicCount*2)
        return 100000;
    uint32_t result=prefix(bindings,stream,shape);
    if(result)return result;
    return aclrtlaunch_v9_parallel_terminal(shape->batch*4,stream,terminal);
}

extern "C" uint32_t kvip_parallel_tail_abi_version(){return terminal_runtime::kVersion;}
extern "C" uint32_t kvip_parallel_tail_binding_bytes(){return sizeof(terminal_runtime::Binding);}
