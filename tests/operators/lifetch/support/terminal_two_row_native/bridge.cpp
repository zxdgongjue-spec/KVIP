// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
extern "C" uint32_t aclrtlaunch_v9_test_two_row_terminal(uint32_t,void*,void*,void*,uint64_t);
extern "C" uint32_t kvip_test_two_rows(uint32_t cores,void* stream,void* binding,void* workspace,uint64_t ticks){
    if(cores!=64 || !stream || !binding || !workspace || !ticks) return 1;
    return aclrtlaunch_v9_test_two_row_terminal(cores,stream,binding,workspace,ticks);
}

namespace {
using Tail=uint32_t(*)(uint32_t,void*,void*);
using Hot=uint32_t(*)(uint32_t,void*,void*,void*);
using Pipeline=uint32_t(*)(void*,void*,void*,const void*,Hot,Tail);
struct Bound {void* binding;void* stream;void* workspace;uint64_t guard;bool called;};
thread_local Bound* active=nullptr;
uint32_t BoundTail(uint32_t cores,void* stream,void* binding){
    auto a=active;
    if(!a || a->called || a->binding!=binding || a->stream!=stream)return 100000;
    a->called=true;
    return kvip_test_two_rows(cores,stream,binding,a->workspace,a->guard);
}
}
// The enclosing C launcher records hot and cold and invokes the final callback synchronously.
// The actual CANN device launch captures its own per-binding workspace pointer by value.
extern "C" uint32_t kvip_test_forward_two_rows(void* input,void* binding,void* stream,
        const void* shape,Pipeline pipeline,Hot hot,void* workspace,uint64_t guard) {
    if(active || !input || !binding || !stream || !shape || !pipeline || !hot || !workspace || !guard)
        return 100000;
    Bound b{binding,stream,workspace,guard,false};active=&b;
    uint32_t rc=pipeline(input,binding,stream,shape,hot,BoundTail);
    active=nullptr;
    return rc?rc:(b.called?0:100000);
}
