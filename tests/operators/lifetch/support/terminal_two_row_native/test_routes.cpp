// SPDX-License-Identifier: Apache-2.0
// Host pointers stand in for GM; DirectRows and the Context/Args layouts are shared with the true AIV.
#include "resume.h"
#define KVIP_OUTPUT_WORKER 1
#include "context.h"
#include "pipeline_types.h"
#define __gm__
#define __aicore__
namespace kvaf_device { void CleanCacheLines(uint8_t*,uint64_t){} }
#include "binding_guard.h"
#include <array>
#include <cassert>
#include <iostream>
int main(){
    pipeline_ab::Args a{};rotating::Context c{};std::array<uint64_t,28> u{};
    std::array<int32_t,32> owners{};std::array<int64_t,128> src{},valid{};
    for(uint32_t i=0;i<128;++i){src[i]=i;valid[i]=1;owners[i/4]=i/4;}
    a.updateBindings=uintptr_t(u.data());c.shape.batch=32;c.shape.resident=16384;
    c.shape.history=131072;c.outputStride=16896;c.rowOwners=u[9]=uintptr_t(owners.data());
    u[7]=c.shape.history;u[20]=uintptr_t(src.data());u[21]=uintptr_t(valid.data());
    c.selected=u[22]=0x20000000;c.locations=u[23]=0x21000000;
    auto ok=[&](){return joint_tail::DirectRows(&a,&c);};
    assert(ok());owners[3]=1;assert(!ok());owners[3]=3;
    src[7]=3;assert(!ok());src[7]=7;valid[7]=0;assert(!ok());valid[7]=1;
    ++u[23];assert(!ok());u[23]=c.locations;
    u[22]=u[23];assert(!ok());u[22]=c.selected;
    --u[7];assert(!ok());++u[7];c.rowOwners+=4;assert(!ok());c.rowOwners-=4;
    u[23]=0x22000000;assert(ok());c.shape.batch=28;assert(!ok());
    std::cout<<"PASS live ownership, remapped source query, padding, partial/checkpoint output aliases, stride, batch scope\n";
}
