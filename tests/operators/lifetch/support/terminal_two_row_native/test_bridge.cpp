// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
using Hot=uint32_t(*)(uint32_t,void*,void*,void*);
using Tail=uint32_t(*)(uint32_t,void*,void*);
using Pipeline=uint32_t(*)(void*,void*,void*,const void*,Hot,Tail);
extern "C" uint32_t kvip_test_forward_two_rows(void*,void*,void*,const void*,Pipeline,Hot,void*,uint64_t);
namespace {
thread_local unsigned step=0;thread_local int mode=0;
thread_local void* expected[4];thread_local uint64_t ticks;
uint32_t hot(uint32_t cores,void* stream,void* input,void* binding){
    assert(cores==32 && step++==0 && input==expected[0] && binding==expected[1] && stream==expected[2]);return 0;
}
uint32_t launch(void* input,void* binding,void* stream,const void*shape,Hot h,Tail t){
    assert(shape==expected[0]);if(mode==1)return 65;
    assert(!h(32,stream,input,binding));++step; // captured cold launch
    if(mode==2)return 0;
    if(mode==3)return t(64,stream,input);
    auto rc=t(64,stream,binding);
    if(mode==4 && !rc)return t(64,stream,binding);
    return rc;
}
}
extern "C" uint32_t aclrtlaunch_v9_test_two_row_terminal(uint32_t cores,void* stream,void* binding,void* ws,uint64_t guard){
    assert(cores==64 && step++==2 && stream==expected[2] && binding==expected[1] && ws==expected[3] && ticks==guard);
    return mode==5?66:0;
}
static void replay(){
    unsigned marker[4]{};
    for(unsigned i=0;i<4;++i)expected[i]=marker+i;
    for(int i=0;i<60;++i){mode=i%6;ticks=1700+i;step=0;
        uint32_t rc=kvip_test_forward_two_rows(expected[0],expected[1],expected[2],expected[0],launch,hot,expected[3],ticks);
        assert(rc==(mode==0?0:mode==1?65:mode==5?66:100000));
        assert(step==(mode==1?0:(mode==2||mode==3)?2:3));
    }
}
int main(){std::thread a(replay),b(replay);replay();a.join();b.join();std::cout<<"PASS 180 bound forward recordings on three host threads, original hot/cold plus one tail, propagated errors and rejected stale/missing/duplicate callbacks\n";}
