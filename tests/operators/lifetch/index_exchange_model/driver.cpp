// SPDX-License-Identifier: Apache-2.0
#include "args.h"
#include <acl/acl.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>
#include <stdexcept>
extern "C" uint32_t aclrtlaunch_v9_parallel_terminal(uint32_t,void*,void*);
void ck(int v){if(v)throw std::runtime_error("ACL "+std::to_string(v));}
template<class T> uint64_t put(const std::vector<T>& v){void*p;ck(aclrtMalloc(&p,v.size()*sizeof(T),ACL_MEM_MALLOC_HUGE_FIRST));ck(aclrtMemcpy(p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));return(uint64_t)p;}
template<class T> void restore(uint64_t p,const std::vector<T>&v){ck(aclrtMemcpy((void*)p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));}
template<class T> std::vector<T> get(uint64_t p,size_t n){std::vector<T> v(n);ck(aclrtMemcpy(v.data(),n*sizeof(T),(void*)p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
int main(int argc,char**argv){setvbuf(stdout,nullptr,_IONBF,0);try{
 uint32_t count=argc>1?std::stoul(argv[1]):230,cycle=argc>2?std::stoul(argv[2]):0;
 constexpr uint32_t R=16384,H=131328;ck(aclInit(nullptr));ck(aclrtSetDevice(0));aclrtStream stream;ck(aclrtCreateStream(&stream));
 std::vector<uint32_t> src(count),dst(count),inv(R,UINT32_MAX),scale(H),valid(R,1),flags(32*32);
 for(uint32_t i=0;i<count;++i){src[i]=R+i;dst[i]=i;inv[i]=i;}
 // A cross-worker hot cycle plus many cold exchanges exercises the full fallback.
 uint32_t hot=cycle?std::min(count,cycle):0;
 for(uint32_t i=0;i<hot;++i)src[i]=(i+1)%hot;
 std::vector<uint64_t> keys(H*16),addr(H),logical(H);
 for(uint32_t i=0;i<H;++i){scale[i]=i+13;addr[i]=0x100000000ULL+656ULL*i;logical[i]=i*3+7;for(uint32_t j=0;j<16;++j)keys[16*i+j]=16ULL*i+j;}
 TestArgs a{};a.src=put(src);a.dst=put(dst);a.keys=put(keys);a.scales=put(scale);a.addresses=put(addr);a.logical=put(logical);a.valid=put(valid);a.flags=put(flags);a.inverse=put(inv);a.snapshot=put(std::vector<uint64_t>(2*10240*19));a.status=put(std::vector<uint32_t>(4));a.count=count;a.resident=R;a.history=H;
 auto device=put(std::vector<TestArgs>{a});
 auto finish=[&](int code){ck(aclrtDestroyStream(stream));ck(aclrtResetDevice(0));ck(aclFinalize());return code;};
 std::vector<uint32_t> expected(H);std::iota(expected.begin(),expected.end(),0);
 for(uint32_t i=0;i<count;++i)expected[dst[i]]=src[i];
 for(uint32_t i=hot;i<count;++i)expected[src[i]]=dst[i];
 for(uint32_t round=1;round<=2;++round){
 restore(a.keys,keys);restore(a.scales,scale);restore(a.addresses,addr);restore(a.logical,logical);a.generation=round;restore(device,std::vector<TestArgs>{a});
 printf("launch round=%u count=%u\n",round,count);ck(aclrtlaunch_v9_parallel_terminal(4,stream,(void*)device));ck(aclrtSynchronizeStream(stream));printf("sync complete\n");auto status=get<uint32_t>(a.status,4);
 printf("count=%u hot=%u round=%u status=%u,%u,%u,%u\n",count,hot,round,status[0],status[1],status[2],status[3]);
 if(std::any_of(status.begin(),status.end(),[](uint32_t x){return x!=0;}))return finish(2);
 auto k=get<uint64_t>(a.keys,H*16),ad=get<uint64_t>(a.addresses,H),lo=get<uint64_t>(a.logical,H);auto sc=get<uint32_t>(a.scales,H);size_t bad=0;
 for(uint32_t i=0;i<H;++i){auto e=expected[i];bad+=sc[i]!=scale[e]||ad[i]!=addr[e]||lo[i]!=logical[e];for(uint32_t j=0;j<16;++j)bad+=k[16*i+j]!=keys[16*e+j];}
 printf("audit_bad=%zu\n",bad);if(bad)return finish(3);
 }
 return finish(0);
}catch(const std::exception&e){printf("ERROR %s\n",e.what());return 1;}}
