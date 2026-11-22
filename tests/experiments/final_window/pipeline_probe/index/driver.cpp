#include <acl/acl.h>
#include <vector>
#include <numeric>
#include <cstdio>
#include <stdexcept>
#include "context.h"
#include "pipeline_types.h"
extern "C" uint32_t aclrtlaunch_test_service(uint32_t,void*,void*,uint32_t);
void ck(int v){if(v)throw std::runtime_error(std::to_string(v));}
template<class T>uint64_t put(const std::vector<T>&v){void*p;ck(aclrtMalloc(&p,v.size()*sizeof(T),ACL_MEM_MALLOC_HUGE_FIRST));ck(aclrtMemcpy(p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));return(uint64_t)p;}
template<class T>std::vector<T>get(uint64_t p,size_t n){std::vector<T>v(n);ck(aclrtMemcpy(v.data(),n*sizeof(T),(void*)p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
int main(int argc,char**argv){try{uint32_t mode=argc>1?std::stoul(argv[1]):0;ck(aclInit(nullptr));ck(aclrtSetDevice(0));aclrtStream stream;ck(aclrtCreateStream(&stream));constexpr uint32_t H=131328,R=16384;
std::vector<uint64_t> keys(H*16),addr(H),logical(H);std::vector<uint32_t> scales(H),src(230),dst(230);for(uint32_t i=0;i<H;++i){addr[i]=uint64_t(i)*656;logical[i]=i*3+7;scales[i]=i;for(uint32_t j=0;j<16;++j)keys[i*16+j]=uint64_t(i)*16+j;}for(uint32_t i=0;i<230;++i){src[i]=H-20000+i;dst[i]=10000+i;}
pipeline_ab::Args a{};a.history=H;a.resident=R;a.keys=put(keys);a.scales=put(scales);a.addresses=put(addr);a.logical=put(logical);a.eventsSrc=put(src);a.eventsDst=put(dst);a.times=put(std::vector<uint64_t>(16));a.queryErrors=put(std::vector<uint32_t>(32));ck(aclrtlaunch_test_service(1,stream,(void*)put(std::vector<pipeline_ab::Args>{a}),mode));ck(aclrtSynchronizeStream(stream));auto k=get<uint64_t>(a.keys,H*16),ad=get<uint64_t>(a.addresses,H),lo=get<uint64_t>(a.logical,H);auto sc=get<uint32_t>(a.scales,H),err=get<uint32_t>(a.queryErrors,32);size_t bad=err[0];for(uint32_t i=0;i<230;++i){bad+=ad[src[i]]!=addr[dst[i]]||ad[dst[i]]!=addr[src[i]]||lo[src[i]]!=logical[dst[i]]||lo[dst[i]]!=logical[src[i]]||sc[src[i]]!=scales[dst[i]]||sc[dst[i]]!=scales[src[i]];for(uint32_t j=0;j<16;++j)bad+=k[src[i]*16+j]!=keys[dst[i]*16+j]||k[dst[i]*16+j]!=keys[src[i]*16+j];}
auto ts=get<uint64_t>(a.times,16);for(uint32_t i=0;i<12;++i)printf("stamp%u=%llu\n",i,(unsigned long long)ts[i]);printf("INDEX mode=%u bad=%zu\n",mode,bad);ck(aclrtDestroyStream(stream));ck(aclrtResetDevice(0));ck(aclFinalize());return bad?1:0;}catch(const std::exception&e){printf("ERROR %s\n",e.what());return 2;}}
