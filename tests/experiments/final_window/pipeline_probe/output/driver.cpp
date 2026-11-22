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
int main(){try{ck(aclInit(nullptr));ck(aclrtSetDevice(0));aclrtStream stream;ck(aclrtCreateStream(&stream));constexpr uint32_t H=131328,R=16384,K=2048;rotating::Context c{};c.shape.history=H;c.shape.resident=R;c.shape.batch=1;
std::vector<uint32_t>ids(8192),map(H,UINT32_MAX),lru(2048),want(8192),slot(8192);std::vector<uint64_t>logical(H);std::iota(lru.begin(),lru.end(),10000);for(uint32_t i=0;i<H;++i)logical[i]=i*3+7;
uint32_t cnt[4]={11,0,94,87},beg[4]={0,0,11,59};for(uint32_t i=0;i<146;++i)map[H-16384+i]=R+84+i;
for(uint32_t q=0;q<4;++q)for(uint32_t i=0;i<K;++i){uint32_t token=i;ids[q*K+i]=i|0x80000000U;slot[q*K+i]=i;if(i<cnt[q]){token=H-16384+beg[q]+i;ids[q*K+i]=token;slot[q*K+i]=lru[map[token]-R];}else if(i==100){token=H-512;ids[q*K+i]=token|0x40000000U;slot[q*K+i]=R+256;}want[q*K+i]=logical[token];}
c.selected=put(ids);c.locations=put(std::vector<uint32_t>(8192,UINT32_MAX));c.mapping=put(map);c.lruSlots=put(lru);pipeline_ab::Args a{};a.context=put(std::vector<rotating::Context>{c});a.logical=put(logical);a.eventsSrc=put(std::vector<uint32_t>(8192));a.queryErrors=put(std::vector<uint32_t>(128));ck(aclrtlaunch_test_service(1,stream,(void*)put(std::vector<pipeline_ab::Args>{a}),0));ck(aclrtSynchronizeStream(stream));auto got=get<uint32_t>(a.eventsSrc,8192),sl=get<uint32_t>(c.locations,8192),err=get<uint32_t>(a.queryErrors,128);size_t bad=0;for(uint32_t i=0;i<8192;++i)bad+=got[i]!=want[i]||sl[i]!=slot[i];for(auto v:err)bad+=v!=0;printf("OUTPUT 4Q logical+slots bad=%zu\n",bad);ck(aclrtDestroyStream(stream));ck(aclrtResetDevice(0));ck(aclFinalize());return bad?1:0;}catch(const std::exception&e){printf("ERROR %s\n",e.what());return 2;}}
