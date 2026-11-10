// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" uint32_t aclrtlaunch_test_topk_classify(uint32_t,void*,void*,void*,void*,void*,void*,void*,void*,uint32_t);
static void check(int v){if(v)throw std::runtime_error("ACL error "+std::to_string(v));}
template<class T> void* upload(const std::vector<T>& v){void* p=nullptr;check(aclrtMalloc(&p,v.size()*sizeof(T),ACL_MEM_MALLOC_HUGE_FIRST));check(aclrtMemcpy(p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));return p;}
template<class T> std::vector<T> download(void* p,size_t n){std::vector<T> v(n);check(aclrtMemcpy(v.data(),n*sizeof(T),p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
int main(int argc,char**argv){try{
 bool classify=argc>1 && std::string(argv[1])=="classify";constexpr uint32_t h=131072,k=2048,r=16384;
 std::vector<uint16_t> scores(4*h,1);std::vector<uint32_t> valid(r,1),mapping(h,UINT32_MAX),lru(2048),result(4*k,UINT32_MAX),top(4*k,UINT32_MAX);
 std::vector<int32_t> reserved(r,-1);std::iota(lru.begin(),lru.end(),0);
 for(uint32_t i=0;i<2048;++i)reserved[i]=i;
 reserved[4096]=0;valid[10000]=0;
 for(uint32_t i=0;i<64;++i)mapping[r+i]=r+i;
 for(uint32_t q=0;q<4;++q)for(uint32_t i=0;i<k;++i){uint32_t token=i<205?r+(q%2)*205+i:((i%2)?4096U:10000U)+(i-205)/2;scores[q*h+token]=uint16_t(30000+i);}
 check(aclInit(nullptr));check(aclrtSetDevice(0));aclrtStream stream;check(aclrtCreateStream(&stream));
 auto s=upload(scores),v=upload(valid),re=upload(reserved),m=upload(mapping),l=upload(lru),t=upload(top),o=upload(result);
 check(aclrtlaunch_test_topk_classify(4,stream,s,v,re,m,l,t,o,uint32_t(classify)));check(aclrtSynchronizeStream(stream));
 top=download<uint32_t>(t,4*k);if(classify)result=download<uint32_t>(o,4*k);
 for(uint32_t q=0;q<4;++q){std::vector<uint8_t> seen(h);for(uint32_t i=0;i<k;++i){uint32_t token=top[q*k+i];
  if(token>=h||seen[token]++||scores[q*h+token]<=1)throw std::runtime_error("incorrect TopK");
  uint32_t expected=token<r&&valid[token]&&reserved[token]<0?token:mapping[token];if(expected>=r&&expected<r+2048)expected=lru[expected-r];
  if(classify&&result[q*k+i]!=expected)throw std::runtime_error("classification mismatch");}}
 std::printf("PASS topk Q4 4 AIV history=131072 k=2048 classify=%u\n",classify);
 for(void* p:{s,v,re,m,l,t,o})check(aclrtFree(p));check(aclrtDestroyStream(stream));check(aclrtResetDevice(0));check(aclFinalize());
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
