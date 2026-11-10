#include <acl/acl.h>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <numeric>
#include <stdexcept>
#include "context.h"
#include "pipeline_types.h"
extern "C" uint32_t aclrtlaunch_test_real_topk(uint32_t,void*,void*,void*,void*,void*);
static void check(int v){if(v)throw std::runtime_error("ACL "+std::to_string(v));}
template<class T> void* upload(const std::vector<T>& v){void* p=nullptr;check(aclrtMalloc(&p,v.size()*sizeof(T),ACL_MEM_MALLOC_HUGE_FIRST));check(aclrtMemcpy(p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));return p;}
template<class T> void* obj(const T& x){return upload(std::vector<T>{x});}
template<class T> std::vector<T> download(void* p,size_t n){std::vector<T> v(n);check(aclrtMemcpy(v.data(),n*sizeof(T),p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
int main(int argc,char**argv){try{
 constexpr uint32_t h=131072,k=2048,r=16384;
 std::vector<uint16_t> scores(4*h,1);std::vector<uint32_t> valid(r,1),mapping(h,UINT32_MAX),lru(2048),bits(4*512),errors(4*32),locations(4*k),top(4*k,UINT32_MAX),progress(128);
 std::vector<int32_t> reserved(r,-1);std::iota(lru.begin(),lru.end(),0);
 for(uint32_t i=0;i<2048;++i)reserved[i]=i;
 for(uint32_t i=0;i<512;++i)mapping[r+i]=r+i;
 if(argc<3)throw std::runtime_error("usage: real_topk_sim classify|baseline scores.bin"); std::ifstream file(argv[2],std::ios::binary); file.read(reinterpret_cast<char*>(scores.data()),scores.size()*2); if(!file)throw std::runtime_error("cannot read exact fixture");
 check(aclInit(nullptr));check(aclrtSetDevice(0));aclrtStream stream;check(aclrtCreateStream(&stream));
 void *s=upload(scores),*v=upload(valid),*re=upload(reserved),*m=upload(mapping),*l=upload(lru),*t=upload(top),*bm=upload(bits),*er=upload(errors),*lo=upload(locations),*pr=upload(progress);
 rotating::Row row{};void* rows=obj(row);rotating::Context c{};c.progress=(uint64_t)pr;c.rows=(uint64_t)rows;c.mapping=(uint64_t)m;c.locations=(uint64_t)lo;c.lruSlots=(uint64_t)l;c.reservedHot=(uint64_t)re;c.hotValid=(uint64_t)v;c.shape.batch=1;c.shape.history=h;c.shape.resident=r;c.shape.indexHeads=32;void* ctx=obj(c);
 pipeline_ab::Args a{};a.context=(uint64_t)ctx;a.queryBitmaps=(uint64_t)bm;a.queryErrors=(uint64_t)er;a.history=h;a.resident=r;void* ws=obj(a);
 check(aclrtlaunch_test_real_topk(4,stream,s,t,ctx,std::string(argv[1])=="classify"?ws:nullptr));check(aclrtSynchronizeStream(stream));top=download<uint32_t>(t,4*k);
 size_t bad=0,wrong=0;for(uint32_t q=0;q<4;++q){auto sorted=std::vector<uint16_t>(scores.begin()+q*h,scores.begin()+(q+1)*h);std::nth_element(sorted.begin(),sorted.end()-k,sorted.end());uint16_t cutoff=sorted[h-k];std::vector<uint8_t> seen(h);for(uint32_t i=0;i<k;++i){auto id=top[q*k+i];if(id>=h){++bad;continue;}if(seen[id]++||scores[q*h+id]<cutoff)++wrong;}}
 if(std::string(argv[1])=="classify" && !bad){
  locations=download<uint32_t>(lo,4*k);bits=download<uint32_t>(bm,4*512);errors=download<uint32_t>(er,4*32);
  for(uint32_t q=0;q<4;++q){std::vector<uint32_t> expected(512);
   for(uint32_t i=0;i<k;++i){uint32_t id=top[q*k+i],slot=(id<r && reserved[id]<0)?id:UINT32_MAX;
    if(locations[q*k+i]!=slot)++wrong;if(slot<r)expected[slot/32]|=1U<<(slot%32);}
   for(uint32_t i=0;i<512;++i)if(bits[q*512+i]!=expected[i])++wrong;
   if(errors[q*32])++wrong;
  }
 }
 std::printf("REAL_TOPK classify=%d invalid=%zu wrong=%zu first=%u\n",std::string(argv[1])=="classify",bad,wrong,top[0]);
 if(bad||wrong)return 1;
 for(void* p:{s,v,re,m,l,t,bm,er,lo,pr,rows,ctx,ws})check(aclrtFree(p));check(aclrtDestroyStream(stream));check(aclrtResetDevice(0));check(aclFinalize());return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
