// SPDX-License-Identifier: Apache-2.0
// Identical BS16/Q4 fixture for standalone baseline and staged resolver kernels.
#include <acl/acl.h>
#include "context.h"
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#if STAGED_RESOLVER
extern "C" uint32_t aclrtlaunch_test_staged_resolver(uint32_t,void*,void*);
#define RESOLVER_LAUNCH aclrtlaunch_test_staged_resolver
#else
extern "C" uint32_t aclrtlaunch_test_hot_resolver(uint32_t,void*,void*);
#define RESOLVER_LAUNCH aclrtlaunch_test_hot_resolver
#endif
static void check(int code){if(code)throw std::runtime_error("ACL/launch error "+std::to_string(code));}
template<class T> static void* upload(const std::vector<T>& input){void* p=nullptr;
 check(aclrtMalloc(&p,input.size()*sizeof(T),ACL_MEM_MALLOC_HUGE_FIRST));
 check(aclrtMemcpy(p,input.size()*sizeof(T),input.data(),input.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));return p;}
template<class T> static std::vector<T> download(void* p,size_t n){std::vector<T> out(n);
 check(aclrtMemcpy(out.data(),n*sizeof(T),p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return out;}
int main(int argc,char** argv){try{
 const bool cross=argc>1 && std::string(argv[1])=="cross-query";
 if(argc>1 && !cross)throw std::runtime_error("expected optional cross-query scenario");
 constexpr unsigned batch=16,h=131072,r=16384,k=8192,cap=10240,used=512,final=346;
 check(aclInit(nullptr));check(aclrtSetDevice(0));aclrtStream stream;check(aclrtCreateStream(&stream));
 rotating::Context ctx{};ctx.shape.batch=batch;ctx.shape.history=h;ctx.shape.resident=r;
 ctx.shape.finalWorkers=1;ctx.shape.tokenBytes=656;ctx.outputStride=r;
 std::vector<void*> allocations;auto bind=[&](uint64_t& field,auto const& data){
  void* p=upload(data);allocations.push_back(p);field=(uint64_t)p;};
 std::vector<uint32_t> selected(batch*k),mapping(batch*h,UINT32_MAX),lru(batch*2048),slots(batch*cap),changes(batch*cap,UINT32_MAX);
 std::vector<int32_t> reserved(batch*r,-1);std::vector<uint64_t> addresses(h);
 std::iota(addresses.begin(),addresses.end(),0x100000);
 for(unsigned b=0;b<batch;++b){
  for(unsigned i=0;i<2048;++i){lru[b*2048+i]=slots[b*cap+i]=i;reserved[b*r+i]=i;}
  for(unsigned i=0;i<used;++i){
   // 64 selected prefetches and 448 speculative prefetches not in TopK.
   uint32_t token=r+(i<64?i:10000+i);
   mapping[b*h+token]=r+i;changes[b*cap+i]=token;
  }
  for(unsigned q=0;q<4;++q){
   for(unsigned i=0;i<205;++i)selected[b*k+q*2048+i]=(cross && q==1 && i==0)?r+100:r+(q%2)*205+i;
   for(unsigned i=205;i<2048;++i)selected[b*k+q*2048+i]=4096+i-205;
  }
 }
 bind(ctx.args,std::vector<uint32_t>(batch*k*16,0xdeadbeef));bind(ctx.selected,selected);bind(ctx.mapping,mapping);
 bind(ctx.lruSlots,lru);bind(ctx.reservedHot,reserved);bind(ctx.changeSlots,slots);bind(ctx.changes,changes);
 bind(ctx.changeCount,std::vector<uint32_t>(batch,2048));bind(ctx.hotValid,std::vector<uint32_t>(batch*r,1));
 bind(ctx.locations,std::vector<uint32_t>(batch*k,UINT32_MAX));bind(ctx.sourceBlocks,std::vector<uint32_t>(batch*k));
 bind(ctx.sourceOffsets,std::vector<uint32_t>(batch*k));bind(ctx.destinations,std::vector<uint32_t>(batch*k));
 bind(ctx.count,std::vector<uint32_t>(1,0));bind(ctx.fallbackAddresses,std::vector<uint64_t>(batch*k));
 bind(ctx.rowStatus,std::vector<uint32_t>(batch*32));uint64_t addr=0;bind(addr,addresses);
 std::vector<rotating::Row> rows(batch);for(auto& row:rows){row.slots=used;row.addresses=addr;}
 bind(ctx.rows,rows);void* deviceContext=upload(std::vector<rotating::Context>{ctx});allocations.push_back(deviceContext);
 check(RESOLVER_LAUNCH(batch,stream,deviceContext));check(aclrtSynchronizeStream(stream));
 auto status=download<uint32_t>((void*)ctx.rowStatus,batch*32);
 auto out=download<uint32_t>((void*)ctx.locations,batch*k),change=download<uint32_t>((void*)ctx.changes,batch*cap);
 auto dest=download<uint32_t>((void*)ctx.changeSlots,batch*cap),descriptors=download<uint32_t>((void*)ctx.destinations,batch*k);
 auto remote=download<uint64_t>((void*)ctx.fallbackAddresses,batch*k);
 for(unsigned b=0;b<batch;++b){
  if(status[b*32] || status[b*32+1]!=final)throw std::runtime_error("status/count batch="+std::to_string(b));
#if STAGED_RESOLVER
  const unsigned expected[4]={141,cross?204U:205U,0,cross?1U:0U};
  unsigned prefix=0;
  for(unsigned q=0;q<4;++q){prefix+=expected[q];
   if(status[b*32+4+q]!=expected[q] || status[b*32+8+q]!=prefix)
    throw std::runtime_error("phase count/published prefix mismatch");}
#endif
  std::vector<uint32_t> ownership(r);std::iota(ownership.begin(),ownership.end(),0);
  std::set<uint32_t> written,sources;
  for(unsigned i=0;i<cap;++i){uint32_t token=change[b*cap+i],slot=dest[b*cap+i];
   if(token==UINT32_MAX)continue;
   if(slot>=r || !written.insert(slot).second || !sources.insert(token).second)throw std::runtime_error("invalid change");
   ownership[slot]=token;
  }
  for(unsigned i=0;i<k;++i){uint32_t slot=out[b*k+i];
   if(slot>=r || ownership[slot]!=selected[b*k+i])throw std::runtime_error("incorrect slot batch="+std::to_string(b)+" pos="+std::to_string(i));}
  for(unsigned i=0;i<final;++i){uint32_t dst=descriptors[b*k+i];
   if(dst<b*r || dst>=(b+1)*r || remote[b*k+i]!=addresses[ownership[dst-b*r]])throw std::runtime_error("descriptor mismatch");}
 }
 std::printf("PASS resolver BS=%u workers=1 Q4 unique_miss=410 prefetch=512 selected_prefetch=64 final=%u; source addresses, slots, dedup, hot protection verified\n",batch,final);
#if STAGED_RESOLVER
 if(cross)std::printf("STAGED phase_unique_repairs_per_request=[141,204,0,1]; cumulative=[141,345,345,346]\n");
 else std::printf("STAGED phase_unique_repairs_per_request=[141,205,0,0]; cumulative=[141,346,346,346]\n");
#endif
 for(auto p:allocations)check(aclrtFree(p));check(aclrtDestroyStream(stream));check(aclrtResetDevice(0));check(aclFinalize());
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
