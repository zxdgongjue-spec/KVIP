// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include "context.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" uint32_t aclrtlaunch_test_hot_resolver(uint32_t,void*,void*);
static void check(int code) { if(code) throw std::runtime_error("ACL/launch error " + std::to_string(code)); }
template<class T> static void* upload(const std::vector<T>& input) {
  void* p=nullptr;check(aclrtMalloc(&p,input.size()*sizeof(T),ACL_MEM_MALLOC_HUGE_FIRST));
  check(aclrtMemcpy(p,input.size()*sizeof(T),input.data(),input.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));return p;
}
template<class T> static std::vector<T> download(void* p,size_t n) {
  std::vector<T> output(n);check(aclrtMemcpy(output.data(),n*sizeof(T),p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return output;
}
int main(int argc,char** argv) {
 try {
  const unsigned h=131072,r=16384,capacity=10240;
  const unsigned workers=argc>2?std::stoul(argv[2]):4;
  if(workers!=1 && workers!=2 && workers!=4)throw std::runtime_error("workers must be 1, 2 or 4");
  unsigned misses=argc>1?std::stoul(argv[1]):3000;
  if(misses<512 || misses>8192)throw std::runtime_error("misses must be 512..8192");
  check(aclInit(nullptr));check(aclrtSetDevice(0));aclrtStream stream;check(aclrtCreateStream(&stream));
  rotating::Context ctx{};ctx.shape.history=h;ctx.shape.resident=r;ctx.shape.finalWorkers=workers;ctx.shape.tokenBytes=656;
  std::vector<void*> allocations;
  auto bind=[&](uint64_t& field,auto const& data){void* p=upload(data);allocations.push_back(p);field=(uint64_t)p;};
  std::vector<uint32_t> selected(8192),mapping(h,UINT32_MAX),candidates(2048),slots(capacity),changes(capacity,UINT32_MAX);
  std::vector<int32_t> reserved(r,-1);
  std::vector<uint64_t> addresses(h);std::iota(addresses.begin(),addresses.end(),0x100000);
  for(unsigned i=0;i<8192;++i)selected[i]=i<misses?r+i:r/2+i-misses;
  for(unsigned i=0;i<2048;++i){candidates[i]=slots[i]=i;reserved[i]=i;}
  for(unsigned i=0;i<512;++i){unsigned token=r+(i<256?i:10000+i);mapping[token]=r+i;changes[i]=token;}
  bind(ctx.selected,selected);bind(ctx.mapping,mapping);bind(ctx.lruSlots,candidates);bind(ctx.reservedHot,reserved);
  bind(ctx.changeSlots,slots);bind(ctx.changes,changes);bind(ctx.changeCount,std::vector<uint32_t>{2048});
  bind(ctx.hotValid,std::vector<uint32_t>(r,1));bind(ctx.locations,std::vector<uint32_t>(8192,UINT32_MAX));
  bind(ctx.sourceBlocks,std::vector<uint32_t>(8192));bind(ctx.sourceOffsets,std::vector<uint32_t>(8192));
  bind(ctx.destinations,std::vector<uint32_t>(8192));bind(ctx.count,std::vector<uint32_t>{0});
  bind(ctx.fallbackAddresses,std::vector<uint64_t>(8192));bind(ctx.rowStatus,std::vector<uint32_t>(workers*32));
  uint64_t addr=0;bind(addr,addresses);rotating::Row row{};row.slots=512;row.addresses=addr;
  bind(ctx.rows,std::vector<rotating::Row>{row});void* deviceContext=upload(std::vector<rotating::Context>{ctx});allocations.push_back(deviceContext);
  check(aclrtlaunch_test_hot_resolver(workers,stream,deviceContext));check(aclrtSynchronizeStream(stream));
  auto status=download<uint32_t>((void*)ctx.rowStatus,workers*32);
  for(unsigned w=0;w<workers;++w)if(status[w*32] || status[w*32+1]!=misses-256 || !status[w*32+2])throw std::runtime_error("Resolver status/count/deferred mismatch");
  auto out=download<uint32_t>((void*)ctx.locations,8192),change=download<uint32_t>((void*)ctx.changes,capacity),dest=download<uint32_t>((void*)ctx.changeSlots,capacity);
  std::vector<uint32_t> ownership(r);std::iota(ownership.begin(),ownership.end(),0);std::set<uint32_t> written,sources;
  for(unsigned i=0;i<capacity;++i)if(change[i]!=UINT32_MAX){if(dest[i]>=r || !written.insert(dest[i]).second || !sources.insert(change[i]).second)throw std::runtime_error("Invalid change records");ownership[dest[i]]=change[i];}
  for(unsigned i=0;i<8192;++i)if(out[i]>=r || ownership[out[i]]!=selected[i])throw std::runtime_error("Final TopK hot ownership mismatch at "+std::to_string(i));
  auto descriptors=download<uint32_t>((void*)ctx.destinations,misses-256);auto remote=download<uint64_t>((void*)ctx.fallbackAddresses,misses-256);
  for(unsigned i=0;i<misses-256;++i)if(descriptors[i]>=r || remote[i]!=addresses[ownership[descriptors[i]]])throw std::runtime_error("Descriptor mismatch");
  auto count=download<uint32_t>((void*)ctx.changeCount,1)[0];
  if(count>capacity || count<2048)throw std::runtime_error("Record bound error");
  std::printf("PASS production resolver: workers=%u final_union=8192 misses=%u prefetch=512 final_reads=%u records=%zu bound=%u; all final KV use persistent hot; descriptors/records/protection checked\n",workers,misses,misses-256,written.size(),count);
  for(auto p:allocations)check(aclrtFree(p));check(aclrtDestroyStream(stream));check(aclrtResetDevice(0));check(aclFinalize());
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
