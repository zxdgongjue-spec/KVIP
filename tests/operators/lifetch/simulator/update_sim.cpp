// SPDX-License-Identifier: Apache-2.0
// Actual planner + updater, no LI, network, Torch or fixture work in kernel timing.
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" uint32_t kvop_direct_update(uint32_t, void*, void*);
extern "C" uint32_t kvop_direct_update_outputs(uint32_t, void*, void*);
extern "C" uint32_t kvop_direct_update_plan_words();
extern "C" uint32_t kvop_direct_update_abi_version();
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
    unsigned n=argc>1?std::stoul(argv[1]):410, batch=argc>2?std::stoul(argv[2]):1;
    // "outputs" also runs the service adapter that maps TopK rows to logical IDs
    // and slots, which is the path production captures; the default stays owned.
    const bool outputs=argc>3 && std::string(argv[3])=="outputs";
    constexpr unsigned history=131072,hot=16384;
    bool extended=kvop_direct_update_abi_version()>=3;
    unsigned capacity=extended?10240:2048;
    if(n>capacity || !batch || batch>32)throw std::runtime_error("Invalid fixture");
    check(aclInit(nullptr));check(aclrtSetDevice(0));aclrtStream stream;check(aclrtCreateStream(&stream));
    std::vector<uint64_t> keys(uint64_t(batch)*history*16),addresses(uint64_t(batch)*history),ids(addresses.size());
    std::vector<uint32_t> scales(addresses.size()),lru(batch*capacity),changes(lru.size(),UINT32_MAX);
    std::iota(keys.begin(),keys.end(),100);std::iota(addresses.begin(),addresses.end(),1000000);
    std::iota(ids.begin(),ids.end(),2000000);std::iota(scales.begin(),scales.end(),3000000);
    for(unsigned b=0;b<batch;++b)for(unsigned i=0;i<capacity;++i){lru[b*capacity+i]=i;if(i<n)changes[b*capacity+i]=hot+i;}
    std::vector<uint64_t> a(28);std::vector<void*> allocations;
    auto bind=[&](unsigned index,auto const& data){void* p=upload(data);allocations.push_back(p);a[index]=(uint64_t)p;};
    bind(0,keys);bind(1,scales);bind(2,addresses);bind(3,ids);bind(4,lru);bind(5,changes);
    bind(6,std::vector<uint32_t>(batch));a[7]=history;bind(8,std::vector<uint32_t>(batch*256));
    if(outputs) {
      bind(9,[](){std::vector<uint32_t> owners(32);for(unsigned i=0;i<32;++i)owners[i]=i;return owners;}());
      bind(16,std::vector<uint32_t>(batch*hot,1));
    } else bind(9,std::vector<uint32_t>(batch*hot,1));
    bind(10,std::vector<uint32_t>(batch*kvop_direct_update_plan_words()));
    a[24]=history;a[25]=hot;
    if(extended){bind(26,std::vector<uint32_t>(batch,n<=2048?2048:n));bind(27,std::vector<uint32_t>(batch*40960));}
    std::vector<int32_t> topk,slots(std::size_t(batch)*4*2048);
    std::vector<int64_t> unpack(std::size_t(batch)*4),valid(std::size_t(batch)*4,1),logical_out,slot_out;
    const uint64_t rows=uint64_t(batch)*4;
    if(outputs) {
      topk.resize(rows*2048);logical_out.assign(rows*2048,-7);slot_out.assign(rows*2048,-7);
      for(uint64_t r=0;r<rows;++r) {unpack[r]=int64_t(r);for(uint32_t k=0;k<2048;++k){
        topk[r*2048+k]=int32_t((uint64_t(k)*7+r*13)%history);slots[r*2048+k]=int32_t(100000+r*2048+k);}}
      bind(18,topk);bind(19,slots);bind(20,unpack);bind(21,valid);bind(22,logical_out);bind(23,slot_out);
      // The service form also gates on completed transport, so the two queue
      // state arrays must carry the KVAF magic with a clear error word.
      constexpr uint32_t kStatusBytes=1024;
      constexpr uint64_t kMagic=0x4B5641464247444FULL;
      std::vector<uint64_t> states(std::size_t(batch)*2*(kStatusBytes/8),0);
      for(uint64_t s=0;s<std::size_t(batch)*2;++s) states[s*(kStatusBytes/8)]=kMagic;
      bind(11,std::vector<uint32_t>(std::size_t(batch)*2*32,0));
      bind(12,states);
      bind(13,states);
      a[14]=2;a[15]=kStatusBytes; // scalar ABI fields, not pointers
      // The service form writes its plan to slot 17 and reads per-row status
      // from slot 10, the reverse of the owned test path.
      bind(10,std::vector<uint32_t>(std::size_t(batch)*32,0));
      bind(17,std::vector<uint32_t>(std::size_t(batch)*kvop_direct_update_plan_words()));
    }
    void* bindings=upload(a);allocations.push_back(bindings);
    if(outputs) check(kvop_direct_update_outputs(batch,bindings,stream)); else check(kvop_direct_update(batch,bindings,stream));
    check(aclrtSynchronizeStream(stream));
    auto status=download<uint32_t>((void*)a[6],batch);for(auto s:status)if(s)throw std::runtime_error("Planner error");
    auto out=download<uint64_t>((void*)a[0],keys.size());
    for(unsigned b=0;b<batch;++b)for(unsigned t=0;t<history;++t){unsigned src=t<n?hot+t:(t>=hot && t<hot+n?t-hot:t);
      for(unsigned f=0;f<16;++f)if(out[(uint64_t(b)*history+t)*16+f]!=keys[(uint64_t(b)*history+src)*16+f])throw std::runtime_error("Key mismatch");}
    if(outputs) {
      auto lo=download<int32_t>((void*)a[22],logical_out.size()),so=download<int32_t>((void*)a[23],slot_out.size());
      for(uint64_t r=0;r<rows;++r) {const uint64_t owner=(r/4);for(uint32_t k=0;k<2048;++k){
        int32_t token=topk[r*2048+k];
        int32_t want=token>=0?int32_t(ids[owner*history+uint64_t(token)]):-1;
        if(lo[r*2048+k]!=want)throw std::runtime_error("Logical output mismatch");
        if(so[r*2048+k]!=slots[r*2048+k])throw std::runtime_error("Slot output mismatch");}}
      std::printf("outputs adapter: %llu rows x 2048 mapped\n",(unsigned long long)rows);
    }
    auto ai=download<uint64_t>((void*)a[2],addresses.size()),li=download<uint64_t>((void*)a[3],ids.size());
    auto sc=download<uint32_t>((void*)a[1],scales.size());
    for(unsigned b=0;b<batch;++b)for(unsigned t=0;t<history;++t){unsigned src=t<n?hot+t:(t>=hot && t<hot+n?t-hot:t);
      size_t d=uint64_t(b)*history+t,s=uint64_t(b)*history+src;
      if(ai[d]!=addresses[s] || li[d]!=ids[s] || sc[d]!=scales[s])throw std::runtime_error("Metadata mismatch");}
    std::printf("PASS update simulator: batch=%u changes=%u history=%u hot=%u mode=%s; all key/scale/address/id rows checked\n",batch,n,history,hot,outputs?"outputs":"owned");
    for(auto p:allocations)check(aclrtFree(p));check(aclrtDestroyStream(stream));check(aclrtResetDevice(0));check(aclFinalize());
  } catch(const std::exception& e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
