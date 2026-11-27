// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include "aclrtlaunch_gate_camodel.h"
#include "batch_get_abi.h"
#include <array>
#include <vector>
#include <cstdio>
#include <stdexcept>
#include <string>

static void require(bool pass, const char* what) { if (!pass) throw std::runtime_error(what); }
static void ck(aclError rc) { if (rc) throw std::runtime_error("ACL error " + std::to_string(rc)); }
struct Device {
    std::vector<void*> blocks;
    uint64_t alloc(size_t n, const void* seed=nullptr) {
        void* p; ck(aclrtMalloc(&p,n,ACL_MEM_MALLOC_HUGE_FIRST)); blocks.push_back(p);
        if (seed) ck(aclrtMemcpy(p,n,seed,n,ACL_MEMCPY_HOST_TO_DEVICE)); else ck(aclrtMemset(p,n,0,n));
        return reinterpret_cast<uint64_t>(p);
    }
    template<class T> uint64_t copy(const T& x) { return alloc(sizeof(T),&x); }
    template<class T> T read(uint64_t addr) {
        T t; ck(aclrtMemcpy(&t,sizeof(T),(void*)addr,sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST)); return t;
    }
    ~Device() { for(auto p:blocks) aclrtFree(p); }
};
static void run(aclrtStream stream,uint32_t enabled,unsigned fault,bool protocolOnly=false) {
    constexpr unsigned n=12, cursor=7, head=93;
    Device m; std::array<KvafExecuteBatchGetKernelArgs,n> args{};
    auto data=m.alloc(n*64*8);
    for(unsigned i=0;i<n;++i) {
        KvafBatchGetQueueState state{};state.sq_producer_index=head;state.cq_consumer_index=cursor;
        KvafBatchGetDirectUrmaEntity ent{};ent.sq_depth=4096;ent.cq_depth=8;ent.cqe_size=ent.wqe_size=64;
        ent.sq_db_va=m.alloc(128);ent.cq_db_va=m.alloc(128);
        std::array<uint32_t,8*16> cq{};
        for(unsigned k=0;k<8;++k) {
            auto at=cursor+k;
            cq[at%8*16]=((1U^((at/8)&1U))<<2) | (((fault==1 && i==3 && k==1) || (fault==2 && i==2 && k==0)) ? (1U<<16) : 0U);
        }
        ent.cq_va=m.copy(cq);
        args[i].queue_state=m.copy(state);args[i].core_status=m.copy(KvafBatchGetCoreStatus{});
        args[i].channel_entity=m.copy(ent);args[i].indices=m.alloc(128);
    }
    auto argbuf=m.copy(args);
    constexpr unsigned width=32+4096;
    std::vector<uint32_t> plans(n*width);
    const uint32_t count[4]={4,0,6,3},pred[4]={3,4,0,4};
    for(unsigned i=0;i<n;++i){
        auto p=plans.data()+i*width;
        unsigned q=i%4; p[0]=count[q];p[1]=i==9?1:pred[q];
        for(unsigned j=0;j<count[q];++j){p[32+j]=500+10*q+count[q]-1-j;p[32+2048+j]=500+10*q+j;}
    }
    auto table=m.alloc(plans.size()*4,plans.data());
    std::array<uint32_t,32+1024> shared{};shared[0]=127;
    auto scratch=m.copy(shared);
    auto permits=m.alloc(3*2*32*sizeof(uint32_t));
    const unsigned blocks=protocolOnly?n:4U;
    for (unsigned offset=0; offset<n; offset+=blocks) {
        std::printf("START enabled=%u fault=%u offset=%u\n",enabled,unsigned(fault),offset);
        ck(ACLRT_LAUNCH_KERNEL(gate_camodel)(blocks,stream,(void*)argbuf,(void*)data,enabled|(protocolOnly?2U:0U),offset,(void*)table,(void*)scratch,(void*)permits));
        ck(aclrtSynchronizeStream(stream));
    }
    for(unsigned i=0;i<n;++i) {
        auto result=m.read<std::array<uint64_t,64>>(data+i*64*8);
        if (!protocolOnly) {
            const auto observed=reinterpret_cast<const uint32_t*>(result.data()+48);
            const auto query=i%4;
            const uint32_t prior=i/4*13+127, expected[4]={prior+3,0,prior+7,prior};
            std::printf("Reservation queue=%u base=%u expected=%u source_guard=%u first=%u\n",i,observed[0],expected[query],observed[8],observed[1]);
            require(observed[0]==expected[query],"wrong archived Q reservation predecessor");
            for(unsigned j=0;j<count[query];++j){
                unsigned want=i==8?(j==0?777:500+j):i==6?(j==1?520:520+j):500+query*10+count[query]-1-j;
                require(observed[j+1]==want,"ordered final list or bad-member fallback");
            }
            require(observed[8]==(i==9?5U:(i==8||i==6)?2U:1U),"guard or bounded fallback");
        }
        auto bells=m.read<std::array<uint64_t,16>>(args[i].indices);
        auto state=m.read<KvafBatchGetQueueState>(args[i].queue_state);
        auto status=m.read<KvafBatchGetCoreStatus>(args[i].core_status);
        const unsigned q=i%4,own=i==0?0:q==1?2:31+q;
        std::vector<unsigned> ends;
        if((i<4||i>=8)&&own>1) for(unsigned cut: {1U,6U,11U}) {
            auto end=(own*cut+15)/16;
            if(end<own && (ends.empty()||ends.back()!=end)) ends.push_back(end);
        }
        ends.push_back(own);
        const bool suppressed=enabled && fault==2 && i==3;
        const bool first_error=fault==2 && i==2;
        const bool later_error=fault==1 && i==3;
        const bool bad=suppressed||first_error||later_error;
        const unsigned issued=suppressed?0U:first_error?1U:later_error?2U:own?ends.size():0U;
        const bool hold=enabled && (i<4 || i>=8) && ((i/4^q)&1U) && own;
        require(result[3]==ends.size() && bool(result[4])==hold, "first receipt policy");
        require(bool(result[2])==bad && bells[0]==issued,"transport error or extra doorbell");
        unsigned last=0;
        for(unsigned j=0;j<issued;++j) {
            require(bells[2+j*2]==head+ends[j],"private prefix changed");
            last=ends[j];
        }
        if(hold && !suppressed && (i^1U)!=0){
            auto peer=m.read<std::array<uint64_t,64>>(data+(i^1U)*64*8);
            require(peer[29]>0 && result[27]>=peer[29],"issued before paired receipt");
        }
        require(state.sq_producer_index==head+last && state.cq_consumer_index==cursor+issued, "private SQ/CQ cursor");
        require(status.processed_request_count==last && status.completed_batch_count==issued-unsigned(first_error||later_error) && bool(status.result)==bad,"private status");
    }
    std::printf("PASS enabled=%u CQ_error=%u queues=12 zero_short_long=covered\n",enabled,unsigned(fault));
}
int main(int argc,char** argv) { try {
    ck(aclInit(nullptr)); ck(aclrtSetDevice(0)); aclrtStream stream; ck(aclrtCreateStream(&stream));
    const bool protocolOnly=argc>1 && std::string(argv[1])=="--receipt-only";
    if (argc>1) require(protocolOnly && argc<=4,"usage: probe [--receipt-only [fault [enabled]]]");
    if (argc>=3) require(std::stoul(argv[2])<=2,"fault must be 0, 1, or 2");
    if (argc>=4) require(std::stoul(argv[3])<=1,"enabled must be 0 or 1");
    for(unsigned fault: {0U,1U,2U}) for(unsigned enabled: {0U,1U}) {
        if (argc>=3 && std::stoul(argv[2])!=fault) continue;
        if (argc>=4 && std::stoul(argv[3])!=enabled) continue;
        run(stream,enabled,fault,protocolOnly);
    }
    ck(aclrtDestroyStream(stream));ck(aclrtResetDevice(0));ck(aclFinalize());return 0;
} catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;} }
