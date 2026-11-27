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
static void run(aclrtStream stream,uint32_t enabled,bool fault) {
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
            cq[at%8*16]=((1U^((at/8)&1U))<<2) | (fault && i==3 && k==1 ? (1U<<16) : 0U);
        }
        ent.cq_va=m.copy(cq);
        args[i].queue_state=m.copy(state);args[i].core_status=m.copy(KvafBatchGetCoreStatus{});
        args[i].channel_entity=m.copy(ent);args[i].indices=m.alloc(128);
    }
    auto argbuf=m.copy(args);
    for (unsigned offset: {0U, 4U, 8U}) {
        std::printf("START enabled=%u fault=%u offset=%u\n",enabled,unsigned(fault),offset);
        ck(ACLRT_LAUNCH_KERNEL(gate_camodel)(4,stream,(void*)argbuf,(void*)data,enabled,offset));
        ck(aclrtSynchronizeStream(stream));
    }
    for(unsigned i=0;i<n;++i) {
        auto result=m.read<std::array<uint64_t,64>>(data+i*64*8);
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
        const bool bad=fault&&i==3;
        const unsigned issued=bad?2:own?ends.size():0;
        const unsigned group=enabled&&ends.size()==4?(i/4+q)&3U:4U;
        require(result[3]==ends.size() && result[4]==group, "gate selection/count");
        require(bool(result[2])==bad && bells[0]==issued,"transport error or extra doorbell");
        unsigned last=0;
        for(unsigned j=0;j<issued;++j) {
            require(bells[2+j*2]==head+ends[j],"private prefix changed");
            if(group<4) {
                auto clk=bells[3+j*2], delta=(clk - (group<<12)) & 16383U;
                std::printf("group=%u prefix=%u issued_at=%llu quarter_offset=%llu\n",group,j,
                            (unsigned long long)clk,(unsigned long long)delta);
                require(delta < 4096+256,"doorbell outside its quarter");
            }
            last=ends[j];
        }
        require(state.sq_producer_index==head+last && state.cq_consumer_index==cursor+issued, "private SQ/CQ cursor");
        require(status.processed_request_count==last && status.completed_batch_count==issued-bad && bool(status.result)==bad,"private status");
    }
    std::printf("PASS enabled=%u CQ_error=%u queues=12 zero_short_long=covered\n",enabled,unsigned(fault));
}
int main() { try {
    ck(aclInit(nullptr)); ck(aclrtSetDevice(0)); aclrtStream stream; ck(aclrtCreateStream(&stream));
    for(bool fault: {false,true}) for(unsigned enabled: {0U,1U}) run(stream,enabled,fault);
    ck(aclrtDestroyStream(stream));ck(aclrtResetDevice(0));ck(aclFinalize());return 0;
} catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;} }
