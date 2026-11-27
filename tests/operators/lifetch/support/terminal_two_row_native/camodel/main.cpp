// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include "aclrtlaunch_two_row_camodel.h"
#include "batch_get_abi.h"
#include "context.h"
#include "pipeline_types.h"
#include "terminal_runtime_abi.h"
#include "fixture.h"
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
void ck(aclError rc){if(rc)throw std::runtime_error("ACL "+std::to_string(rc));}
void req(bool v,const std::string&msg){if(!v)throw std::runtime_error(msg);}
struct Mem{
    std::vector<void*> blocks;
    uint64_t put(const void* src,size_t n){
        void* p;ck(aclrtMalloc(&p,n,ACL_MEM_MALLOC_HUGE_FIRST));blocks.push_back(p);
        if(src)ck(aclrtMemcpy(p,n,src,n,ACL_MEMCPY_HOST_TO_DEVICE));else ck(aclrtMemset(p,n,0,n));
        return reinterpret_cast<uint64_t>(p);
    }
    template<class T>uint64_t put(const T&t){return put(&t,sizeof(t));}
    template<class T>T read(uint64_t addr){T v;ck(aclrtMemcpy(&v,sizeof(v),(void*)addr,sizeof(v),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
    ~Mem(){for(auto p:blocks)aclrtFree(p);}
};
struct QueueRef{uint64_t state,status,entity,sqdb,cqdb;};
static void run(int fault,aclrtStream stream){
    Mem m;
    constexpr uint32_t rows=32,qs=4;
    auto trace=m.put(nullptr,rows*qs*sizeof(tiny::Trace)),ws=m.put(nullptr,(1+32*26)*128);
    auto sched=m.put(terminal_runtime::TerminalRowSchedule{});
    auto gates=ws+(1+tiny::first*26+16)*128;
    std::array<KvafExecuteBatchGetKernelArgs,128> finalArgs{};
    std::array<KvafExecuteBatchGetKernelArgs,64> preArgs{};
    std::array<rotating::Row,32> rowstate{};
    std::array<std::array<QueueRef,5>,2> refs{};
    const unsigned counts[][4]={{29,51,24,198},{4,6,8,299}};
    const unsigned cuts[][4][4]={{{2,11,20,29},{4,20,36,51},{2,9,17,24},{13,75,137,198}},
                                {{1,2,3,4},{1,3,5,6},{1,3,6,8},{19,113,206,299}}};
    for(unsigned which=0;which<2;++which){unsigned row=which?tiny::second:tiny::first;
        for(unsigned j=0;j<=4;++j){
            auto& args=j<4?finalArgs[row*4+j]:preArgs[row*2];
            KvafBatchGetQueueState queue{};queue.sq_producer_index=tiny::head;queue.cq_consumer_index=tiny::cursor;
            KvafBatchGetDirectUrmaEntity e{};e.sq_depth=4096;e.cq_depth=8;e.cqe_size=e.wqe_size=64;
            e.sq_db_va=m.put(nullptr,128);e.cq_db_va=m.put(nullptr,128);
            std::array<uint32_t,8*16> cq{};
            for(unsigned k=0;k<8;++k){auto pos=tiny::cursor+k;
                bool ready=(j==4 && which==0 && k<2) || (fault==3 && which==0 && j==0 && !k);
                cq[pos%8*16]=uint32_t((pos/8&1)^ready)<<2;}
            e.cq_va=m.put(cq);
            args.queue_state=m.put(queue);args.channel_entity=m.put(e);args.core_status=m.put(KvafBatchGetCoreStatus{});
            args.output_rows=33*16896;args.table_capacity=fault;args.request_capacity=row;
            args.indices=trace+(row*4+j)*sizeof(tiny::Trace);args.offsets=gates;
            if(j<4)args.token_bytes=counts[which][j];
            refs[which][j]={args.queue_state,args.core_status,args.channel_entity,e.sq_db_va,e.cq_db_va};
        }
        rowstate[row].pending=2;rowstate[row].requests=2;
    }
    rotating::Context c{};c.shape.batch=32;c.shape.history=131072;c.shape.resident=16384;c.outputStride=16896;
    c.selected=m.put(nullptr,128*2048*4);c.locations=m.put(nullptr,128*2048*4);
    auto posts=m.put(nullptr,rows*qs*sizeof(rotating::FinalPost));c.owners=posts;
    preArgs[0].output_rows=33*16896;
    c.rows=m.put(rowstate);c.args=m.put(preArgs);
    std::array<uint32_t,32> owners{};std::array<int64_t,128> src{},active{};
    for(unsigned i=0;i<128;++i){owners[i/4]=1+i/4;src[i]=i;active[i]=1;}
    c.rowOwners=m.put(owners);
    std::array<uint64_t,28> update{};update[7]=c.shape.history;update[9]=c.rowOwners;update[20]=m.put(src);
    update[21]=m.put(active);update[22]=c.selected;update[23]=c.locations;
    pipeline_ab::Args pipe{};pipe.context=m.put(c);pipe.updateBindings=m.put(update);
    terminal_runtime::Binding binding{};binding.queueArgs=m.put(finalArgs);binding.reserved[1]=sched;
    binding.pipeline=m.put(pipe);binding.flags=m.put(nullptr,32*32*32*4);binding.times=m.put(nullptr,32*4*32*8);
    auto bp=m.put(binding);
    ck(ACLRT_LAUNCH_KERNEL(two_row_camodel)(4,stream,(void*)bp,(void*)ws,1));
    ck(aclrtSynchronizeStream(stream));
    auto schedule=m.read<terminal_runtime::TerminalRowSchedule>(sched);
    req(schedule.ready==0 && schedule.finishedTeams==16 && schedule.next==33,"four-core scheduler failed to retire");
    const bool parked=fault!=3;
    req(schedule.arrivals[tiny::first]==(parked?0U:4U) &&
        schedule.successor[tiny::first]==(parked?0U:tiny::second+1) &&
        schedule.arrivals[tiny::second]==4 && schedule.successor[tiny::second]==33,
        "early and completed claims were duplicated or did not use four arrivals");
    uint64_t lastReturn=0,firstOrdinary=UINT64_MAX,lastThird=0,minBell=UINT64_MAX;uint64_t tokens=0;
    for(unsigned which=0;which<2;++which){unsigned row=which?tiny::second:tiny::first;
        for(unsigned q=0;q<4;++q){auto ref=refs[which][q];auto id=row*4+q;
            auto tr=m.read<tiny::Trace>(trace+id*sizeof(tiny::Trace));
            auto post=m.read<rotating::FinalPost>(posts+id*sizeof(rotating::FinalPost));
            auto queue=m.read<KvafBatchGetQueueState>(ref.state);auto status=m.read<KvafBatchGetCoreStatus>(ref.status);
            const bool errorCase=fault==1 || fault==2;
            const bool failed=errorCase&&(which || q==3);const bool held=errorCase&&which;
            const unsigned published=held?0:4;
            const unsigned consumed=held?0:(!which&&q==3&&fault==2?2:4);
            const unsigned successful=consumed-(!which&&q==3&&fault==1);
            const unsigned requested=published?cuts[which][q][published-1]:0;
            std::printf("fault=%d r=%u q=%u prep=%llu refused=%llu parked=%llu gate=%llu issued=%llu seen=%llu success=%llu pending=%llu earlyQ2=%llu retire=%llu err=%llu\n",fault,row,q,
                (unsigned long long)tr.prepare,(unsigned long long)tr.refused,(unsigned long long)tr.atPark,(unsigned long long)tr.mask,
                (unsigned long long)tr.issued,(unsigned long long)tr.seen,(unsigned long long)tr.success,(unsigned long long)tr.pending,
                (unsigned long long)tr.earlyQ2,(unsigned long long)tr.retired,(unsigned long long)tr.error);
            req(tr.prepare==1&&tr.refused==1&&tr.retired==1&&post.head==tiny::head&&tr.initialSq==tiny::head,"preparation or base head changed");
            req(tr.issued==published&&tr.seen==consumed&&tr.success==successful&&tr.pending==published-consumed&&bool(tr.error)==failed,"retirement/poison state");
            req(post.count==requested&&post.pending==published-consumed&&queue.sq_producer_index==tiny::head+requested,"SQ protocol");
            req(queue.cq_consumer_index==tiny::cursor+consumed&&queue.completed_batch_count==successful&&bool(queue.poisoned)==failed,"CQ protocol");
            req(status.completed_batch_count==successful&&status.processed_request_count==requested&&bool(status.result)==failed,"KVAF private status");
            if((!parked || which) && q==(parked?3u:2u))
                req(tr.forcedFourth==1 && tr.expectedArrival==3,"wrong worker won normal ownership");
            if(!which && !parked)lastReturn=std::max(lastReturn,tr.returnTick);
            if(which && !parked)firstOrdinary=std::min(firstOrdinary,tr.ordinaryStart);
            if(!which){req(tr.atPark==(parked?2u:0u),"park did not hold two unseen markers");lastThird=std::max(lastThird,tr.parentThird);}
            else if(errorCase || !parked){req(!tr.firstChild&&!tr.mask,"child doorbell after fault");}
            else{req(tr.firstChild&&tr.mask==15,"child doorbell before all q reached third");minBell=std::min(minBell,tr.firstChild);}
            req(q==1?(tr.q2Start==(!which&&parked?9:7)&&tr.q2End==9&&tr.earlyQ2==(!which&&parked)):
                       (!tr.earlyQ2&&!tr.q2Start&&!tr.q2End),"prefetch consumed by wrong owner/twice");
            tokens+=post.count;
        }
        auto pq=m.read<KvafBatchGetQueueState>(refs[which][4].state);
        auto ps=m.read<KvafBatchGetCoreStatus>(refs[which][4].status);
        req(pq.cq_consumer_index==9&&pq.completed_batch_count==2&&ps.completed_batch_count==2,"prefetch CQ not fully consumed");
    }
    if(!fault)req(lastThird>0&&minBell>=lastThird&&tokens==619,"early doorbell or lost tokens");
    if(!parked)req(lastReturn>0 && firstOrdinary>=lastReturn && tokens==619,"next row began before fourth query returned");
    std::printf("NORMAL_JOIN fault=%d parent_arrivals=%u child_arrivals=%u forced_fourth_Q=%u last_parent_return=%llu next_prepare=%llu\n",
                fault,schedule.arrivals[tiny::first],schedule.arrivals[tiny::second],parked?4U:3U,
                (unsigned long long)lastReturn,(unsigned long long)(parked?0:firstOrdinary));
    std::printf("PASS fault=%d two_rows=7,29 four_AIV=4 source_lists=%llu last_third=%llu first_next=%llu\n",
                fault,(unsigned long long)tokens,(unsigned long long)lastThird,(unsigned long long)(fault?0:minBell));
}
int main(){try{ck(aclInit(nullptr));ck(aclrtSetDevice(0));aclrtStream s;ck(aclrtCreateStream(&s));
    for(int fault=0;fault<4;++fault)run(fault,s);
    ck(aclrtDestroyStream(s));ck(aclrtResetDevice(0));ck(aclFinalize());std::puts("JOINT_TWO_ROW_CAMODEL=PASS");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
