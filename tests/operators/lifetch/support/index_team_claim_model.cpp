// SPDX-License-Identifier: Apache-2.0
// Execute the production row scheduler with a CPU CAS and observable field transfers.
// The companion index_team_model.cpp tests the real gather/scatter body.
#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define LAUNCH_BOUND(n)
#define __gm__
#define __ubuf__
constexpr unsigned CHUNK=512,kIndexEdgesOffset=32,kPrefetch=2048,kMaxFinal=8192,
                   kIndexTeamMinHistory=10496,kFlagCount=16,kBatchFlagPayloadBytes=12;
struct BatchFlag {std::atomic<uint32_t> ready{0};uint32_t count{},error{};std::array<std::atomic<uint32_t>,29> reserved{};};
struct Args {
    uintptr_t indexTeam,denseShadow,times,flags,context,updateBindings,keys,scales,addresses,logical;
    uint32_t history,resident,prefetchCount;
};
struct Context {uintptr_t rowOwners{},hotValid;};
thread_local struct {unsigned x=0;} threadIdx;
thread_local std::barrier<>* waveBarrier=nullptr;
template<class T> T asc_ldcg(T* p){return *p;}
uint32_t asc_ldcg(uint32_t* p){return std::atomic_ref<uint32_t>(*p).load();}
template<class T> void asc_stcg(T* p,T value){*p=value;}
void asc_syncthreads(){if(waveBarrier)waveBarrier->arrive_and_wait();}
void asc_threadfence(){std::atomic_thread_fence(std::memory_order_seq_cst);}
uint32_t asc_atomic_cas(uint32_t* p,uint32_t expected,uint32_t desired){
    std::atomic_ref<uint32_t>(*p).compare_exchange_strong(expected,desired);
    return expected;
}
uint64_t GetSystemCycle(){static std::atomic<uint64_t> clock{1};return clock++;}
namespace kvaf_device {void CleanCacheLines(uint8_t*,size_t){}}
uint32_t AwaitPublication(BatchFlag* flag){
    // Inputs are always published before entry; a losing Get must never wait for the winner.
    auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(!flag->ready.load()){assert(std::chrono::steady_clock::now()<end);std::this_thread::yield();}
    return flag->error;
}
enum class TPosition {VECCALC};
enum class HardEvent {V_S};
constexpr int EVENT_ID0=0;
template<HardEvent> void SetFlag(int){}
template<HardEvent> void WaitFlag(int){}
template<class T> struct GlobalTensor {
    T* p;
    void SetGlobalBuffer(T* data,size_t){p=data;}
    T GetValue(size_t i){return std::atomic_ref<T>(p[i]).load();}
    void SetValue(size_t i,T value){std::atomic_ref<T>(p[i]).store(value);}
};
template<class T> struct LocalTensor {
    T* p;
    T* GetPhyAddr(){return p;}
    T GetValue(size_t i){return p[i];}
};
template<TPosition> struct TBuf {
    std::vector<uint64_t> data;
    template<class T> LocalTensor<T> Get(){return {reinterpret_cast<T*>(data.data())};}
};
struct TPipe {
    template<TPosition P> void InitBuffer(TBuf<P>& b,size_t size){b.data.resize(size/8);}
    void Destroy(){}
};
std::array<std::atomic<unsigned>,4> copies{};
std::atomic<bool> pauseFourth,enteredFourth;
namespace index_team { void TryFourthBody(uint32_t*,uint32_t,uint32_t*); }
void SimulatedCopy(uint32_t*,uint32_t,uint32_t field,uint64_t*,uint32_t*,uint64_t*,uint64_t*,uint32_t*,uint64_t* saved,uint32_t* claim,uint32_t block) {
    ++copies[field];
    if(field==3){
        enteredFourth=true;
        while(pauseFourth.load())std::this_thread::yield();
    }
    if(claim)index_team::TryFourthBody(claim,block,reinterpret_cast<uint32_t*>(saved));
}
std::atomic<unsigned> stagesDone{},pauseAfterStage{},batchSuffixRuns{};
namespace Simt {
struct Dim3 {unsigned x,y,z;};
template<auto Function,class... A> void VF_CALL(Dim3,A... args){
    if constexpr(sizeof...(A)==3)Function(args...); // Actual device claim.
    else if constexpr(sizeof...(A)==7 || sizeof...(A)==6){
        if constexpr(sizeof...(A)==7){
            auto bounds=std::tie(args...);
            if(std::get<3>(bounds)-std::get<2>(bounds)>1)++batchSuffixRuns;
        }
        // Run the actual fourth-field handoff with device-width barriers.
        std::barrier wave(256);
        std::vector<std::thread> threads;
        for(unsigned t=0;t<256;++t)threads.emplace_back([&,t]{
            threadIdx.x=t;waveBarrier=&wave;Function(args...);
        });
        for(auto& thread:threads)thread.join();
        if constexpr(sizeof...(A)==6){
            const auto ordinal=++stagesDone;
            while(pauseAfterStage==ordinal)std::this_thread::yield();
        }
    }else SimulatedCopy(args...);
}
}
#include "index_team.h"

int main(){
    constexpr uint32_t batch=16,row=7,history=kIndexTeamMinHistory;
    constexpr std::array<unsigned,4> blocks{3*row,3*row+1,3*row+2,3*batch+row};
    std::array<BatchFlag,batch*5> team{};
    std::array<BatchFlag,batch*kFlagCount> flags{};
    std::vector<uint32_t> dense(batch*history*6,UINT32_MAX),valid(batch*history),scales(batch*history);
    std::vector<uint64_t> times(batch*(256+4*32+4*8)),keys(uint64_t(batch)*history*16),metadata(batch*history);
    std::array<uint64_t,8> update{};
    update[7]=history;
    Context ctx{0,uintptr_t(valid.data())};
    Args a{uintptr_t(team.data()),uintptr_t(dense.data()),uintptr_t(times.data()),uintptr_t(flags.data()),
           uintptr_t(&ctx),uintptr_t(update.data()),uintptr_t(keys.data()),uintptr_t(scales.data()),
           uintptr_t(metadata.data()),uintptr_t(metadata.data()),history,history,0x80000000U};
    auto control=index_team::Control(&a,row);
    auto plan=index_team::Moves(&a,row);
    auto trace=times.data()+batch*(256+4*32);
    auto run=[&](unsigned field){index_team::Copy(&a,blocks[field],batch);};
    // Force each potential claimant, then let four contenders race. Repeat on
    // the same plan and flags across input failures, zero work, and recovery.
    for(unsigned round=0;round<40;++round){
        unsigned schedule=round%5,kind=(round/5)%4;
        unsigned planError=kind==1?4:0,snapshotError=kind==2?8:0,chunks=kind==3?0:1;
        unsigned error=planError|snapshotError;
        for(unsigned i=0;i<=4;++i)control[i].ready=0;
        for(auto& n:copies)n=0;
        plan[0]=17;
        index_team::PublishPlan(&a,row,chunks,planError);
        index_team::Publish(flags.data()+row*kFlagCount+4,0,snapshotError);
        if(schedule<4){
            if(!error && chunks && schedule<3){
                // Hold a winning helper inside field 3 and make the Get core
                // return while both the fourth publication and row join are pending.
                pauseFourth=true;enteredFourth=false;
                std::thread winner(run,schedule);
                auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
                while(!enteredFourth.load()){
                    assert(std::chrono::steady_clock::now()<deadline);
                    std::this_thread::yield();
                }
                run(3);
                assert(control[1+schedule].ready && !control[4].ready);
                pauseFourth=false;
                winner.join();
                for(unsigned field=0;field<3;++field)if(field!=schedule)run(field);
            }else{
                run(schedule);
                for(unsigned field=0;field<4;++field)if(field!=schedule)run(field);
            }
        }else{
            std::barrier start(4);
            std::array<std::thread,4> workers;
            for(unsigned field=0;field<4;++field)workers[field]=std::thread([&,field]{start.arrive_and_wait();run(field);});
            for(auto& worker:workers)worker.join();
        }
        unsigned joinedError=0;
        for(unsigned field=0;field<4;++field){
            assert(control[1+field].ready);
            joinedError|=AwaitPublication(control+1+field);
            assert(copies[field]==unsigned(!error && chunks));
            auto record=trace+blocks[field]*8;
            assert(record[0]==index_team::kTraceMagic && record[1]==row);
            assert(uint32_t(record[2])==field && record[3]==(!error && chunks?17:0));
            assert(record[4]<=record[6] && record[6]<=record[5] && record[7]==error);
            unsigned worker=record[2]>>32;
            assert(field==3?worker+1==plan[index_team::kClaimOffset]:worker==blocks[field]);
            if(field==3 && schedule<4)assert(worker==blocks[schedule]);
        }
        assert(joinedError==error);
    }
    // The first nonempty chunk is a 1024-position cycle. The second reads
    // from positions changed by the first. Repeating or omitting either
    // chunk corrupts the reference keys even if the move count matches.
    const unsigned target=0, comm=3*batch;
    auto edges=index_team::Moves(&a,target);
    auto ctrl=index_team::Control(&a,target);
    auto snapshot=flags.data()+target*kFlagCount+4;
    for(auto& f:team)f.ready=0;
    a.prefetchCount|=2U<<16;
    const unsigned width=1024,tailCount=37;
    std::array<std::array<uint64_t,4>,width> initial{};
    for(unsigned i=0;i<width;++i)for(unsigned w=0;w<4;++w)initial[i][w]=uint64_t(i)*4+w+1;
    for(unsigned mode=0;mode<5;++mode){
        const bool oneChunk=mode==3;
        for(unsigned i=0;i<width;++i)for(unsigned w=0;w<4;++w)keys[16*i+12+w]=initial[i][w];
        auto expected=initial;
        edges[0]=0;edges[1]=width;edges[2]=oneChunk?width:width+tailCount;
        auto rotate=[&](unsigned offset,unsigned count){
            auto prior=expected;
            for(unsigned i=0;i<count;++i){
                unsigned dst=(i+1)%count;
                edges[kIndexEdgesOffset+2*(offset+i)]=i;
                edges[kIndexEdgesOffset+2*(offset+i)+1]=dst;
                expected[dst]=prior[i];
            }
        };
        rotate(0,width);
        if(!oneChunk)rotate(width,tailCount);
        for(unsigned f=0;f<5;++f)ctrl[f].ready=0;
        snapshot->ready=0;
        index_team::TailTicket ticket{}, other{};
        assert(!index_team::TryBeforeCq(&a,comm,batch,ticket));
        index_team::PublishPlan(&a,target,3,0);
        assert(!index_team::TryBeforeCq(&a,comm,batch,ticket));
        index_team::Publish(snapshot,0,1);
        assert(!index_team::TryBeforeCq(&a,comm,batch,ticket));
        index_team::Publish(snapshot,0,0);
        assert(index_team::TryBeforeCq(&a,comm,batch,ticket));
        assert(ticket.row==target && ticket.firstMoves==width && ticket.moves==edges[2]);
        assert(!index_team::TryBeforeCq(&a,comm,batch,other));
        if(oneChunk){
            assert(ctrl[4].ready);
        }else{
            assert(!ctrl[4].ready && ticket.nextChunk==2);
            if(mode==0)assert(index_team::FinishAfterCq(&a,comm,batch,ticket));
            else if(mode!=4)index_team::Copy(&a,mode==1?3*target:comm+target,batch);
            else {
                // Row 15's first core has published its own result. It can
                // finish row 0 while the original ticket holder is in CQ.
                index_team::HelpAfterLocalWork(&a,45,batch);
                assert(edges[index_team::kClaimOffset]==46);
            }
        }
        assert(!index_team::FinishAfterCq(&a,comm,batch,ticket));
        assert(ctrl[4].ready && ctrl[4].error==0);
        for(unsigned i=0;i<width;++i)for(unsigned w=0;w<4;++w)assert(keys[16*i+12+w]==expected[i][w]);
    }
    // The standby may read a sealed chunk while the row plan or Q4 is
    // pending. A later chunk reads a position overwritten by the first.
    const unsigned standby=56,home=8,peer=0;
    auto deadline=[](const auto& check) {
        auto end=std::chrono::steady_clock::now()+std::chrono::seconds(15);
        while(!check()){assert(std::chrono::steady_clock::now()<end);std::this_thread::yield();}
    };
    for(unsigned mode=0;mode<5;++mode){
        auto target=mode==4?home:peer;
        for(auto& f:team){f.ready=0;f.reserved[0]=0;}
        for(auto& f:flags)f.ready=0;
        for(auto r:{home,peer}){
            auto e=index_team::Moves(&a,r);
            e[0]=2;e[1]=3;
            const std::array<unsigned,6> srcDst{0,1,1,0,1,2};
            std::copy(srcDst.begin(),srcDst.end(),e+kIndexEdgesOffset);
            for(unsigned k=0;k<3;++k)for(unsigned w=0;w<16;++w)
                keys[(uint64_t(r)*history+k)*16+w]=1000*r+100*k+w;
        }
        auto localKeys=[&](unsigned r,unsigned k,unsigned w){
            return keys[(uint64_t(r)*history+k)*16+w];
        };
        bool terminal=false;
        auto field=index_team::Control(&a,target);
        stagesDone=0;
        index_team::PublishPrefix(&a,target);
        std::thread helper([&]{terminal=index_team::CopySealedPrefix(&a,standby,batch);});
        deadline([&]{return stagesDone.load()==1;});
        assert(!field[4].ready && localKeys(target,0,12)==1000*target+12);
        // Pre-read made no GM writes. Competing post-CQ and primary helpers
        // cannot double-claim the peer. If home becomes ready first, it can
        // finish while its peer is still waiting on the Q4 snapshots.
        if(mode==0){
            index_team::PublishPlan(&a,home,2,0);
            index_team::Publish(flags.data()+home*kFlagCount+4,0,0);
            auto homeClaim=index_team::Moves(&a,home)+index_team::kClaimOffset;
            deadline([&]{return index_team::ReadClaim(homeClaim)==(index_team::kFourthSuffixReady|1U);});
            assert(localKeys(home,0,12)==1000*home+112 && !field[4].ready);
            index_team::HelpReadyFourth(&a,57,batch,index_team::ProbeFourth(&a,home,false));
            assert(index_team::Control(&a,home)[4].ready && localKeys(home,2,12)==1000*home+12);
        }
        index_team::PublishPlan(&a,target,2,mode==2?4:0,true);
        assert(index_team::ReadClaim(index_team::Moves(&a,target)+index_team::kClaimOffset)==standby+1);
        index_team::TailTicket race{};
        if(mode!=4)assert(!index_team::TryBeforeCq(&a,48,batch,race));
        index_team::Publish(flags.data()+target*kFlagCount+4,0,mode==3?8:0);
        helper.join();
        assert(terminal==(target==home));
        assert(field[4].ready && field[4].error==(mode==2?4:mode==3?8:0));
        for(unsigned k=0;k<3;++k)for(unsigned w=0;w<16;++w){
            unsigned src=(w<12 || mode==2 || mode==3)?k:k==0?1:0;
            assert(localKeys(target,k,w)==1000*target+100*src+w);
        }
        auto timeslot=times.data()+batch*256+32*standby;
        assert(timeslot[16]<=timeslot[17] && timeslot[17]<=timeslot[18]);
        assert(timeslot[19]==2 && timeslot[20]==target+1);
    }
    // Hold both gates closed while the same AIV owns two sealed prefixes.
    // Open one gate between slices and assert its order, then allow an error
    // in the partner. All source words in a chunk form one long permutation.
    for(unsigned scenario=0;scenario<4;++scenario){
        constexpr unsigned n=777;
        auto key=[&](unsigned r,unsigned i,unsigned w)->uint64_t&{
            return keys[(uint64_t(r)*history+i)*16+w];
        };
        for(auto& f:team){f.ready=0;f.reserved[0]=0;}
        for(auto& f:flags)f.ready=0;
        for(auto r:{home,peer}){
            auto e=index_team::Moves(&a,r);
            e[0]=n;e[1]=n+1;e[2]=n+2;
            for(unsigned i=0;i<n+2;++i){
                e[kIndexEdgesOffset+2*i]=i<n?i:(i==n?1:n);
                e[kIndexEdgesOffset+2*i+1]=i<n?(i+1)%n:i;
                for(unsigned w=0;w<16;++w)key(r,i,w)=100000*r+32*i+w;
            }
            index_team::PublishPrefix(&a,r);
        }
        auto publish=[&](unsigned r,unsigned planError,unsigned qError){
            index_team::PublishPlan(&a,r,3,planError,true);
            index_team::Publish(flags.data()+r*kFlagCount+4,0,qError);
        };
        stagesDone=0;pauseAfterStage=2;batchSuffixRuns=0;
        bool complete=false;
        std::thread helper([&]{complete=index_team::CopySealedPrefix(&a,standby,batch);});
        deadline([&]{return stagesDone.load()==2;}); // peer entirely read; home has <=256 moves
        for(auto r:{home,peer}){
            assert(index_team::ReadClaim(index_team::Moves(&a,r)+index_team::kClaimOffset)==standby+1);
            assert(key(r,1,12)==100000*r+32+12);
        }
        bool homeFirst=scenario==1 || scenario==3;
        unsigned first=homeFirst?home:peer, second=homeFirst?peer:home;
        publish(first,0,0);pauseAfterStage=0;
        deadline([&]{return index_team::Control(&a,first)[4].ready.load();});
        assert(!index_team::Control(&a,second)[4].ready && key(second,1,12)==100000*second+32+12);
        publish(second,scenario==2?4:0,scenario==3?8:0);
        helper.join();
        assert(complete);
        for(auto r:{home,peer}){
            unsigned err=r==second && scenario>=2?(scenario==2?4:8):0;
            assert(index_team::Control(&a,r)[4].ready && index_team::Control(&a,r)[4].error==err);
            for(unsigned i=0;i<n+2;++i)for(unsigned w=0;w<16;++w){
                unsigned src=w<12 || err?i:(i>=n?0:(i+n-1)%n);
                assert(key(r,i,w)==100000*r+32*src+w);
            }
        }
        const auto t=times.data()+batch*256+32*standby;
        assert(t[20]==peer+1 && t[25]==home+1 && t[19]==n && t[24]==n);
        assert(t[26]==256 && t[27]>0 && t[29]>=t[27]);
        assert(t[28]>=1 && t[28]<=4 && (scenario!=1 || t[28]==4));
        if(scenario<2)assert(batchSuffixRuns>=1);
        if(scenario==0)assert(t[18]<t[22]); // serve the first row before gathering all of the second
    }
    std::cout<<"TWO_SEALED_ROWS_PASS ready_order=both max_secondary_gather=256 "
                "ordered_suffix=1 failed_plan_or_Q4_no_writes=1 exactly_once=1\n";
    std::cout<<"SEALED_PREFIX_PASS pending_plan_and_Q4=1 failed_plan_and_Q4=1 "
                "dependent_suffix=1 paired_home_progress=1 no_early_writes=1 fields_exact=1\n";
    std::cout<<"INDEX_CQ_PREFIX_PASS full_chunk=1024 suffix_owner=comm,local_index,nominal_fourth,remote_index exactly_once=true\n";
    auto homeControl=index_team::Control(&a,home);
    auto homeEdges=index_team::Moves(&a,home);
    auto homeSnapshot=flags.data()+home*kFlagCount+4;
    // The standby is paired with row 0. Test early peer arrival, relative
    // workloads, a missing Q4 snapshot and the error/empty home fallback.
    for(unsigned schedule=0;schedule<6;++schedule){
        for(auto& flag:team)flag.ready=0;
        snapshot->ready=0;homeSnapshot->ready=0;
        edges[0]=2;
        edges[kIndexEdgesOffset]=0;edges[kIndexEdgesOffset+1]=1;
        edges[kIndexEdgesOffset+2]=1;edges[kIndexEdgesOffset+3]=0;
        for(unsigned i=0;i<2;++i)for(unsigned w=0;w<4;++w)keys[16*i+12+w]=initial[i][w];
        index_team::PublishPlan(&a,target,1,0);
        if(schedule!=3)index_team::Publish(snapshot,0,0);
        homeEdges[0]=schedule==1?1:2;
        unsigned homeChunks=schedule==5?0:1,homeError=schedule==4?4:0;
        index_team::PublishPlan(&a,home,homeChunks,homeError);
        if(schedule!=0)index_team::Publish(homeSnapshot,0,0);
        index_team::HelpBeforeHome(&a,standby,batch);
        bool peerFirst=schedule==0 || schedule==1 || homeError || !homeChunks;
        assert(bool(ctrl[4].ready)==peerFirst);
        assert(!homeControl[4].ready);
        if(peerFirst){
            assert(edges[index_team::kClaimOffset]==standby+1);
            for(unsigned i=0;i<2;++i)for(unsigned w=0;w<4;++w)
                assert(keys[16*i+12+w]==initial[1-i][w]);
        }
        index_team::Publish(homeSnapshot,0,0);
        index_team::Copy(&a,standby,batch);
        assert(homeControl[4].ready && homeControl[4].error==homeError);
    }
    std::cout<<"INDEX_STANDBY_PAIR_PASS earlier_ready=1 larger_ready=1 ties_home=1 unpublished_q4=1 error_empty_fallback=1\n";
    // All sixteen rows must be reachable by exactly one prefix queue under
    // each width, including the fourth request group owned by a single lane.
    for(unsigned w:{1U,2U,4U}){
        a.prefetchCount=0x80000000U|(w<<16);
        for(unsigned r=0;r<batch;++r){
            for(auto& flag:team)flag.ready=0;
            auto e=index_team::Moves(&a,r);
            e[0]=1;e[kIndexEdgesOffset]=0;e[kIndexEdgesOffset+1]=0;
            index_team::PublishPlan(&a,r,1,0);
            index_team::Publish(flags.data()+r*kFlagCount+4,0,0);
            const unsigned expected=r/4*w+r%w;
            index_team::TailTicket t{};
            for(unsigned queue=0;queue<4*w;++queue)if(queue!=expected)
                assert(!index_team::TryBeforeCq(&a,comm+queue,batch,t));
            assert(index_team::TryBeforeCq(&a,comm+expected,batch,t));
            assert(t.row==r && index_team::Control(&a,r)[4].ready);
        }
    }
    std::cout<<"INDEX_COMM_WIDTH_PASS workers_per_q=1,2,4 prefix_partition=exact rows=16 each=true\n";
    std::cout<<"INDEX_TEAM_CLAIM_CPU_PASS rounds=40 forced_owners=4 simultaneous_claims=4 failed_plan_and_snapshot=1 reuse=1\n";
}
