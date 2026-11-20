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
#include <vector>

#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define LAUNCH_BOUND(n)
#define __gm__
#define __ubuf__
constexpr unsigned CHUNK=512,kIndexEdgesOffset=32,kPrefetch=2048,kMaxFinal=8192,
                   kIndexTeamMinHistory=10496,kFlagCount=16;
struct BatchFlag {std::atomic<uint32_t> ready{0};uint32_t count{},error{};};
struct Args {
    uintptr_t indexTeam,denseShadow,times,flags,context,updateBindings,keys,scales,addresses,logical;
    uint32_t history,resident,prefetchCount;
};
struct Context {uintptr_t rowOwners{},hotValid;};
struct {unsigned x=0;} threadIdx;
template<class T> T asc_ldcg(T* p){return *p;}
template<class T> void asc_stcg(T* p,T value){*p=value;}
void asc_syncthreads(){}
void asc_threadfence(){}
uint32_t asc_atomic_cas(uint32_t* p,uint32_t expected,uint32_t desired){
    std::atomic_ref<uint32_t>(*p).compare_exchange_strong(expected,desired);
    return expected;
}
uint64_t GetSystemCycle(){static std::atomic<uint64_t> clock{1};return clock++;}
namespace kvaf_device {void CleanCacheLines(uint8_t*,size_t){}}
uint32_t AwaitPublication(BatchFlag* flag){
    // Inputs are always published before entry; a losing Get must never wait for the winner.
    assert(flag->ready.load());
    return flag->error;
}
enum class TPosition {VECCALC};
enum class HardEvent {V_S};
constexpr int EVENT_ID0=0;
template<HardEvent> void SetFlag(int){}
template<HardEvent> void WaitFlag(int){}
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
void SimulatedCopy(uint32_t*,uint32_t,uint32_t field,uint64_t*,uint32_t*,uint64_t*,uint64_t*,uint32_t*,uint64_t* saved,uint32_t* claim,uint32_t block) {
    ++copies[field];
    if(field==3){
        enteredFourth=true;
        while(pauseFourth.load())std::this_thread::yield();
    }
    if(claim)*reinterpret_cast<uint32_t*>(saved)=asc_atomic_cas(claim,0,block+1)==0;
}
namespace Simt {
struct Dim3 {unsigned x,y,z;};
template<auto Function,class... A> void VF_CALL(Dim3,A... args){
    if constexpr(sizeof...(A)==3)Function(args...); // Actual device claim.
    else SimulatedCopy(args...);
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
    std::cout<<"INDEX_TEAM_CLAIM_CPU_PASS rounds=40 forced_owners=4 simultaneous_claims=4 failed_plan_and_snapshot=1 reuse=1\n";
}
