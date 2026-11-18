// SPDX-License-Identifier: Apache-2.0
// Exercise the actual RunRow control flow. Model one FIFO per Q and a separate
// prefetch queue, with explicit ownership, hazard and draining assertions.
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#define __gm__
#define __ubuf__
#define __aicore__
#define __simt_vf__
#define LAUNCH_BOUND(n)
thread_local struct { unsigned x; } threadIdx;
thread_local unsigned lane;
thread_local bool appliedThisLane=false;
uint64_t GetSystemCycle() { return std::chrono::steady_clock::now().time_since_epoch().count(); }
template<class T> T asc_ldcg(T* p) { return *p; }
void asc_syncthreads() {}
void asc_atomic_or(uint32_t* p, uint32_t n) { *p |= n; }
namespace kvaf_device { constexpr uint32_t kInvalidArgs=1; void CleanCacheLines(uint8_t*,size_t) {} }
enum class HardEvent { V_S }; enum class TPosition { VECCALC }; constexpr uint32_t EVENT_ID0=0;
template<HardEvent> void SetFlag(unsigned) {} template<HardEvent> void WaitFlag(unsigned) {}
struct View { uint32_t* p; uint32_t* GetPhyAddr(){return p;} uint32_t GetValue(unsigned i){return p[i];} };
template<TPosition> struct TBuf { uint32_t w[8]{}; template<class T> View Get(){return {w};} };
struct TPipe { template<class T> void InitBuffer(T&,unsigned){} void Destroy(){} };
namespace Simt { struct Dim3 {unsigned x,y,z;}; template<auto F,class... A> void VF_CALL(Dim3 d,A... a){
    for(unsigned i=0;i<d.x;++i){threadIdx.x=i; F(a...);} } }
constexpr uint32_t kFlagCount=16,kTop=8,kSlots=4*kTop,kRowStatusWords=32,kCapacity=4,kRowBytes=128;
struct BatchFlag { std::atomic<uint32_t> ready{},count{},error{},reserved[29]{}; };
struct Edge {uint32_t s,d;}; struct Tuple {uint32_t padding;};
struct FinalPost {uint32_t count{},nonconflicting{},descriptor{},pending{},deferred{},error{};};
struct Row {uint32_t error=0, pending=1;};
struct Context {uint64_t fallbackAddresses{},destinations{},sourceBlocks{},sourceOffsets{},fallbackValid{},rowStatus{},rows{},rowOwners{};FinalPost post[4]{};};
struct Args {uint64_t context{}, flags{},postIndex{},updateBindings{},times{},highIndex{}; uint32_t prefetchCount{};};
struct Shape {unsigned batch=16,finalPostOffset=0,resident=16;}; Shape LoadRuntimeShape(Context*){return {};}
struct Entity {uint32_t cq_depth=4;uint64_t remote_host_ddr_address=128,remote_host_ddr_size=65536,local_hbm_size=65536;};
using KvafBatchGetDirectUrmaEntity=Entity;
struct QueueArgs {uint64_t channel_entity{},core_status{},indices{},offsets{},output_indices{},address_table{},valid_table{};
    uint32_t block_count{},jettys_per_core{},table_capacity{},request_capacity{},tokens_per_block{},output_rows=1024,token_bytes=16;};
struct PrivatePlan {unsigned pairs=3,singles=7,invalidations=1,error=0;};
struct Input {std::array<unsigned,4> n,safe; unsigned cqDepth=4; int blocking=-1,badPreflight=-1, badBlock=-1,badWait=-1,badDescriptor=-1;bool badPlan=false,badDrain=false,badTotal=false,pending=true;};
struct FIFO {unsigned issued=0,drained=0,reads=0,capacity=3;std::vector<unsigned> positions;};
struct World {Input input;Entity e;FIFO q[4];std::atomic<unsigned> prefetch{0}, applied{0}, finalized{0},published{0};unsigned status=0;};
World* sim;
unsigned FinalQueue(unsigned batch,unsigned workers,unsigned b,unsigned q){return !q ? 2*b+1 : 2*batch+b*(workers-1)+q-1;}
QueueArgs GetArgs(Context*,unsigned queue){assert(queue==FinalQueue(16,4,0,lane));return {uint64_t(&sim->e)};}
FinalPost* FinalPostAt(unsigned,Context* c,unsigned base){assert(!base); return c->post;}
unsigned AwaitPublication(BatchFlag* f){while(!f->ready.load())std::this_thread::yield();return f->error;}
unsigned Producer(Args*,unsigned){assert(lane==0);return 5;}
template<bool> void OutputWorker(Args*,unsigned){assert(lane==1);}
void Plan(Args*,unsigned,Edge*,uint32_t*,bool,PrivatePlan& p){assert(lane==2);p.error=sim->input.badPlan;}
void CompletePrefetch(Context* c,unsigned,bool){assert(lane==0);assert(sim->prefetch.fetch_add(1)==0);reinterpret_cast<Row*>(c->rows)->error|=sim->input.badDrain;}
namespace post_index_tail {struct Record {uint64_t error{},globalError{},count{},fallback{},completed{},active{},start{},planDone{},packDone{},postBegin{},postDone{},indexDone{},cqDone{};};void Flush(Record*){} }
namespace post_index_transport {
struct State{unsigned seq=0,count=0,completed=0;};
uint32_t Preflight(QueueArgs&,unsigned,unsigned n,bool& blocking){if(!n)return 0;blocking=int(lane)==sim->input.blocking;return int(lane)==sim->input.badPreflight;}
uint32_t Post(QueueArgs&,unsigned first,unsigned n,State& s){auto& q=sim->q[lane];assert(!s.seq && n);
    assert(n+first<=sim->input.n[lane]);assert(first+n<=sim->input.safe[lane] || !sim->input.pending || sim->prefetch.load());
    for(unsigned i=first;i<first+n;++i)q.positions.push_back(i);
    s.seq=++q.issued;s.count=n;assert(q.issued-q.drained<sim->e.cq_depth);return 0;}
uint32_t Wait(State& s){if(!s.seq)return 0;auto& q=sim->q[lane];assert(appliedThisLane);assert(++q.drained==s.seq);
    if(int(lane)==sim->input.badWait && s.seq==1)return 1;
    s.completed=s.count;return 0;}
uint32_t RetirePair(State& a,State& b){uint32_t e=Wait(a);return e|Wait(b);}
uint32_t GetBlocking(QueueArgs& args,unsigned begin,unsigned count,State& s){if(int(lane)==sim->input.badBlock)return 1;
    Post(args,begin,count,s);auto err=Wait(s);s.seq=0;return err;}
}
template<class T> uint64_t ApplyRowIndex(Args*,Context*,unsigned,unsigned,unsigned q,unsigned lanes,unsigned pairs,
        unsigned singles,unsigned invalidations,uint64_t,uint64_t*,bool,T& team){
    assert(q==lane && lanes==4 && pairs==3 && singles==7 && invalidations==1);
    appliedThisLane=true;sim->applied.fetch_add(1);return team.Gathered();}
void PublishStreamingUpdateStatus(uint32_t* p,unsigned,unsigned err){if(!threadIdx.x)*p=err;}
void PublishStreamingResidency(uint32_t,unsigned,Context*,unsigned){if(!threadIdx.x)sim->published.fetch_add(1);}
template<bool,bool> void FinalizeStreaming(Args*,unsigned,unsigned useful){assert(useful==5 && lane==0);sim->finalized.fetch_add(1);}
#include "high_index_async.h"

void run(Input in){World world{in};sim=&world;world.e.cq_depth=in.cqDepth;
    std::array<BatchFlag,16> flags;std::array<post_index_tail::Record,64> records{};std::array<Edge,128> edges{};
    std::array<uint64_t,64*32+16*256> ticks{};uint64_t update[16]{};update[6]=uint64_t(&world.status);
    uint32_t info[32]{};Row row;row.pending=in.pending; uint64_t addresses[kSlots]{};int dst[kSlots]{},blk[kSlots]{},off[kSlots]{};uint8_t valid[kSlots]{};
    Context c{uint64_t(addresses),uint64_t(dst),uint64_t(blk),uint64_t(off),uint64_t(valid),uint64_t(info),uint64_t(&row)};
    Args a{uint64_t(&c),uint64_t(flags.data()),uint64_t(records.data()),uint64_t(update),uint64_t(ticks.data()),uint64_t(edges.data()),0x90000000};
    for(unsigned i=0;i<kSlots;++i){addresses[i]=128+16*i;blk[i]=i;dst[i]=i;valid[i]=1;}
    for(unsigned q=0;q<4;++q){c.post[q]={in.n[q],in.safe[q],q*kTop};info[1]+=in.n[q];flags[1+q].ready=1;}
    if(in.badTotal)++info[1];
    if(in.badDescriptor>=0)valid[in.badDescriptor*kTop]=0;
    std::array<std::thread,4> cores;
    for(unsigned q=0;q<4;++q)cores[q]=std::thread([&,q]{lane=q;RunRow(&a,q*16);});
    for(auto& core:cores)core.join();
    bool failed=in.badPlan||in.badDrain||in.badTotal||in.badBlock>=0||in.badWait>=0||in.badDescriptor>=0||in.badPreflight>=0;
    assert(bool(world.status)==failed && world.finalized==1 && world.published==!failed && world.prefetch==1);
    unsigned totalCompleted=0;
    for(unsigned q=0;q<4;++q){auto& queue=world.q[q];
        assert(queue.issued==queue.drained);
        if(in.badPlan || in.badDescriptor>=0 || in.badPreflight>=0 || in.badTotal)assert(!queue.issued);
        if(in.badDrain && in.safe[q]<in.n[q])assert(queue.positions.size()<=in.safe[q]);
        if(!failed){assert(queue.positions.size()==in.n[q]);for(unsigned i=0;i<in.n[q];++i)assert(queue.positions[i]==i);
            assert(queue.issued==(in.n[q]==0 ? 0u : (in.pending && in.safe[q] && in.safe[q]<in.n[q] && in.cqDepth>2 && int(q)!=in.blocking ? 2u:1u)));}
        totalCompleted+=records[q*16].completed;assert(bool(c.post[q].error)==failed);}
    assert(c.post[0].pending==totalCompleted);
    if(!failed){
        assert(world.applied==4);
        for(unsigned q=0;q<4;++q){
            auto& rec=records[q*16];
            auto* mark=ticks.data()+16*256+q*16*32;
            assert(rec.indexDone>=mark[3] && rec.cqDone>=rec.indexDone);
            assert(mark[8]==world.q[q].issued && bool(rec.postBegin)==bool(in.n[q]));
            if(mark[8])assert(rec.postBegin<=mark[4] && mark[4]<=rec.postDone);
            if(mark[8]==2)assert(rec.postBegin<mark[3] && mark[4]>=rec.indexDone);
        }
    }
    if(in.badPlan||in.badDescriptor>=0||in.badPreflight>=0||in.badTotal)assert(world.applied==0);
    if(in.badBlock>=0 || in.badDrain || in.badWait>=0)assert(world.applied==4);
    if(in.badDrain){assert(world.q[0].issued==1 && world.q[0].positions.size()==in.safe[0]);}
}
int main(){Input mixed{{4,3,0,5},{2,3,0,0}};
    run(mixed);auto drained=mixed;drained.pending=false;run(drained);auto c=mixed;c.cqDepth=2;run(c);c=mixed;c.blocking=0;run(c);c.badBlock=0;run(c);
    c=mixed;c.badPlan=true;run(c);c=mixed;c.badTotal=true;run(c);c=mixed;c.badDescriptor=1;run(c);
    c=mixed;c.badPreflight=3;run(c);c=mixed;c.badWait=0;run(c);c=mixed;c.badDrain=true;run(c);
    run(Input{{4,3,0,5},{4,3,0,5}});run(Input{{0,0,0,0},{0,0,0,0}});
}
