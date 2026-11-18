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
namespace kvaf_device { constexpr uint32_t kInvalidArgs=1,kWqesPerPost=128; void CleanCacheLines(uint8_t*,size_t) {} }
enum class HardEvent { V_S }; enum class TPosition { VECCALC }; constexpr uint32_t EVENT_ID0=0;
template<HardEvent> void SetFlag(unsigned) {} template<HardEvent> void WaitFlag(unsigned) {}
struct View { uint32_t* p; uint32_t* GetPhyAddr(){return p;} uint32_t GetValue(unsigned i){return p[i];} };
template<TPosition> struct TBuf { uint32_t w[8]{}; template<class T> View Get(){return {w};} };
struct TPipe { template<class T> void InitBuffer(T&,unsigned){} void Destroy(){} };
namespace Simt { struct Dim3 {unsigned x,y,z;}; template<auto F,class... A> void VF_CALL(Dim3 d,A... a){
    for(unsigned i=0;i<d.x;++i){threadIdx.x=i; F(a...);} } }
constexpr uint32_t kFlagCount=16,kTop=512,kSlots=4*kTop,kRowStatusWords=32,kCapacity=4,kRowBytes=128;
struct BatchFlag { std::atomic<uint32_t> ready{},count{},error{},reserved[29]{}; };
struct Edge {uint32_t s,d;}; struct Tuple {uint32_t padding;};
struct FinalPost {uint32_t count{},nonconflicting{},descriptor{},pending{},deferred{},error{};};
struct Row {uint32_t error=0, pending=1;};
struct Context {uint64_t fallbackAddresses{},destinations{},sourceBlocks{},sourceOffsets{},fallbackValid{},rowStatus{},rows{},rowOwners{};FinalPost post[4]{};};
struct Args {uint64_t context{}, flags{},postIndex{},updateBindings{},times{},highIndex{}; uint32_t prefetchCount{};};
struct Shape {unsigned batch=16,finalPostOffset=0,resident=16;}; Shape LoadRuntimeShape(Context*){return {};}
struct Entity {uint32_t cq_depth=4,sq_depth=4096,remote_token_id=1,remote_token_value=2,local_token_id=3,local_token_value=4;uint8_t remote_eid[16]{};uint64_t local_hbm_address=64;uint64_t remote_host_ddr_address=128,remote_host_ddr_size=65536,local_hbm_size=65536;};
using KvafBatchGetDirectUrmaEntity=Entity;
struct QueueArgs {uint64_t channel_entity{},core_status{},indices{},offsets{},output_indices{},address_table{},valid_table{};
    uint64_t output=64;uint32_t batch_get_max_inflight_total=0,block_count{},jettys_per_core{},table_capacity{},request_capacity{},tokens_per_block{},output_rows=4096,token_bytes=16;};
struct PrivatePlan {unsigned pairs=3,singles=7,invalidations=1,error=0;};
struct Input {std::array<unsigned,4> n,safe; unsigned cqDepth=4; int blocking=-1,badPreflight=-1, badBlock=-1,badWait=-1,badDescriptor=-1,badPost=-1;bool badPlan=false,badDrain=false,badTotal=false,pending=true,badTnd=false; int badMr=-1, badEid=-1, localPath=-1, badRegion=-1, badLocalSize=-1, differentOutput=-1; bool expectLoan=false; unsigned maxInflight=0;};
struct FIFO {unsigned issued=0,drained=0,reads=0,capacity=3;std::vector<unsigned> positions;};
struct World {Input input;Entity e[4];uint64_t blocks=0;FIFO q[4];std::atomic<unsigned> prefetch{0}, applied{0}, finalized{0},published{0};unsigned status=0;};
World* sim;
unsigned FinalQueue(unsigned batch,unsigned workers,unsigned b,unsigned q){return !q ? 2*b+1 : 2*batch+b*(workers-1)+q-1;}
QueueArgs GetArgs(Context*,unsigned queue){for(unsigned q=0;q<4;++q)if(queue==FinalQueue(16,4,0,q)){ QueueArgs args{uint64_t(&sim->e[q])};args.output=sim->e[q].local_hbm_address;args.batch_get_max_inflight_total=sim->input.maxInflight;return args;} assert(false);return {};}
FinalPost* FinalPostAt(unsigned,Context* c,unsigned base){assert(!base); return c->post;}
unsigned AwaitPublication(BatchFlag* f){while(!f->ready.load())std::this_thread::yield();return f->error;}
template<bool PostIndex,bool RowOutput> unsigned Producer(Args*,unsigned){static_assert(!PostIndex && RowOutput);assert(lane==0);return 5;}
enum class OutputRole { Descriptors, Tnd };
template<bool,OutputRole Role> unsigned OutputWorker(Args*,unsigned){assert(lane==(Role==OutputRole::Descriptors?1:3));return Role==OutputRole::Tnd && sim->input.badTnd;}
void Plan(Args*,unsigned,Edge*,uint32_t*,bool,PrivatePlan& p){assert(lane==2);p.error=sim->input.badPlan;}
void CompletePrefetch(Context* c,unsigned,bool){assert(lane==0);assert(sim->prefetch.fetch_add(1)==0);reinterpret_cast<Row*>(c->rows)->error|=sim->input.badDrain;}
namespace post_index_tail {struct Record {uint64_t error{},globalError{},count{},fallback{},completed{},active{},start{},planDone{},packDone{},postBegin{},postDone{},indexDone{},cqDone{};};void Flush(Record*){} }
namespace post_index_transport {
struct State{unsigned seq=0,count=0,completed=0;};
uint32_t Preflight(QueueArgs&,unsigned,unsigned n,bool& blocking){if(!n)return 0;assert(sim->input.n[lane]);blocking=int(lane)==sim->input.blocking || (sim->input.maxInflight && n>sim->input.maxInflight);return int(lane)==sim->input.badPreflight;}
uint32_t Post(QueueArgs& args,unsigned first,unsigned n,State& s){auto& q=sim->q[lane];assert(!s.seq && n);
    if(int(lane)==sim->input.badPost && first==0)return 1;
    assert(n+first<=args.request_capacity);
    unsigned begin=(args.indices-sim->blocks)/4+first;
    unsigned owner=begin/kTop;
    assert(args.channel_entity==uint64_t(&sim->e[lane]) && owner<4);
    assert(begin%kTop+n<=sim->input.n[owner]);
    assert(begin%kTop+n<=sim->input.safe[owner] || !sim->input.pending || sim->prefetch.load());
    if(owner!=lane)assert(sim->e[owner].remote_token_id==sim->e[lane].remote_token_id &&
                          sim->e[owner].remote_eid[0]==sim->e[lane].remote_eid[0]);
    for(unsigned i=begin;i<begin+n;++i)q.positions.push_back(i);
    s.seq=++q.issued;s.count=n;assert(q.issued-q.drained<sim->e[lane].cq_depth);return 0;}
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
template<bool,bool,bool JoinedRow> void FinalizeStreaming(Args* a,unsigned,unsigned useful,
        unsigned total,unsigned completed,unsigned error){
    static_assert(JoinedRow);assert(useful==5 && lane==0);
    unsigned expected=0;
    auto* c=reinterpret_cast<Context*>(a->context);
    for(auto& post:c->post){expected+=post.count;post.pending=0;post.deferred=0;post.error=error;}
    assert(expected==total && (completed==total || error));
    reinterpret_cast<unsigned*>(c->rowStatus)[4]=completed;
    sim->finalized.fetch_add(1);
}
#include "high_index_async.h"

void run(Input in){World world{in};sim=&world;for(auto& e:world.e)e.cq_depth=in.cqDepth;
    if(in.badMr>=0)world.e[in.badMr].remote_token_id++;
    if(in.badEid>=0)world.e[in.badEid].remote_eid[15]++;
    if(in.localPath>=0){world.e[in.localPath].local_token_id++;world.e[in.localPath].local_token_value++;}
    if(in.badLocalSize>=0)world.e[in.badLocalSize].local_hbm_size++;
    if(in.differentOutput>=0)world.e[in.differentOutput].local_hbm_address+=65536;
    if(in.badRegion>=0)world.e[in.badRegion].remote_host_ddr_size++;
    const bool badRequiredQueue=in.badPreflight>=0 && in.n[in.badPreflight]!=0;
    std::array<BatchFlag,16> flags;std::array<post_index_tail::Record,64> records{};std::array<Edge,128> edges{};
    std::array<uint64_t,64*32+16*256> ticks{};uint64_t update[16]{};update[6]=uint64_t(&world.status);
    uint32_t info[32]{};Row row;row.pending=in.pending; uint64_t addresses[kSlots]{};int dst[kSlots]{},blk[kSlots]{},off[kSlots]{};uint8_t valid[kSlots]{};
    world.blocks=uint64_t(blk);
    Context c{uint64_t(addresses),uint64_t(dst),uint64_t(blk),uint64_t(off),uint64_t(valid),uint64_t(info),uint64_t(&row)};
    Args a{uint64_t(&c),uint64_t(flags.data()),uint64_t(records.data()),uint64_t(update),uint64_t(ticks.data()),uint64_t(edges.data()),0x90000000};
    for(unsigned i=0;i<kSlots;++i){addresses[i]=128+16*i;blk[i]=i;dst[i]=i;valid[i]=1;}
    for(unsigned q=0;q<4;++q){c.post[q]={in.n[q],in.safe[q],q*kTop};info[1]+=in.n[q];flags[1+q].ready=1;}
    if(in.badTotal)++info[1];
    if(in.badDescriptor>=0)valid[in.badDescriptor*kTop]=0;
    std::array<std::thread,4> cores;
    for(unsigned q=0;q<4;++q)cores[q]=std::thread([&,q]{lane=q;RunRow(&a,q*16);});
    for(auto& core:cores)core.join();
    bool failed=in.badPost>=0||in.badPlan||in.badTnd||in.badDrain||in.badTotal||in.badBlock>=0||in.badWait>=0||in.badDescriptor>=0||badRequiredQueue;
    assert(bool(world.status)==failed && world.finalized==1 && world.published==!failed && world.prefetch==1);
    unsigned totalCompleted=0,loans=0;std::array<unsigned,kSlots> reads{};
    for(unsigned q=0;q<4;++q){auto& queue=world.q[q];
        assert(queue.issued==queue.drained);
        if(in.badPlan || in.badTnd || in.badDescriptor>=0 || badRequiredQueue || in.badTotal)assert(!queue.issued);
        if(in.badDrain && in.safe[q]<in.n[q])assert(queue.positions.size()<=in.safe[q]);
        if(!failed){
            if(!in.n[q])assert(queue.positions.empty() && !queue.issued);
            uint64_t* mark=ticks.data()+16*256+q*16*32;
            assert(mark[20]==in.n[q] && queue.positions.size()==mark[20]+mark[21]-mark[22]);
            if(mark[21]){++loans;assert(mark[23]>=1 && mark[23]<=4 && mark[23]!=q+1);}
            unsigned i=0;
            for(unsigned pos:queue.positions){
                ++reads[pos];
                if(i<mark[20]-mark[22]) assert(pos==q*kTop+i);
                else {unsigned d=mark[23]-1;assert(pos==d*kTop+in.n[d]-mark[21]+i-(mark[20]-mark[22]));}
                ++i;
            }
        }
        totalCompleted+=records[q*16].completed;assert(bool(c.post[q].error)==failed);}
    assert(info[4]==totalCompleted && c.post[0].pending==0);
    if(!failed){
        assert(loans==unsigned(in.expectLoan));
        for(unsigned q=0;q<4;++q)for(unsigned i=0;i<kTop;++i)assert(reads[q*kTop+i]==unsigned(i<in.n[q]));
        assert(world.applied==4);
        for(unsigned q=0;q<4;++q){
            auto& rec=records[q*16];
            auto* mark=ticks.data()+16*256+q*16*32;
            assert(rec.indexDone>=mark[3] && rec.cqDone>=rec.indexDone);
            assert(mark[8]==world.q[q].issued && bool(rec.postBegin)==bool(rec.count));
            if(mark[8])assert(rec.postBegin<=mark[4] && mark[4]<=rec.postDone);
            if(mark[8]==2){assert(rec.postBegin<mark[3]); if(mark[21])assert(mark[4]<mark[3]);else assert(mark[4]>=rec.indexDone);}
        }
    }
    if(in.badPlan||in.badTnd||in.badDescriptor>=0||badRequiredQueue||in.badTotal)assert(world.applied==0);
    if(in.badBlock>=0 || in.badDrain || in.badWait>=0)assert(world.applied==4);
    if(in.badPost>=0){
        for(unsigned q=0;q<4;++q)assert(flags[8+q].reserved[2]!=0 || records[q*16].error!=0);
        assert(flags[8+in.badPost].reserved[2]!=0);
    }
    if(in.badDrain){assert(world.q[0].issued==1 && world.q[0].positions.size()==in.safe[0]);}
}
int main(){
    auto banks=[](unsigned n){return (n+127)/128;};
    for(unsigned n=0;n<=2048;++n)for(unsigned h=0;h<=n;++h)
        for(unsigned cap: {0U,63U,64U,128U,255U,2048U}){
            unsigned x=LoanSize(n,h,cap);
            if(x){
                assert(x<=cap && x<=n && h+x<n && banks(n-x)+banks(x)==banks(n));
                assert(n-(h+x)>=64 && x>=64);
            }
        }
    Input mixed{{4,3,0,5},{2,3,0,0}};
    run(mixed);auto drained=mixed;drained.pending=false;run(drained);auto c=mixed;c.cqDepth=2;run(c);c=mixed;c.blocking=0;run(c);c.badBlock=0;run(c);
    c=mixed;c.badPlan=true;run(c);c=mixed;c.badTnd=true;run(c);c=mixed;c.badTotal=true;run(c);c=mixed;c.badDescriptor=1;run(c);
    c=mixed;c.badPost=1;run(c);c=mixed;c.badPreflight=3;run(c);c=mixed;c.badWait=0;run(c);c=mixed;c.badDrain=true;run(c);
    Input high{{384,40,40,48},{384,40,40,48}};high.expectLoan=true;
    run(high); c=high;c.n[1]=c.safe[1]=0;run(c);
    c=high;c.badMr=0;c.expectLoan=false;run(c);c=high;c.badEid=1;c.expectLoan=false;run(c);
    c=high;c.localPath=1;run(c);
    c=high;c.badLocalSize=1;c.expectLoan=false;run(c);
    c=high;c.differentOutput=1;c.expectLoan=false;run(c);
    c=high;c.badRegion=1;c.expectLoan=false;run(c);
    c=high;c.n[1]=c.safe[1]=0;c.badPreflight=1;run(c);
    c=high;c.cqDepth=2;c.expectLoan=false;run(c);
    c=high;c.n[1]=c.safe[1]=0;c.cqDepth=2;c.expectLoan=false;run(c);
    c=high;c.safe[0]=200;c.expectLoan=false;run(c);
    c=high;c.n={384,0,0,0};c.safe=c.n;c.expectLoan=false;run(c);
    c=high;c.maxInflight=100;c.expectLoan=false;run(c);
    c=high;c.badPlan=true;run(c);c=high;c.badDescriptor=0;run(c);c=high;c.badPreflight=1;run(c);
    c=high;c.badPost=1;run(c);c=high;c.badWait=0;run(c);c=high;c.badWait=1;run(c);
    run(Input{{4,3,0,5},{4,3,0,5}});run(Input{{0,0,0,0},{0,0,0,0}});
}
