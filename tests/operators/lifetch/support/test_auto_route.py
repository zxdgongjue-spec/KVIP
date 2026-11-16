"""Execute the device routing and successful-feedback protocol on the CPU."""

from pathlib import Path
import subprocess

from test_recent_index import function


INCLUDE = Path(__file__).resolve().parents[4] / "csrc/operators/lightning_indexer_with_fetch/include"


def test_auto_route_decision_identity_and_success_commit(tmp_path):
    source = tmp_path / "route.cpp"
    source.write_text(r'''
#include <array>
#include <cassert>
#include <cstring>
#include "auto_route.h"
using namespace kvip_auto_route;
struct Context {
 uint64_t autoRoute,admissionControls,rowOwners,queryEnds;
 struct {uint32_t batch=16,history=131072,tokenBytes=656;} shape;
};
struct Fixture {
 std::array<Request,16> requests{};
 std::array<Feedback,17> feedback{};
 std::array<Pending,16> pending{};
 std::array<int32_t,64> controls{};
 std::array<uint32_t,16> owners{};
 std::array<uint32_t,64> ends{};
 Config config{1,17,uint64_t(requests.data()),uint64_t(feedback.data()),uint64_t(pending.data()),
               128,192,1,1,512,64,0,0,{0,0,0}};
 Context ctx{uint64_t(&config),uint64_t(controls.data()),uint64_t(owners.data()),uint64_t(ends.data())};
 Fixture(){ends.fill(131072);for(unsigned b=0;b<16;++b){owners[b]=b+1;requests[b]={b+1,1,1,4};}}
 void check(unsigned mode){
  // Different init blocks can observe block 0 before or after publication.
  // Neither pending nor controls may influence their read-only decisions.
  std::memset(pending.data(),0xa5,sizeof(pending));controls.fill(-12345);
  auto savedPending=pending;auto savedControls=controls;auto savedFeedback=feedback;
  bool noPrefetch=true;assert(Evaluate<false>(&ctx,noPrefetch));assert(noPrefetch==(mode==0));
  assert(!std::memcmp(savedPending.data(),pending.data(),sizeof(pending)));
  assert(savedControls==controls&&!std::memcmp(savedFeedback.data(),feedback.data(),sizeof(feedback)));
  assert(Prepare(&ctx));savedPending=pending;savedControls=controls;
  for(unsigned block=0;block<16;++block){
   noPrefetch=!bool(mode==0);assert(Evaluate<false>(&ctx,noPrefetch));assert(noPrefetch==(mode==0));
   assert(!std::memcmp(savedPending.data(),pending.data(),sizeof(pending))&&savedControls==controls);
  }
  // A read-only block does not even dereference publication storage.
  auto pendingAddress=config.pending,controlsAddress=ctx.admissionControls;
  config.pending=ctx.admissionControls=1;
  assert(Evaluate<false>(&ctx,noPrefetch)&&noPrefetch==(mode==0));
  config.pending=pendingAddress;ctx.admissionControls=controlsAddress;
  for(unsigned b=0;b<16;++b){
  assert(pending[b].mode==mode);assert(controls[b*4]==(mode?512:0));
  assert(controls[b*4+1]==64&&controls[b*4+2]==0&&controls[b*4+3]==0);}}
 void commit(unsigned reads){for(unsigned b=0;b<16;++b){pending[b].necessary=reads;assert(Commit(&config,b,0));}}
 void next(){for(auto& r:requests)++r.step;}
};
int main(){
 // All supported Q changes use exactly the same rounded-up read prediction.
 for(unsigned reads=0;reads<=8192;++reads)for(unsigned q=1;q<=4;++q)
  for(unsigned previous=1;previous<=4;++previous)
   assert(ScaleReads(reads,q,previous)==(uint64_t(reads)*q+previous-1)/previous);
 Fixture f;f.check(1);for(auto& p:f.pending)assert(p.reason==Unknown&&p.learnable);
 f.commit(80);f.next();f.check(0); // 1280 useful reads, below enter threshold.
 for(auto& p:f.pending)assert(p.reason==Predicted);
 f.commit(120);f.next();f.check(0); // Hysteresis retains low at 1920 reads.
 f.commit(160);f.next();f.check(1); // 2560 reads exceed exit threshold.
 f.commit(120);f.next();f.check(1); // Intermediate demand does not re-enter low.
 f.commit(80);f.next();f.check(0);
 // Failure is non-mutating, and the resulting step gap cannot predict.
 auto prior=f.feedback;assert(!Commit(&f.config,0,1));assert(!std::memcmp(prior.data(),f.feedback.data(),sizeof(prior)));
 f.next();f.check(1);assert(f.pending[0].reason==StaleStep&&f.pending[0].learnable);f.commit(80);
 // Duplicate/backward steps never train; this is safe without a transport fault.
 prior=f.feedback;f.check(1);for(unsigned b=0;b<16;++b)assert(!f.pending[b].learnable&&!Commit(&f.config,b,0));
 for(auto& r:f.requests)--r.step;f.check(1);for(unsigned b=0;b<16;++b)assert(!Commit(&f.config,b,0));
 assert(!std::memcmp(prior.data(),f.feedback.data(),sizeof(prior)));
 // Reused owner: higher generation starts high and may establish new history.
 f.requests[0].generation=2;f.requests[0].step=1;f.check(1);assert(f.pending[0].learnable);
 f.pending[0].necessary=10;assert(Commit(&f.config,0,0));assert(f.feedback[1].generation==2);
 f.requests[0].generation=1;f.requests[0].step=100;f.check(1);assert(!f.pending[0].learnable&&!Commit(&f.config,0,0));
 // No owner can be updated by two concurrently active rows.
 Fixture duplicate;duplicate.requests[1].owner=duplicate.requests[0].owner;
 duplicate.owners[1]=duplicate.owners[0];duplicate.check(1);
 assert(!duplicate.pending[0].learnable&&!duplicate.pending[1].learnable);
 assert(!Commit(&duplicate.config,0,0)&&!Commit(&duplicate.config,1,0));
 // IDs outside the bitmask fast path retain the full-width duplicate check.
 Fixture wide;std::array<Feedback,128> wideFeedback{};
 wide.config.owners=128;wide.config.feedback=uint64_t(wideFeedback.data());
 wide.requests[0].owner=wide.owners[0]=67;wide.check(1);
 assert(wide.pending[0].learnable);wide.pending[0].necessary=10;
 assert(Commit(&wide.config,0,0)&&wideFeedback[67].valid==kValid);
 wide.next();wide.requests[1].owner=wide.owners[1]=67;wide.check(1);
 assert(!wide.pending[0].learnable&&!wide.pending[1].learnable);
 // An inactive owner can share a live row's slot without rejecting that row.
 Fixture padding;padding.requests[1].owner=padding.owners[1]=padding.owners[0];
 padding.requests[1].queries=0;for(unsigned i=4;i<8;++i)padding.ends[i]=0;
 padding.check(1);assert(padding.pending[0].learnable&&!padding.pending[1].learnable);
 // Capture/warmup may run the full graph but cannot populate predictor state.
 Fixture warm;warm.config.learning=0;warm.check(1);warm.pending[0].necessary=10;
 assert(warm.pending[0].reason==LearningDisabled&&!Commit(&warm.config,0,0));
 for(auto& value:warm.feedback)assert(value.valid==0);
 // Query counts scale previous necessary reads, while actual query ends set LI work.
 Fixture q;q.check(1);q.commit(80);q.next();
 for(unsigned b=0;b<16;++b){q.requests[b].queries=2;q.ends[b*4+2]=q.ends[b*4+3]=0;}
 q.check(0); // Both work and estimated bytes halve: same decision.
 for(auto& end:q.ends)if(end)end/=4;q.check(1); // Less LI work changes the ratio.
 // Owner-keyed feedback survives row reordering; a new request still starts high.
 Fixture reordered;reordered.check(1);reordered.commit(80);reordered.next();
 std::swap(reordered.requests[0],reordered.requests[15]);
 std::swap(reordered.owners[0],reordered.owners[15]);
 for(unsigned i=0;i<4;++i)std::swap(reordered.ends[i],reordered.ends[60+i]);
 reordered.check(0);reordered.requests[0].generation=2;reordered.requests[0].step=1;reordered.check(1);
 // A wrong low prediction changes no correctness work: actual successful
 // demand becomes the next round's feedback and switches the same graph high.
 Fixture recovery;recovery.check(1);recovery.commit(0);recovery.next();recovery.check(0);
 recovery.commit(4096);recovery.next();recovery.check(1);
 // Current packed token bytes affect Get demand; the first changed shape is unknown.
 Fixture packed;packed.check(1);packed.commit(80);packed.next();packed.ctx.shape.tokenBytes=2624;
 packed.check(1);assert(packed.pending[0].reason==ShapeChange);packed.commit(80);packed.next();packed.check(1);
 // Invalid IDs/queries are not allowed to index feedback or commit partial samples.
 for(unsigned fault=0;fault<5;++fault){Fixture bad;
  if(fault==0)bad.requests[0].owner=UINT64_MAX;
  if(fault==1)bad.requests[0].generation=0;
  if(fault==2)bad.requests[0].step=0;
  if(fault==3)bad.requests[0].queries=3;
  if(fault==4)bad.ends[0]=bad.ctx.shape.history+1;
  bad.check(1);assert(!bad.pending[0].learnable&&!Commit(&bad.config,0,0));
 }
 Fixture empty;empty.ends.fill(0);for(auto& r:empty.requests)r.queries=0;empty.check(1);
 for(unsigned b=0;b<16;++b)assert(!Commit(&empty.config,b,0));
 Fixture invalid;invalid.config.fullWindow=0;auto controls=invalid.controls;
 bool noPrefetch=true;auto pending=invalid.pending;
 assert(!Evaluate<false>(&invalid.ctx,noPrefetch)&&!noPrefetch);
 assert(!std::memcmp(pending.data(),invalid.pending.data(),sizeof(pending))&&controls==invalid.controls);
 assert(!Prepare(&invalid.ctx));assert(controls==invalid.controls);
}
''')
    binary = tmp_path / "route"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
        "-I", str(INCLUDE), str(source), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)


def test_auto_route_device_publication_order():
    kernel = INCLUDE.parent / "op_kernel"
    source = (kernel / "streaming.cpp").read_text()
    final = function(source, "__aicore__ inline void FinalizeStreaming(")
    commit = final.index("kvip_auto_route::Commit(config,b,error|updateError)")
    prefix = final[:commit]
    assert prefix.rfind("CleanCacheLines(record,128)") > prefix.rfind("CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(pending),128)")
    assert prefix.rfind("CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(pending),128)") > prefix.rfind("CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(config),128)")
    assert final.index("for(uint32_t f=6;f<12;++f)") < commit
    assert final.index("uint32_t updateError=*updateStatus") < commit
    assert "if(kvip_auto_route::Commit(config,b,error|updateError))kvaf_device::CleanCacheLines(record,128)" in final
    producer = function(source, "__aicore__ inline uint32_t Producer(")
    assert "auto_route::Config" not in producer
    assert "auto_route::Pending" not in producer
    assert "usefulPrefetch=ctl.GetValue(8)" in producer
    assert "return usefulPrefetch;" in producer
    assert "uint32_t useful=Producer(a,b);FinalizeStreaming(a,b,useful)" in source
    write = final.index("pending->usefulPrefetch=")
    assert final[:write].rfind("CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(pending),128)") >= 0
    assert final[:write].rfind("CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),kRowStatusWords*4)") >= 0
    assert "pending->necessary=uint64_t(info[1])+pending->usefulPrefetch" in final
    assert "pending->usefulPrefetch=usefulPrefetch" in final
    assert final.index("CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(pending),128)", commit) > commit
    indexer = (kernel / "indexer.cpp").read_text()
    assert indexer.index("CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->admissionControls)") < indexer.index("rotating::NoPrefetchRound(ctx)")


def test_actual_prefetch_usefulness_counter_excludes_waste(tmp_path):
    kernel = INCLUDE.parent / "op_kernel"
    source = function((kernel / "resolver_staged_dma2.h").read_text(), "inline void StagedInitializeReady(")
    counter = source[source.index("uint32_t counts=missing;"):source.index("asc_syncthreads();StagedPrefix(counts,ctl);")]
    cpp = tmp_path / "useful.cpp"
    cpp.write_text(r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <vector>
#define KVIP_AUTO_ROUTE 1
uint32_t Count(uint32_t tid,uint32_t missing,uint32_t* lru,uint32_t used,
               uint32_t resident,uint32_t* ctl,bool recordFeedback){
''' + counter + r'''
 return counts;
}
int main(){
 constexpr unsigned resident=16384;std::mt19937 rng(53);
 std::vector<uint32_t> slots(resident);std::iota(slots.begin(),slots.end(),0U);
 for(unsigned iteration=0;iteration<100;++iteration){
  std::shuffle(slots.begin(),slots.end(),rng);unsigned used=iteration*19%2049;
  std::array<uint32_t,1536> ctl{};std::set<uint32_t> expectedPrefetch,expectedFinal;
  std::array<unsigned,512> missing{};
  for(unsigned i=0;i<8192;++i){
   unsigned type=rng()%4,index=rng()%2048,token=0,slot=UINT32_MAX;
   if(type==0){token=resident+index;if(index<used)slot=slots[index];}
   if(type==1){token=slots[2048+rng()%(resident-2048)];slot=token;}
   if(type==2)token=slots[index]; // Reserved hot token must be refetched.
   if(type==3){token=resident+2048+rng()%512;slot=resident+token%512;} // ABC hit.
   if(slot<resident)ctl[1024+slot/32]|=1U<<(slot%32);
   if(type==0 && index<used)expectedPrefetch.insert(token);
   if(slot==UINT32_MAX){++missing[i%512];expectedFinal.insert(token);}
  }
  unsigned packed=0;
  for(unsigned tid=0;tid<512;++tid)packed+=Count(tid,missing[tid],slots.data(),used,resident,ctl.data(),true);
  assert((packed>>16)==expectedPrefetch.size());
  assert((packed&0xffff)==std::accumulate(missing.begin(),missing.end(),0U));
  assert(expectedFinal.size()+(packed>>16)<=8192);
  // The counter never charges an unused admitted read to the next decision.
  assert((packed>>16)<=used);
  for(unsigned tid=0;tid<512;++tid)assert(Count(tid,missing[tid],slots.data(),used,resident,ctl.data(),false)==missing[tid]);
 }
}
''')
    binary = tmp_path / "useful"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
        str(cpp), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)
