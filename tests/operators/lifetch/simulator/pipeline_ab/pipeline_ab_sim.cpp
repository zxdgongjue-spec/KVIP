// SPDX-License-Identifier: Apache-2.0
// Single-request comparison: real staged resolver + existing 1/4-AIV update
// against two-AIV streaming shadow-swap experiment. No LI, Fetch or SFA.
#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "context.h"
#include "pipeline_types.h"
#ifdef PIPELINE_PARALLEL
extern "C" uint32_t aclrtlaunch_test_pipeline_ab_parallel(uint32_t,void*,void*);
#define PIPELINE_LAUNCH aclrtlaunch_test_pipeline_ab_parallel
#elif defined(PIPELINE_BATCH)
extern "C" uint32_t aclrtlaunch_test_pipeline_ab_batch(uint32_t,void*,void*);
#define PIPELINE_LAUNCH aclrtlaunch_test_pipeline_ab_batch
#elif defined(PIPELINE_OPT)
extern "C" uint32_t aclrtlaunch_test_pipeline_ab_opt(uint32_t,void*,void*);
#define PIPELINE_LAUNCH aclrtlaunch_test_pipeline_ab_opt
#else
extern "C" uint32_t aclrtlaunch_test_pipeline_ab(uint32_t,void*,void*);
#define PIPELINE_LAUNCH aclrtlaunch_test_pipeline_ab
#endif
extern "C" uint32_t aclrtlaunch_test_ready_resolver(uint32_t,void*,void*);
extern "C" uint32_t kvop_direct_update(uint32_t,void*,void*);
extern "C" uint32_t kvop_direct_update_plan_words();
static void check(int v){if(v)throw std::runtime_error("ACL error "+std::to_string(v));}
static void graphCheck(int v,const char* stage){if(v)throw std::runtime_error(std::string(stage)+" ACL error "+std::to_string(v));}
template<class T> void* upload(const std::vector<T>& v){void* p=nullptr;check(aclrtMalloc(&p,v.size()*sizeof(T),ACL_MEM_MALLOC_HUGE_FIRST));check(aclrtMemcpy(p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));return p;}
template<class T> std::vector<T> download(void* p,size_t n){std::vector<T> v(n);check(aclrtMemcpy(v.data(),n*sizeof(T),p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
int main(int argc,char**argv){try{
 const std::string mode=argc>1?argv[1]:"serial", fixture=argc>2?argv[2]:"normal";
 if((mode!="serial"&&mode!="pipeline")||(fixture!="normal"&&fixture!="collision"&&fixture!="cross-query"&&fixture!="smoke"))throw std::runtime_error("mode serial|pipeline; fixture normal|collision|cross-query");
 constexpr uint32_t h=131072,r=16384,k=8192,cap=10240; const uint32_t used=fixture=="smoke"?8:512;
 const int device=argc>3?std::stoi(argv[3]):0;if(device<0)throw std::runtime_error("device must be nonnegative");
 const bool graph=argc>4&&std::string(argv[4])=="graph";
 const int repeats=argc>5?std::stoi(argv[5]):1;if(repeats<1)throw std::runtime_error("repeats must be positive");
 const uint32_t batch=argc>6?std::stoul(argv[6]):1;if(batch<1||batch>16)throw std::runtime_error("batch must be 1..16");
 check(aclInit(nullptr));check(aclrtSetDevice(device));aclrtStream stream;check(aclrtCreateStream(&stream));
 rotating::Context ctx{};ctx.shape.batch=batch;ctx.shape.history=h;ctx.shape.resident=r;ctx.shape.finalWorkers=1;ctx.shape.tokenBytes=656;ctx.outputStride=r;
 std::vector<void*> allocations;std::vector<std::pair<void*,std::vector<uint8_t>>> initial;
 auto save=[&](void* p,auto const& v){std::vector<uint8_t> bytes(v.size()*sizeof(v[0]));std::memcpy(bytes.data(),v.data(),bytes.size());initial.emplace_back(p,std::move(bytes));};
 auto bind=[&](uint64_t& f,auto const& v){void*p=upload(v);allocations.push_back(p);save(p,v);f=(uint64_t)p;};
 std::vector<uint64_t> keys(uint64_t(h)*16),addresses(h),logical(h);std::vector<uint32_t> scales(h),lru(2048),slots(cap),changes(cap,UINT32_MAX),mapping(h,UINT32_MAX),selected(k),loc(k,UINT32_MAX),bitmap(r/32),valid(r,1);std::vector<int32_t> reserved(r,-1);
 for(uint32_t i=0;i<h;++i){addresses[i]=1000000ULL+i;logical[i]=2000000ULL+i;scales[i]=3000000U+i;for(uint32_t f=0;f<16;++f)keys[uint64_t(i)*16+f]=uint64_t(i)*16+f+100;}
 for(uint32_t i=0;i<2048;++i){lru[i]=slots[i]=i;reserved[i]=i;}
 for(uint32_t i=0;i<used;++i){uint32_t token=r+(i<64?i:10000+i);mapping[token]=r+i;changes[i]=token;}
 for(uint32_t q=0;q<4;++q){for(uint32_t i=0;i<205;++i)selected[q*2048+i]=(fixture=="cross-query"&&q==1&&i==0)?r+100:r+(q%2)*205+i;for(uint32_t i=205;i<2048;++i)selected[q*2048+i]=4096+i-205;}
 if(fixture=="smoke")for(uint32_t q=0;q<4;++q)for(uint32_t i=0;i<2048;++i)selected[q*2048+i]=i<2?r+16+q%2*2+i:4096+i;
 if(fixture=="collision"){selected[205]=0;selected[2*2048+205]=0;}
 for(uint32_t i=0;i<k;++i){uint32_t token=selected[i],slot=UINT32_MAX;if(token<r&&reserved[token]<0)slot=token;else {auto v=mapping[token];if(v>=r&&v<r+2048)slot=lru[v-r];else slot=v;}loc[i]=slot;if(slot<r)bitmap[slot/32]|=1U<<(slot%32);}
 auto repeat=[&](auto& v){auto single=v;v.reserve(single.size()*batch);for(uint32_t b=1;b<batch;++b)v.insert(v.end(),single.begin(),single.end());};
 repeat(keys);repeat(scales);repeat(addresses);repeat(logical);repeat(lru);repeat(slots);repeat(changes);repeat(mapping);repeat(selected);repeat(loc);repeat(bitmap);repeat(valid);repeat(reserved);
 // Different contents per request detect accidental cross-batch addressing.
 for(uint32_t b=1;b<batch;++b)for(uint32_t i=0;i<h;++i){uint64_t row=uint64_t(b)*h+i;addresses[row]=1000000ULL+row;logical[row]=2000000ULL+row;scales[row]=3000000U+row;for(uint32_t f=0;f<16;++f)keys[row*16+f]=row*16+f+100;}

 bind(ctx.args,std::vector<uint32_t>(uint64_t(batch)*k*16,0xdeadbeef));bind(ctx.selected,selected);bind(ctx.mapping,mapping);bind(ctx.lruSlots,lru);bind(ctx.reservedHot,reserved);bind(ctx.changeSlots,slots);bind(ctx.changes,changes);bind(ctx.changeCount,std::vector<uint32_t>(batch,2048));bind(ctx.hotValid,valid);bind(ctx.locations,loc);bind(ctx.hits,bitmap);
 bind(ctx.sourceBlocks,std::vector<uint32_t>(batch*k));bind(ctx.sourceOffsets,std::vector<uint32_t>(batch*k));bind(ctx.destinations,std::vector<uint32_t>(batch*k));bind(ctx.count,std::vector<uint32_t>(batch));bind(ctx.fallbackAddresses,std::vector<uint64_t>(batch*k));bind(ctx.rowStatus,std::vector<uint32_t>(batch*32));
 uint64_t dkeys,dscale,daddr,dlogical;bind(dkeys,keys);bind(dscale,scales);bind(daddr,addresses);bind(dlogical,logical);
 std::vector<rotating::Row> rows(batch);for(uint32_t b=0;b<batch;++b){rows[b].slots=used;rows[b].addresses=daddr+uint64_t(b)*h*sizeof(uint64_t);}bind(ctx.rows,rows);
 uint64_t dctx;bind(dctx,std::vector<rotating::Context>{ctx});
 pipeline_ab::Args args{};args.context=dctx;args.keys=dkeys;args.scales=dscale;args.addresses=daddr;args.logical=dlogical;args.valid=ctx.hotValid;args.history=h;args.resident=r;args.prefetchCount=used;args.maxEvents=512+k;
 bind(args.flags,std::vector<pipeline_ab::BatchFlag>(batch*5));bind(args.eventsSrc,std::vector<uint32_t>(batch*(512+k),UINT32_MAX));bind(args.eventsDst,std::vector<uint32_t>(batch*(512+k),UINT32_MAX));bind(args.workerStatus,std::vector<uint32_t>(batch*32));bind(args.times,std::vector<uint64_t>(batch*64));uint64_t dargs;bind(dargs,std::vector<pipeline_ab::Args>{args});
 std::vector<uint64_t> u(28);u[0]=dkeys;u[1]=dscale;u[2]=daddr;u[3]=dlogical;u[4]=ctx.changeSlots;u[5]=ctx.changes;u[7]=h;u[8]=ctx.rows;u[9]=ctx.hotValid;u[24]=h;u[25]=r;u[26]=ctx.changeCount;
 auto bindu=[&](unsigned i,auto const& v){void*p=upload(v);allocations.push_back(p);save(p,v);u[i]=(uint64_t)p;};bindu(6,std::vector<uint32_t>(batch));bindu(10,std::vector<uint32_t>(batch*kvop_direct_update_plan_words()));bindu(27,std::vector<uint32_t>(batch*40960));uint64_t du;bind(du,u);
 auto launch=[&](){if(mode=="serial"){check(aclrtlaunch_test_ready_resolver(batch,stream,(void*)dctx));check(kvop_direct_update(batch,(void*)du,stream));}else check(PIPELINE_LAUNCH(2*batch,stream,(void*)dargs));};
 aclmdlRI model=nullptr;
 if(graph){
  // Register kernels before capture; the fixture is restored outside timing.
  launch();graphCheck(aclrtSynchronizeStream(stream),"pre-capture warmup synchronization");
  for(const auto& snap:initial)graphCheck(aclrtMemcpy(snap.first,snap.second.size(),snap.second.data(),snap.second.size(),ACL_MEMCPY_HOST_TO_DEVICE),"pre-capture restore");
  graphCheck(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"graph capture begin");launch();graphCheck(aclmdlRICaptureEnd(stream,&model),"graph capture end");
 }
 aclrtEvent start,end;check(aclrtCreateEvent(&start));check(aclrtCreateEvent(&end));float eventMs=0;int evt=0;
 for(int iteration=0;iteration<repeats;++iteration){
  if(graph){for(const auto& snap:initial)graphCheck(aclrtMemcpy(snap.first,snap.second.size(),snap.second.data(),snap.second.size(),ACL_MEMCPY_HOST_TO_DEVICE),"replay fixture restore");}
  graphCheck(aclrtRecordEvent(start,stream),"start event record");if(graph)graphCheck(aclmdlRIExecuteAsync(model,stream),"graph replay submission");else launch();
  graphCheck(aclrtRecordEvent(end,stream),"end event record");graphCheck(aclrtSynchronizeStream(stream),"replay synchronization");evt=aclrtEventElapsedTime(&eventMs,start,end);
  if(graph){
   // Status checks and fixture restoration are deliberately outside the event.
   for(uint32_t b=0;b<batch;++b){
    auto row=download<uint32_t>((void*)(ctx.rowStatus+uint64_t(b)*32*sizeof(uint32_t)),32);
    auto worker=mode=="serial"?download<uint32_t>((void*)(u[6]+uint64_t(b)*sizeof(uint32_t)),1):download<uint32_t>((void*)(args.workerStatus+uint64_t(b)*32*sizeof(uint32_t)),32);
    uint32_t expected=fixture=="smoke"?4:fixture=="collision"?347:346;
    if(row[0]||worker[0]||row[1]!=expected)throw std::runtime_error("replay status mismatch iteration="+std::to_string(iteration)+" batch="+std::to_string(b));
   }
   std::printf("GRAPH_SAMPLE iteration=%d event_status=%d event_ms=%.6f\n",iteration,evt,eventMs);
  }
 }
 std::vector<uint32_t> firstStatus;
 for(uint32_t b=0;b<batch;++b){
 auto status=download<uint32_t>((void*)(ctx.rowStatus+uint64_t(b)*32*sizeof(uint32_t)),32);if(status[0])throw std::runtime_error("resolver error "+std::to_string(status[0]));
 if(fixture=="smoke" && status[1]!=4)throw std::runtime_error("smoke expected 4 distinct Q misses; observed="+std::to_string(status[1]));
 if(mode=="serial"){auto st=download<uint32_t>((void*)(u[6]+uint64_t(b)*sizeof(uint32_t)),1);if(st[0])throw std::runtime_error("serial planner error "+std::to_string(st[0]));}
 else {auto st=download<uint32_t>((void*)(args.workerStatus+uint64_t(b)*32*sizeof(uint32_t)),32);if(st[0])throw std::runtime_error("pipeline worker error "+std::to_string(st[0]));}
 auto finalLoc=download<uint32_t>((void*)(ctx.locations+uint64_t(b)*k*sizeof(uint32_t)),k), change=download<uint32_t>((void*)(ctx.changes+uint64_t(b)*cap*sizeof(uint32_t)),cap),changeSlots=download<uint32_t>((void*)(ctx.changeSlots+uint64_t(b)*cap*sizeof(uint32_t)),cap);
 auto outKeys=download<uint64_t>((void*)(dkeys+uint64_t(b)*uint64_t(h)*16*sizeof(uint64_t)),uint64_t(h)*16);auto outScales=download<uint32_t>((void*)(dscale+uint64_t(b)*h*sizeof(uint32_t)),h);auto outAddr=download<uint64_t>((void*)(daddr+uint64_t(b)*h*sizeof(uint64_t)),h);auto outLogical=download<uint64_t>((void*)(dlogical+uint64_t(b)*h*sizeof(uint64_t)),h);
 // Verify every field/row remains an internally consistent permutation.
 std::vector<uint32_t> seen(h);for(uint32_t pos=0;pos<h;++pos){auto id=outLogical[pos];if(id<2000000ULL+uint64_t(b)*h||id>=2000000ULL+uint64_t(b+1)*h)throw std::runtime_error("logical range");uint32_t orig=id-2000000ULL-uint64_t(b)*h;if(seen[orig]++)throw std::runtime_error("logical duplicate");
  if(outScales[pos]!=scales[uint64_t(b)*h+orig]||outAddr[pos]!=addresses[uint64_t(b)*h+orig])throw std::runtime_error("metadata mismatch pos="+std::to_string(pos));for(uint32_t f=0;f<16;++f)if(outKeys[uint64_t(pos)*16+f]!=keys[(uint64_t(b)*h+orig)*16+f])throw std::runtime_error("key mismatch pos="+std::to_string(pos));}
 // Independent ordered-swap oracle: every physical row must match, not only
 // selected hot slots. Consumer publication order is prefetch then Q1..Q4.
 std::vector<uint32_t> reference(h),position(h);std::iota(reference.begin(),reference.end(),0);std::iota(position.begin(),position.end(),0);
 auto expectedSwap=[&](uint32_t original,uint32_t target){if(original>=h||target>=r)throw std::runtime_error("oracle record range");
  uint32_t src=position[original],old=reference[target];std::swap(reference[src],reference[target]);position[original]=target;position[old]=src;};
 if(mode=="pipeline"){
  auto es=download<uint32_t>((void*)(args.eventsSrc+uint64_t(b)*(512+k)*sizeof(uint32_t)),512+k),ed=download<uint32_t>((void*)(args.eventsDst+uint64_t(b)*(512+k)*sizeof(uint32_t)),512+k);
#ifdef PIPELINE_PARALLEL
  // Independent CPU oracle for the device's per-phase partial permutations.
  // Source positions are resolved using the preceding phase's layout.
  auto applyPhase=[&](uint32_t begin,uint32_t count){
   for(uint32_t base=0;base<count;base+=512){
    uint32_t n=std::min(512U,count-base);
    std::vector<uint32_t> src(n),dst(n);std::set<uint32_t> srcSet,dstSet;
    for(uint32_t i=0;i<n;++i){uint32_t row=es[begin+base+i];dst[i]=ed[begin+base+i];
     if(row>=h||dst[i]>=r)throw std::runtime_error("parallel oracle record range");
     src[i]=position[row];if(!srcSet.insert(src[i]).second||!dstSet.insert(dst[i]).second)throw std::runtime_error("parallel oracle duplicate");}
    std::vector<uint32_t> out,free;for(uint32_t i=0;i<n;++i){if(!srcSet.count(dst[i]))out.push_back(dst[i]);if(!dstSet.count(src[i]))free.push_back(src[i]);}
    if(out.size()!=free.size())throw std::runtime_error("parallel oracle closure");
    auto previous=reference;for(uint32_t i=0;i<n;++i)reference[dst[i]]=previous[src[i]];
    for(size_t i=0;i<out.size();++i)reference[free[i]]=previous[out[i]];
    for(uint32_t i=0;i<h;++i)position[reference[i]]=i;
   }
  };
  applyPhase(0,used);uint32_t total=0;
  for(uint32_t q=0;q<4;++q){applyPhase(512+total,status[4+q]);total+=status[4+q];}
  if(total!=status[1])throw std::runtime_error("parallel oracle count mismatch");
  for(uint32_t pos=0;pos<h;++pos)if(outLogical[pos]!=logical[uint64_t(b)*h+reference[pos]])throw std::runtime_error("parallel oracle mismatch pos="+std::to_string(pos));
#else
  for(uint32_t i=0;i<used;++i)expectedSwap(es[i],ed[i]);
  for(uint32_t i=0;i<status[1];++i)expectedSwap(es[512+i],ed[512+i]);
  for(uint32_t pos=0;pos<h;++pos)if(outLogical[pos]!=logical[uint64_t(b)*h+reference[pos]])throw std::runtime_error("ordered oracle mismatch pos="+std::to_string(pos));
#endif
 }
 auto outValid=download<uint32_t>((void*)(ctx.hotValid+uint64_t(b)*r*sizeof(uint32_t)),r);
 for(uint32_t pos=0;pos<r;++pos)if(outValid[pos]!=valid[uint64_t(b)*r+pos])throw std::runtime_error("hot validity mismatch pos="+std::to_string(pos));
 for(uint32_t i=0;i<k;++i){uint32_t pos=finalLoc[i];if(pos>=r||outLogical[pos]!=logical[uint64_t(b)*h+selected[uint64_t(b)*k+i]])throw std::runtime_error("Q4 ownership mismatch i="+std::to_string(i));}
 for(uint32_t i=0;i<2048;++i)if(change[i]!=UINT32_MAX && outLogical[changeSlots[i]]!=logical[uint64_t(b)*h+change[i]])throw std::runtime_error("producer change mismatch");
 uint64_t layoutHash=1469598103934665603ULL,hotHash=1469598103934665603ULL;
 for(uint32_t pos=0;pos<h;++pos){layoutHash=(layoutHash^outLogical[pos])*1099511628211ULL;if(pos<r)hotHash=(hotHash^outLogical[pos])*1099511628211ULL;}
 std::printf("LAYOUT_HASH mode=%s fixture=%s batch_index=%u full=%016llx hot=%016llx\n",mode.c_str(),fixture.c_str(),b,(unsigned long long)layoutHash,(unsigned long long)hotHash);
 if(b==0)firstStatus=status;
 else if(status[1]!=firstStatus[1] || !std::equal(status.begin()+4,status.begin()+8,firstStatus.begin()+4))
  throw std::runtime_error("batch status mismatch at request "+std::to_string(b));
 }
 auto status=firstStatus;
 std::printf("PASS mode=%s fixture=%s final_miss=%u q_counts=[%u,%u,%u,%u] event_status=%d event_ms=%.6f\n",mode.c_str(),fixture.c_str(),status[1],status[4],status[5],status[6],status[7],evt,eventMs);
#ifdef PIPELINE_PARALLEL
 constexpr unsigned consumerBase=32;
#else
 constexpr unsigned consumerBase=0;
#endif
 if(mode=="pipeline")for(uint32_t b=0;b<batch;++b){auto stamps=download<uint64_t>((void*)(args.times+uint64_t(b)*64*sizeof(uint64_t)),64);
  std::printf("PIPELINE_CYCLES batch=%u q_publish=[%llu,%llu,%llu,%llu] resolver_end=%llu update=[",b,(unsigned long long)stamps[4],(unsigned long long)stamps[5],(unsigned long long)stamps[6],(unsigned long long)stamps[7],(unsigned long long)stamps[8]);
  for(int i=0;i<5;++i)std::printf("%s%llu:%llu",i?",":"",(unsigned long long)stamps[consumerBase+12+2*i],(unsigned long long)stamps[consumerBase+13+2*i]);
  std::printf("] metadata=%llu:%llu finish=%llu\n",(unsigned long long)stamps[consumerBase+22],(unsigned long long)stamps[consumerBase+23],(unsigned long long)stamps[consumerBase+24]);}

 if(graph)check(aclmdlRIDestroy(model));check(aclrtDestroyEvent(start));check(aclrtDestroyEvent(end));for(void*p:allocations)check(aclrtFree(p));check(aclrtDestroyStream(stream));check(aclrtResetDevice(device));check(aclFinalize());
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
