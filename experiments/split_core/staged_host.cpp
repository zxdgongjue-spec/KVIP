// Actual QK -> device canonical -> current complete streaming tail.
// Model builds replace final native BatchGet only, with the declared GM adapter.
#include <acl/acl.h>
#include "context.h"
#include "pipeline_types.h"
#include "staged_schedule.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>
static std::string lastFailure;
static void check(uint32_t c){if(c){lastFailure="ACL error "+std::to_string(c);throw std::runtime_error(lastFailure);}}
static void require(bool c,const std::string& s){if(!c){lastFailure=s;throw std::runtime_error(s);}}
struct FailureLog {
 std::string path;
 ~FailureLog(){if(std::uncaught_exceptions()>0){try{std::ofstream(path)<<lastFailure<<"\n";}catch(...){}}}
};
struct Memory {
 std::vector<void*> allocations;
 template<class T> uint64_t put(const std::vector<T>& v){void* p=nullptr;size_t n=v.size()*sizeof(T);require(n,"empty allocation");
  check(aclrtMalloc(&p,n,ACL_MEM_MALLOC_HUGE_FIRST));allocations.push_back(p);check(aclrtMemcpy(p,n,v.data(),n,ACL_MEMCPY_HOST_TO_DEVICE));return uint64_t(p);}
 template<class T> std::vector<T> get(uint64_t p,size_t n){std::vector<T> v(n);check(aclrtMemcpy(v.data(),n*sizeof(T),(void*)p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
 template<class T> void set(uint64_t p,const std::vector<T>& v){check(aclrtMemcpy((void*)p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));}
 ~Memory(){for(auto p:allocations)aclrtFree(p);}
};
#include "standard_host_helpers.h"
extern "C" uint32_t kvip_lifetch_staged_forward(void*,void*,void*,const rotating::RuntimeShape*,const kvip_staged::Plan*);
extern "C" uint32_t kvip_lifetch_streaming_forward_v2(void*,void*,void*,void*,const rotating::RuntimeShape*,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_kvip_model_clock(uint32_t,void*,void*,uint32_t);
struct Options {
 uint32_t windows=2,post=0,prepare=0,service=0,hot=0,cold=0,workers=2;
};
static uint32_t choice(const std::string& value,const std::vector<std::string>& names){
 auto at=std::find(names.begin(),names.end(),value);require(at!=names.end(),"invalid option "+value);return at-names.begin();
}
static void Run(const std::string& out,uint32_t batch,uint32_t h,uint32_t r,uint32_t c,uint32_t v,uint32_t miss,aclrtStream stream,const std::string& forced,const Options& options){
 using namespace rotating;Memory m;FailureLog failure{out+"/failure.txt"};
 RuntimeShape s{};s.batch=batch;s.history=h;s.resident=r;s.indexHeads=32;s.aicCount=c;s.tokenBytes=656;s.rounds=options.windows;
 require(forced=="legacy" || forced=="history" || forced=="bsq" || forced=="auto","invalid route");
 if(forced=="legacy"){
  require(!options.post && !options.prepare && !options.service && !options.hot && !options.cold,"legacy needs default score controls");
  s.splitQ=1;check(rotating::PrepareShape(&s));require(s.finalWorkers==4,"legacy four records");
 }else check(kvip_staged::PrepareShape(&s,v,options.workers));
 auto qseed=ReadFixture<uint8_t>("arg0.bin"),kseed=ReadFixture<uint8_t>("arg1.bin");
 auto wseed=ReadFixture<uint16_t>("weights-bf16.bin");auto zseed=ReadFixture<float>("arg3.bin"),ksseed=ReadFixture<float>("arg4.bin");
 auto oracle=ReadFixture<uint16_t>("oracle.bin");auto expected=ReadFixture<uint32_t>("selected.bin");
 require(qseed.size()==4*32*128 && kseed.size()==uint64_t(h)*128 && oracle.size()==4ULL*h && expected.size()==8192,"fixture size");
 std::vector<uint8_t> keys(uint64_t(batch)*h*128+128,0x5a);std::vector<float> scales(uint64_t(batch)*h+32,7.25f);
 std::vector<uint64_t> logical(uint64_t(batch)*h),addresses(logical.size());std::set<uint32_t> cold;
 for(auto t:expected)if(t>=r)cold.insert(t);require(cold.size()==2*miss,"unique miss fixture");
 // Only selected remote cold payloads need backing; all hot slots have unique payloads.
 const uint32_t remoteStride=r+cold.size();std::vector<uint64_t> payload(uint64_t(batch)*remoteStride*82);
 auto payloadGm=m.put(payload);
 for(uint32_t b=0;b<batch;++b){
  std::copy(kseed.begin(),kseed.end(),keys.begin()+uint64_t(b)*h*128);std::copy(ksseed.begin(),ksseed.end(),scales.begin()+uint64_t(b)*h);
  std::vector<uint32_t> remoteTokens(r);std::iota(remoteTokens.begin(),remoteTokens.end(),0);remoteTokens.insert(remoteTokens.end(),cold.begin(),cold.end());
  for(uint32_t i=0;i<remoteTokens.size();++i){uint32_t t=remoteTokens[i];for(uint32_t f=0;f<82;++f)payload[(uint64_t(b)*remoteStride+i)*82+f]=ExpectedPayload(b,t,f);}
  for(uint32_t t=0;t<h;++t){logical[uint64_t(b)*h+t]=uint64_t(b)*h+t;addresses[uint64_t(b)*h+t]=payloadGm+uint64_t(b)*remoteStride*656;}
  for(uint32_t i=0;i<remoteTokens.size();++i)addresses[uint64_t(b)*h+remoteTokens[i]]=payloadGm+(uint64_t(b)*remoteStride+i)*656;
 }
 m.set(payloadGm,payload);std::vector<uint64_t>().swap(payload);
 auto keysGm=m.put(keys),scalesGm=m.put(scales),logicalGm=m.put(logical),addrGm=m.put(addresses);
 const uint32_t outputStride=r+8192;
 std::vector<uint64_t> output(uint64_t(batch)*outputStride*82+128,0x5a5a5a5a5a5a5a5aULL);
 for(uint32_t b=0;b<batch;++b)for(uint32_t t=0;t<r;++t)for(uint32_t f=0;f<82;++f)output[(uint64_t(b)*outputStride+t)*82+f]=ExpectedPayload(b,t,f);
 auto outputGm=m.put(output);std::vector<uint64_t>().swap(output);
 uint32_t views=2*batch+8*options.workers;
 auto sq=m.put(std::vector<uint8_t>(uint64_t(views)*4096*64)),cq=m.put(std::vector<uint32_t>(views*64*16,4)),db=m.put(std::vector<uint32_t>(views*64));
 std::vector<KvafBatchGetDirectUrmaEntity> entities(views);std::vector<KvafBatchGetQueueState> queues(views);
 std::vector<KvafBatchGetCoreStatus> statuses(views);
 for(uint32_t i=0;i<views;++i){auto& e=entities[i];e.magic=0x4b564155;e.version=1;e.sq_depth=4096;e.cq_depth=64;
  e.sq_va=sq+uint64_t(i)*4096*64;e.sq_db_va=db+i*256;e.cq_va=cq+i*64*64;e.cq_db_va=db+i*256+128;e.wqe_size=e.cqe_size=64;
  e.local_token_id=e.remote_token_id=1;e.local_hbm_address=outputGm;e.local_hbm_size=uint64_t(batch)*outputStride*656;
  e.remote_host_ddr_address=payloadGm;e.remote_host_ddr_size=uint64_t(batch)*remoteStride*656;
  queues[i].magic=0x4b564151;queues[i].initialized=1;statuses[i].magic=KVAF_AIV_BATCH_GET_STATUS_MAGIC;}
 auto eg=m.put(entities),qg=m.put(queues),sg=m.put(statuses);
 auto blocks=m.put(std::vector<int32_t>(batch*8192)),offsets=m.put(std::vector<int32_t>(batch*8192)),dest=m.put(std::vector<int32_t>(batch*8192));
 auto count=m.put(std::vector<uint32_t>(1));auto fallback=m.put(std::vector<uint64_t>(batch*8192));auto fallbackValid=m.put(std::vector<uint8_t>(batch*8192,1));
 std::vector<KvafExecuteBatchGetKernelArgs> args(views);
 for(uint32_t i=0;i<views;++i){auto& a=args[i];a.address_table=addrGm;a.valid_table=fallbackValid;a.indices=blocks;a.offsets=offsets;a.output_indices=dest;
  a.miss_count=count;a.output=outputGm;a.core_status=sg+i*sizeof(KvafBatchGetCoreStatus);a.channel_entity=eg+i*sizeof(KvafBatchGetDirectUrmaEntity);
  a.queue_state=qg+i*sizeof(KvafBatchGetQueueState);a.table_capacity=batch*h;a.request_capacity=batch*8192;a.output_rows=batch*outputStride;
  a.token_bytes=656;a.tokens_per_block=128;a.block_count=1;a.jettys_per_core=1;}
 Context ctx{};ctx.shape=s;ctx.rows=m.put(std::vector<Row>(batch));ctx.args=m.put(args);ctx.progress=m.put(std::vector<Progress>(batch*4));
 ctx.events=m.put(std::vector<Event>(s.rounds*s.serviceLanes));ctx.owners=m.put(std::vector<uint8_t>(s.finalOwnerBytes));
 ctx.mapping=m.put(std::vector<uint32_t>(uint64_t(batch)*h,UINT32_MAX));ctx.locations=m.put(std::vector<uint32_t>(batch*8192+128,0x5a5a5a5a));
 ctx.selected=m.put(std::vector<uint32_t>(batch*8192+128,0x5a5a5a5a));ctx.rowStatus=m.put(std::vector<uint32_t>(batch*32));ctx.finalStatus=m.put(std::vector<KvafBatchGetCoreStatus>(batch*2));
 ctx.sourceBlocks=blocks;ctx.sourceOffsets=offsets;ctx.destinations=dest;ctx.count=count;ctx.fallbackAddresses=fallback;ctx.fallbackValid=fallbackValid;
 ctx.tokenAddresses=addrGm;ctx.addressStride=h;ctx.outputStride=outputStride;ctx.hotValid=m.put(std::vector<uint32_t>(uint64_t(batch)*r,1));
 ctx.queryEnds=m.put(std::vector<uint32_t>(batch*4,h));ctx.prefetchLimit=0;ctx.windowLimit=448;
 std::vector<uint32_t> lru(batch*2048),slots(batch*10240,UINT32_MAX),changes(batch*10240,UINT32_MAX);
 auto lruSeed=ReadFixture<uint32_t>("lru.bin");if(lruSeed.empty()){lruSeed.resize(2048);std::iota(lruSeed.begin(),lruSeed.end(),r-2048);}
 std::set<uint32_t> reservedTokens(lruSeed.begin(),lruSeed.end()),repair=cold;
 require(lruSeed.size()==2048 && reservedTokens.size()==2048 && *reservedTokens.rbegin()<r,"LRU fixture");
 // With prefetch0 and no tail mappings, selected reserved hot tokens also
 // require repair under the existing cache contract; cold count is separate.
 for(auto t:expected)if(reservedTokens.count(t))repair.insert(t);
 std::vector<int32_t> reserved(uint64_t(batch)*r,-1);std::vector<uint32_t> owners(batch);std::iota(owners.begin(),owners.end(),0);
 for(uint32_t b=0;b<batch;++b)for(uint32_t i=0;i<2048;++i){uint32_t t=lruSeed[i];lru[b*2048+i]=slots[b*10240+i]=t;reserved[b*r+t]=i;}
 ctx.lruSlots=m.put(lru);ctx.reservedHot=m.put(reserved);ctx.changeSlots=m.put(slots);ctx.changes=m.put(changes);ctx.changeCount=m.put(std::vector<uint32_t>(batch,2048));ctx.rowOwners=m.put(owners);
 auto ctxGm=m.put(std::vector<Context>{ctx});
 std::vector<uint64_t> update(28);update[0]=keysGm;update[1]=scalesGm;update[2]=addrGm;update[3]=logicalGm;
 update[4]=ctx.changeSlots;update[5]=ctx.changes;update[6]=m.put(std::vector<uint32_t>(batch));update[7]=h;update[8]=ctx.rows;update[9]=ctx.rowOwners;
 update[10]=ctx.rowStatus;update[11]=ctx.owners+s.finalPostOffset;update[12]=sg;update[13]=ctx.finalStatus;update[14]=4;update[15]=sizeof(KvafBatchGetCoreStatus);
 update[16]=ctx.hotValid;update[17]=m.put(std::vector<uint32_t>(batch*(8+2*20480),0x5a5a5a5a));update[18]=ctx.selected;update[19]=ctx.locations;
 std::vector<int64_t> unpack(batch*4);std::iota(unpack.begin(),unpack.end(),0);update[20]=m.put(unpack);update[21]=m.put(std::vector<int64_t>(batch*4,1));
 update[22]=m.put(std::vector<uint32_t>(batch*8192+128,0x5a5a5a5a));update[23]=m.put(std::vector<uint32_t>(batch*8192+128,0x5a5a5a5a));
 update[24]=h;update[25]=r;update[26]=ctx.changeCount;update[27]=m.put(std::vector<uint32_t>(batch*40960));auto updateGm=m.put(update);
 pipeline_ab::Args tail{};tail.context=ctxGm;tail.keys=keysGm;tail.scales=scalesGm;tail.addresses=addrGm;tail.logical=logicalGm;tail.valid=ctx.hotValid;
 tail.flags=m.put(std::vector<pipeline_ab::BatchFlag>(batch*16));tail.eventsSrc=m.put(std::vector<uint32_t>(batch*10240));tail.eventsDst=m.put(std::vector<uint32_t>(batch*10240));
 tail.workerStatus=m.put(std::vector<uint32_t>(batch*32));tail.times=m.put(std::vector<uint64_t>(batch*256+(3*batch+4*options.workers)*32));
 tail.history=h;tail.resident=r;tail.prefetchCount=(options.workers<<16)|0x10000000U;tail.maxEvents=10240;
 tail.queryBitmaps=m.put(std::vector<uint32_t>(batch*4*((r+31)/32)));tail.queryErrors=m.put(std::vector<uint32_t>(batch*4*32));tail.updateBindings=updateGm;
 tail.denseShadow=m.put(std::vector<uint64_t>(uint64_t(batch)*h*3));tail.packedFetch=m.put(std::vector<int32_t>(4*options.workers*(9*batch*2048+((3*batch+31)&~31U)+64)));auto workspace=m.put(std::vector<pipeline_ab::Args>{tail});
 std::vector<uint64_t> binding(16);binding[1]=keysGm;binding[4]=scalesGm;binding[9]=ctx.selected;binding[14]=ctxGm;binding[15]=workspace;
 std::vector<uint8_t> liveQ(uint64_t(batch)*qseed.size());std::vector<uint16_t> liveW(uint64_t(batch)*wseed.size());std::vector<float> liveZ(uint64_t(batch)*zseed.size());
 for(uint32_t b=0;b<batch;++b){std::copy(qseed.begin(),qseed.end(),liveQ.begin()+uint64_t(b)*qseed.size());std::copy(wseed.begin(),wseed.end(),liveW.begin()+uint64_t(b)*wseed.size());std::copy(zseed.begin(),zseed.end(),liveZ.begin()+uint64_t(b)*zseed.size());}
 binding[0]=m.put(liveQ);binding[2]=m.put(liveW);binding[3]=m.put(liveZ);
 binding[5]=m.put(std::vector<uint32_t>(batch,4));binding[6]=m.put(std::vector<uint32_t>(batch,h));
 binding[12]=m.put(std::vector<uint8_t>(std::max<uint64_t>(16ULL*1024*1024,uint64_t(s.liCores)*4*h*4+1024*1024)));
 std::vector<int32_t> livePages(batch*h/128);std::iota(livePages.begin(),livePages.end(),0);binding[7]=m.put(livePages);
 binding[13]=m.put(std::vector<uint8_t>(uint64_t(batch)*(4*2048*6+1024+uint64_t(h)*2)));ctx.hits=binding[13]+uint64_t(batch)*(4*2048*6+1024);
 m.set(ctxGm,std::vector<Context>{ctx});auto bindings=m.put(binding);
 auto canonical=m.put(std::vector<uint16_t>(uint64_t(batch)*4*h+128,0x5a5a));
 std::vector<kvip_staged::ScoreWave> hot,coldWaves;
 auto makeWaves=[&](uint32_t begin,uint32_t end,uint32_t requested,std::vector<kvip_staged::ScoreWave>& waves){
  uint32_t cap=std::min(c,v/2),qsize=2*batch<=cap?4:2,p=qsize==4?std::max(1U,std::min(cap/batch,(end-begin)/128)):1;
  if(forced=="bsq"){qsize=2;p=1;}else if(forced=="history"){qsize=4;p=std::max(1U,std::min(cap/batch,(end-begin)/128));}
  if(requested){require(qsize==4 && requested<=cap && requested<=(end-begin)/128,"invalid explicit history partitions");p=requested;}
  std::vector<kvip_staged::ScoreTask> tasks;
  for(uint32_t b=0;b<batch;++b)for(uint32_t q=0;q<4;q+=qsize)for(uint32_t i=0;i<p;++i)tasks.push_back({b,q,begin+((end-begin)/128*i/p)*128,begin+((end-begin)/128*(i+1)/p)*128});
  for(uint32_t first=0;first<tasks.size();first+=cap){uint32_t rows=std::min(cap,uint32_t(tasks.size())-first),stride=0;
   std::vector<kvip_staged::ScoreTask> wave(tasks.begin()+first,tasks.begin()+first+rows);for(auto t:wave)stride=std::max(stride,t.end-t.begin);
   std::vector<uint8_t> qdata(uint64_t(rows)*qsize*32*128);std::vector<uint16_t> wdata(uint64_t(rows)*qsize*32);std::vector<float> zdata(wdata.size());
   std::vector<uint32_t> lengths(rows),pages(uint64_t(rows)*stride/128);
   for(uint32_t j=0;j<rows;++j){auto t=wave[j];lengths[j]=t.end-t.begin;
    std::copy_n(qseed.begin()+t.queryBegin*32*128,qsize*32*128,qdata.begin()+uint64_t(j)*qsize*32*128);
    std::copy_n(wseed.begin()+t.queryBegin*32,qsize*32,wdata.begin()+uint64_t(j)*qsize*32);
    std::copy_n(zseed.begin()+t.queryBegin*32,qsize*32,zdata.begin()+uint64_t(j)*qsize*32);
    for(uint32_t x=0;x<lengths[j]/128;++x)pages[uint64_t(j)*stride/128+x]=t.owner*h/128+t.begin/128+x;}
   uint64_t score=m.put(std::vector<uint16_t>(uint64_t(rows)*qsize*stride+128,0x5a5a));std::vector<uint64_t> wb(16);
   wb[0]=m.put(qdata);wb[1]=keysGm;wb[2]=m.put(wdata);wb[3]=m.put(zdata);wb[4]=scalesGm;wb[5]=m.put(std::vector<uint32_t>(rows,qsize));
   wb[6]=m.put(lengths);wb[7]=m.put(pages);wb[9]=m.put(std::vector<uint32_t>(uint64_t(rows)*qsize*2048));wb[12]=score;
   waves.push_back({m.put(wb),score,m.put(wave),rows,stride,qsize,first+rows==tasks.size()?end:0});
  }
 };
 if(forced!="legacy"){
  makeWaves(0,r,options.hot,hot);uint32_t begin=r;
  for(uint32_t epoch=1;epoch<s.rounds;++epoch){uint32_t end=r+((h-r)/128*epoch/(s.rounds-1))*128;makeWaves(begin,end,options.cold,coldWaves);begin=end;}
 }
 kvip_staged::Plan plan{canonical,hot.data(),coldWaves.data(),uint32_t(hot.size()),uint32_t(coldWaves.size()),v,options.workers};
 if(forced!="legacy"){
  std::vector<kvip_staged::PrepareWave> descriptors;
  for(auto list:{&hot,&coldWaves})for(auto w:*list)descriptors.push_back({w.bindings,w.tasks,w.rows,w.stride,w.queryGroup,0});
  plan.prepareWaves=m.put(descriptors);plan.prepareCount=descriptors.size();plan.postMode=options.post;plan.prepareMode=options.prepare;plan.serviceMode=options.service;
 }
 auto clock=m.put(std::vector<uint64_t>(2));
 auto start=std::chrono::steady_clock::now();check(aclrtlaunch_kvip_model_clock(1,stream,(void*)clock,0));
 if(forced=="legacy")check(kvip_lifetch_streaming_forward_v2((void*)bindings,(void*)updateGm,(void*)workspace,stream,&s,options.workers,v));
 else check(kvip_lifetch_staged_forward((void*)bindings,(void*)workspace,stream,&s,&plan));
 check(aclrtlaunch_kvip_model_clock(1,stream,(void*)clock,1));check(aclrtSynchronizeStream(stream));
 auto modelClock=m.get<uint64_t>(clock,2);
 {std::ofstream raw(out+"/clock.json");raw<<"{\"start\":"<<modelClock[0]<<",\"end\":"<<modelClock[1]<<"}\n";}
 require(modelClock[1]>modelClock[0],"invalid Model forward clock");
 double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
 auto scores=m.get<uint16_t>(canonical,uint64_t(batch)*4*h+128);if(forced!="legacy")for(size_t i=0;i<uint64_t(batch)*4*h;++i)require(scores[i]==oracle[i%(4*h)],"score mismatch "+std::to_string(i));
 for(size_t i=uint64_t(batch)*4*h;i<scores.size();++i)require(scores[i]==0x5a5a,"score guard");
 auto prefix=m.get<uint16_t>(binding[13],uint64_t(batch)*(4*2048*3+512));
 for(uint32_t b=0;b<batch;++b)for(uint32_t q=0;q<4;++q){
  std::vector<uint16_t> row(oracle.begin()+q*h,oracle.begin()+q*h+r);std::nth_element(row.begin(),row.end()-2048,row.end());
  require(prefix[uint64_t(b)*(4*2048*3+512)+4*2048*3+q*128]==row[r-2048],"exact global resident threshold");
 }
 // With prefetch0, Final correctness alone cannot validate every unused Cold
 // predicate. Validate packed bytes independently, outside clock probes.
 if(forced!="legacy"){
  auto bitmap=m.get<uint8_t>(ctx.hits,uint64_t(batch)*h);
  for(uint32_t b=0;b<batch;++b)for(uint32_t q=0;q<4;++q){
   auto theta=prefix[uint64_t(b)*(4*2048*3+512)+4*2048*3+q*128];
   for(uint32_t t=r;t<h;t+=4){uint8_t want=0;
    for(uint32_t i=0;i<4;++i)if(oracle[uint64_t(q)*h+t+i]>theta)want|=1U<<(2*i);
    require(bitmap[(uint64_t(b)*4+q)*(h/4)+t/4]==want,"Cold bitmap byte "+std::to_string(t/4));
   }
  }
 }
 auto selected=m.get<uint32_t>(ctx.selected,batch*8192+128),loc=m.get<uint32_t>(ctx.locations,batch*8192+128),outLogical=m.get<uint32_t>(update[22],batch*8192+128),outSlots=m.get<uint32_t>(update[23],batch*8192+128);
 auto updated=m.get<uint64_t>(logicalGm,uint64_t(batch)*h),updatedAddr=m.get<uint64_t>(addrGm,uint64_t(batch)*h);
 auto states=m.get<uint32_t>(ctx.rowStatus,batch*32),workerStates=m.get<uint32_t>(tail.workerStatus,batch*32),updateStates=m.get<uint32_t>(update[6],batch);
 auto flags=m.get<pipeline_ab::BatchFlag>(tail.flags,batch*16);auto posts=m.get<FinalPost>(ctx.owners+s.finalPostOffset,batch*4);
 auto keyAfter=m.get<uint8_t>(keysGm,keys.size());auto scaleAfter=m.get<float>(scalesGm,scales.size());
 // Preserve the observed boundary on failure, before any assertion unwinds ACL.
 std::ofstream diagnostic(out+"/diagnostic.json");diagnostic<<"{\"expected_unique_cold\":"<<cold.size()<<",\"expected_unique_repair\":"<<repair.size()<<",\"owners\":[";
 for(uint32_t b=0;b<batch;++b){if(b)diagnostic<<",";std::set<uint32_t> actualCold;
  diagnostic<<"{\"owner\":"<<b<<",\"row_count\":"<<states[b*32+1]<<",\"worker_count\":"<<workerStates[b*32+1]<<",\"final_sets\":[";
  for(uint32_t q=0;q<4;++q){if(q)diagnostic<<",";auto begin=selected.begin()+b*8192+q*2048;
   std::set<uint32_t> got(begin,begin+2048),want(expected.begin()+q*2048,expected.begin()+(q+1)*2048);
   uint32_t missing=0,extra=0;for(auto t:want)missing+=!got.count(t);for(auto t:got){extra+=!want.count(t);if(t>=r)actualCold.insert(t);}
   diagnostic<<"{\"query\":"<<q<<",\"size\":"<<got.size()<<",\"missing\":"<<missing<<",\"extra\":"<<extra<<"}";
  }diagnostic<<"],\"actual_unique_cold\":"<<actualCold.size()<<"}";
 }diagnostic<<"]}\n";diagnostic.close();require(bool(diagnostic),"diagnostic write");
 for(uint32_t b=0;b<batch;++b){require(!states[b*32] && !workerStates[b*32] && !updateStates[b],"tail status b"+std::to_string(b));
  require(states[b*32+1]==repair.size() && workerStates[b*32+1]==repair.size(),"dedup repair count expected="+std::to_string(repair.size())+" row="+std::to_string(states[b*32+1])+" worker="+std::to_string(workerStates[b*32+1]));
  for(uint32_t q=0;q<4;++q){auto begin=selected.begin()+b*8192+q*2048;std::set<uint32_t> got(begin,begin+2048),want(expected.begin()+q*2048,expected.begin()+(q+1)*2048);require(got==want,"global Final set");
   require(flags[b*16+8+q].ready && !flags[b*16+8+q].error && !posts[b*4+q].error && !posts[b*4+q].deferred,"completion join");
   for(uint32_t i=0;i<2048;++i){uint32_t pos=b*8192+q*2048+i,t=selected[pos],slot=loc[pos];require(slot<r,"resolved hot slot");
    require(updated[uint64_t(b)*h+slot]==uint64_t(b)*h+t && outLogical[pos]==uint64_t(b)*h+t && outSlots[pos]==slot,"output stable mapping");}
  }
  auto hotPayload=m.get<uint64_t>(outputGm+uint64_t(b)*outputStride*656,uint64_t(r)*82);
  for(uint32_t t=0;t<h;++t){uint64_t original=updated[uint64_t(b)*h+t];require(original/h==b,"owner isolation");uint32_t old=original%h;
   require(updatedAddr[uint64_t(b)*h+t]==addresses[uint64_t(b)*h+old],"address permutation");
   require(scaleAfter[uint64_t(b)*h+t]==ksseed[old],"scale permutation");
   require(!std::memcmp(keyAfter.data()+(uint64_t(b)*h+t)*128,kseed.data()+uint64_t(old)*128,128),"key permutation");
   if(t<r)for(uint32_t f=0;f<82;++f)require(hotPayload[uint64_t(t)*82+f]==ExpectedPayload(b,old,f),"payload byte "+std::to_string(t));
  }
 }
 auto payloadGuard=m.get<uint64_t>(outputGm+uint64_t(batch)*outputStride*656,128);
 for(auto word:payloadGuard)require(word==0x5a5a5a5a5a5a5a5aULL,"payload allocation guard");
 for(uint32_t b=0;b<batch;++b){auto tailPayload=m.get<uint64_t>(outputGm+(uint64_t(b)*outputStride+r)*656,8192ULL*82);
  for(auto word:tailPayload)require(word==0x5a5a5a5a5a5a5a5aULL,"unused tail payload");}
 for(uint32_t i=0;i<32;++i)require(scaleAfter[uint64_t(batch)*h+i]==7.25f,"scale guard");
 for(auto ws:{&hot,&coldWaves})for(auto wave:*ws){auto raw=m.get<uint16_t>(wave.scores,uint64_t(wave.rows)*wave.queryGroup*wave.stride+128);
  for(uint32_t i=0;i<128;++i)require(raw[uint64_t(wave.rows)*wave.queryGroup*wave.stride+i]==0x5a5a,"wave score guard");}
 for(uint32_t i=0;i<128;++i){require(selected[batch*8192+i]==0x5a5a5a5a && loc[batch*8192+i]==0x5a5a5a5a && outLogical[batch*8192+i]==0x5a5a5a5a && outSlots[batch*8192+i]==0x5a5a5a5a,"output guard");require(keyAfter[uint64_t(batch)*h*128+i]==0x5a,"key guard");}
 std::ofstream report(out+"/correctness.json");report<<"{\"status\":\"PASSED\",\"producer\":\""<<SCORE_VERSION<<"\",\"batch\":"<<batch<<",\"history\":"<<h<<",\"resident\":"<<r<<",\"miss_per_query\":"<<miss
 <<",\"unique_cold\":"<<cold.size()<<",\"unique_reserved_hot_repair\":"<<repair.size()-cold.size()<<",\"unique_repair\":"<<repair.size()
 <<",\"score_keys\":"<<(forced=="legacy"?0:uint64_t(batch)*4*h)<<",\"owner_waves\":"<<(batch+kvip_staged::OwnerCapacity(v,options.workers)-1)/kvip_staged::OwnerCapacity(v,options.workers)
 <<",\"hot_score_waves\":"<<hot.size()<<",\"cold_score_waves\":"<<coldWaves.size()<<",\"score_route\":\""<<forced<<"\",\"prefetch_budget\":0,\"all_index_fields\":true,\"all_hot_payload\":true,\"guards\":true,\"completion\":true,\"physical_npu\":false,\"transport\":\"GM adapter\",\"host_wall_ms\":"<<wall
 <<",\"requested_windows\":"<<options.windows<<",\"effective_windows\":"<<s.rounds<<",\"post\":"<<options.post<<",\"prepare\":"<<options.prepare<<",\"service\":"<<options.service
 <<",\"hot_partitions\":"<<options.hot<<",\"cold_partitions\":"<<options.cold<<",\"fetch_workers\":"<<options.workers<<",\"forward_model_cycles\":"<<modelClock[1]-modelClock[0]
 <<",\"cold_bitmap_bytes_checked\":"<<(forced=="legacy"?0:uint64_t(batch)*(h-r))
 <<",\"model_clock_start\":"<<modelClock[0]<<",\"model_clock_end\":"<<modelClock[1]<<",\"clock_probe_kernels\":2,\"sequence_split\":"<<(rotating::UseSequenceSplit(s)?"true":"false")<<"}\n";
 require(bool(report),"report write");std::cout<<"STAGED_FULL_ORACLE_PASSED B="<<batch<<" miss="<<miss<<"\n";
}
int main(int argc,char** argv){try{
 if(argc==2 && std::string(argv[1])=="--capabilities"){std::cout<<"{\"scope\":\"production score and full tail with GM adapter\",\"max_batch\":32,\"queries\":4,\"producer\":\""<<SCORE_VERSION
 <<"\",\"score_controls\":true,\"legacy_control\":true,\"model_clock_probes\":2,\"prefetch_budget\":0}\n";return 0;}
 require(argc==11 || argc==18,"staged_routes FIXTURE OUTPUT B H R C V MISS ROUTE DEVICE [WINDOWS POST PREPARE SERVICE P_HOT P_COLD FETCH_WORKERS]");
 Options options;
 if(argc==18){options.windows=std::stoul(argv[11]);options.post=choice(argv[12],{"separate","scatter","direct"});options.prepare=choice(argv[13],{"wave","invocation","phase"});
  options.service=choice(argv[14],{"separate","inline"});options.hot=std::stoul(argv[15]);options.cold=std::stoul(argv[16]);options.workers=std::stoul(argv[17]);}
 require(options.windows>=2 && options.windows<=32 && options.hot<=32 && options.cold<=32 && (options.workers==1 || options.workers==2 || options.workers==4),"invalid options");
 require(!options.service || options.post,"inline ready needs scatter/direct");
 require(!std::filesystem::exists(argv[2]),"output must be new");std::filesystem::create_directories(argv[2]);setenv("KVIP_FIXTURE_DIR",argv[1],1);
 check(aclInit(nullptr));check(aclrtSetDevice(std::stoul(argv[10])));aclrtStream stream;check(aclrtCreateStream(&stream));
 Run(argv[2],std::stoul(argv[3]),std::stoul(argv[4]),std::stoul(argv[5]),std::stoul(argv[6]),std::stoul(argv[7]),std::stoul(argv[8]),stream,argv[9],options);
 check(aclrtDestroyStream(stream));check(aclrtResetDevice(std::stoul(argv[10])));check(aclFinalize());return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
