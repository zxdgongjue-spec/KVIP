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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>
static void check(uint32_t c){if(c)throw std::runtime_error("ACL error "+std::to_string(c));}
static void require(bool c,const std::string& s){if(!c)throw std::runtime_error(s);}
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
static void Run(const std::string& out,uint32_t batch,uint32_t h,uint32_t r,uint32_t c,uint32_t v,uint32_t miss,aclrtStream stream,const std::string& forced){
 using namespace rotating;Memory m;
 RuntimeShape s{};s.batch=batch;s.history=h;s.resident=r;s.indexHeads=32;s.aicCount=c;s.tokenBytes=656;s.rounds=2;
 check(kvip_staged::PrepareShape(&s,v,2));
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
 uint32_t views=2*batch+16;
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
 tail.workerStatus=m.put(std::vector<uint32_t>(batch*32));tail.times=m.put(std::vector<uint64_t>(batch*256+(3*batch+8)*32));
 tail.history=h;tail.resident=r;tail.prefetchCount=(2<<16)|0x10000000U;tail.maxEvents=10240;
 tail.queryBitmaps=m.put(std::vector<uint32_t>(batch*4*((r+31)/32)));tail.queryErrors=m.put(std::vector<uint32_t>(batch*4*32));tail.updateBindings=updateGm;
 tail.denseShadow=m.put(std::vector<uint64_t>(uint64_t(batch)*h*3));tail.packedFetch=m.put(std::vector<int32_t>(8*(9*batch*2048+((3*batch+31)&~31U)+64)));auto workspace=m.put(std::vector<pipeline_ab::Args>{tail});
 std::vector<uint64_t> binding(16);binding[1]=keysGm;binding[4]=scalesGm;binding[9]=ctx.selected;binding[14]=ctxGm;binding[15]=workspace;
 std::vector<uint8_t> liveQ(uint64_t(batch)*qseed.size());std::vector<uint16_t> liveW(uint64_t(batch)*wseed.size());std::vector<float> liveZ(uint64_t(batch)*zseed.size());
 for(uint32_t b=0;b<batch;++b){std::copy(qseed.begin(),qseed.end(),liveQ.begin()+uint64_t(b)*qseed.size());std::copy(wseed.begin(),wseed.end(),liveW.begin()+uint64_t(b)*wseed.size());std::copy(zseed.begin(),zseed.end(),liveZ.begin()+uint64_t(b)*zseed.size());}
 binding[0]=m.put(liveQ);binding[2]=m.put(liveW);binding[3]=m.put(liveZ);
 std::vector<int32_t> livePages(batch*h/128);std::iota(livePages.begin(),livePages.end(),0);binding[7]=m.put(livePages);
 binding[13]=m.put(std::vector<uint8_t>(uint64_t(batch)*(4*2048*6+1024+uint64_t(h)*2)));ctx.hits=binding[13]+uint64_t(batch)*(4*2048*6+1024);
 m.set(ctxGm,std::vector<Context>{ctx});auto bindings=m.put(binding);
 auto canonical=m.put(std::vector<uint16_t>(uint64_t(batch)*4*h+128,0x5a5a));
 std::vector<kvip_staged::ScoreWave> hot,coldWaves;
 auto makeWaves=[&](uint32_t begin,uint32_t end,std::vector<kvip_staged::ScoreWave>& waves){
  uint32_t cap=std::min(c,v/2),qsize=2*batch<=cap?4:2,p=qsize==4?std::max(1U,std::min(cap/batch,(end-begin)/128)):1;
  if(forced=="bsq"){qsize=2;p=1;}else if(forced=="history"){qsize=4;p=std::max(1U,std::min(cap/batch,(end-begin)/128));}
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
 makeWaves(0,r,hot);makeWaves(r,h,coldWaves);kvip_staged::Plan plan{canonical,hot.data(),coldWaves.data(),uint32_t(hot.size()),uint32_t(coldWaves.size()),v,2};
 auto start=std::chrono::steady_clock::now();check(kvip_lifetch_staged_forward((void*)bindings,(void*)workspace,stream,&s,&plan));check(aclrtSynchronizeStream(stream));
 double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
 auto scores=m.get<uint16_t>(canonical,uint64_t(batch)*4*h+128);for(size_t i=0;i<uint64_t(batch)*4*h;++i)require(scores[i]==oracle[i%(4*h)],"score mismatch "+std::to_string(i));
 for(size_t i=uint64_t(batch)*4*h;i<scores.size();++i)require(scores[i]==0x5a5a,"score guard");
 auto prefix=m.get<uint16_t>(binding[13],uint64_t(batch)*(4*2048*3+512));
 for(uint32_t b=0;b<batch;++b)for(uint32_t q=0;q<4;++q){
  std::vector<uint16_t> row(oracle.begin()+q*h,oracle.begin()+q*h+r);std::nth_element(row.begin(),row.end()-2048,row.end());
  require(prefix[uint64_t(b)*(4*2048*3+512)+4*2048*3+q*128]==row[r-2048],"exact global resident threshold");
 }
 auto selected=m.get<uint32_t>(ctx.selected,batch*8192+128),loc=m.get<uint32_t>(ctx.locations,batch*8192+128),outLogical=m.get<uint32_t>(update[22],batch*8192+128),outSlots=m.get<uint32_t>(update[23],batch*8192+128);
 auto updated=m.get<uint64_t>(logicalGm,uint64_t(batch)*h),updatedAddr=m.get<uint64_t>(addrGm,uint64_t(batch)*h);
 auto states=m.get<uint32_t>(ctx.rowStatus,batch*32),workerStates=m.get<uint32_t>(tail.workerStatus,batch*32),updateStates=m.get<uint32_t>(update[6],batch);
 auto flags=m.get<pipeline_ab::BatchFlag>(tail.flags,batch*16);auto posts=m.get<FinalPost>(ctx.owners+s.finalPostOffset,batch*4);
 auto keyAfter=m.get<uint8_t>(keysGm,keys.size());auto scaleAfter=m.get<float>(scalesGm,scales.size());
 for(uint32_t b=0;b<batch;++b){require(!states[b*32] && !workerStates[b*32] && !updateStates[b],"tail status b"+std::to_string(b));
  require(states[b*32+1]==cold.size(),"dedup miss count");
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
 <<",\"score_keys\":"<<uint64_t(batch)*4*h<<",\"owner_waves\":"<<(batch+kvip_staged::OwnerCapacity(v,2)-1)/kvip_staged::OwnerCapacity(v,2)
 <<",\"hot_score_waves\":"<<hot.size()<<",\"cold_score_waves\":"<<coldWaves.size()<<",\"score_route\":\""<<forced<<"\",\"prefetch_budget\":0,\"all_index_fields\":true,\"all_hot_payload\":true,\"guards\":true,\"completion\":true,\"physical_npu\":false,\"transport\":\"GM adapter\",\"host_wall_ms\":"<<wall<<"}\n";
 require(bool(report),"report write");std::cout<<"STAGED_FULL_ORACLE_PASSED B="<<batch<<" miss="<<miss<<"\n";
}
int main(int argc,char** argv){try{
 if(argc==2 && std::string(argv[1])=="--capabilities"){std::cout<<"{\"scope\":\"staged full streaming with GM adapter\",\"max_batch\":32,\"queries\":4,\"producer\":\""<<SCORE_VERSION<<"\"}\n";return 0;}
 require(argc==11,"staged_routes FIXTURE OUTPUT B H R C V MISS ROUTE DEVICE");
 require(!std::filesystem::exists(argv[2]),"output must be new");std::filesystem::create_directories(argv[2]);setenv("KVIP_FIXTURE_DIR",argv[1],1);
 check(aclInit(nullptr));check(aclrtSetDevice(std::stoul(argv[10])));aclrtStream stream;check(aclrtCreateStream(&stream));
 Run(argv[2],std::stoul(argv[3]),std::stoul(argv[4]),std::stoul(argv[5]),std::stoul(argv[6]),std::stoul(argv[7]),std::stoul(argv[8]),stream,argv[9]);
 check(aclrtDestroyStream(stream));check(aclrtResetDevice(std::stoul(argv[10])));check(aclFinalize());return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
