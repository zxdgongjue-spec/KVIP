#include <acl/acl.h>
#include "context.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <set>
#include <stdexcept>
#include <string>
extern "C" uint32_t aclrtlaunch_window_model(uint32_t,void*,void*,void*);
static void ck(int n){if(n)throw std::runtime_error("ACL "+std::to_string(n));}
template<class T> void* upload(const std::vector<T>& v){void* p;ck(aclrtMalloc(&p,v.size()*sizeof(T),ACL_MEM_MALLOC_HUGE_FIRST));ck(aclrtMemcpy(p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));return p;}
template<class T> std::vector<T> read(void* p,size_t n){std::vector<T> v(n);ck(aclrtMemcpy(v.data(),v.size()*sizeof(T),p,v.size()*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
int main(){try{
 constexpr uint32_t B=2,Q=8,H=131072,R=16384,K=2048,N=16384;
 ck(aclInit(nullptr));ck(aclrtSetDevice(0));aclrtStream stream;ck(aclrtCreateStream(&stream));
 std::vector<uint16_t> raw(Q*H,0);std::vector<uint32_t> bitmap(Q*H/16,0),progress(Q*32,0),mapping(B*H,UINT32_MAX),lru(B*2048);
 std::vector<uint64_t> privateState((B*512+B*(4*K*6+1024))/8,0),addresses(B*H);
 std::vector<std::set<uint32_t>> wanted(B);
 for(uint32_t q=0;q<Q;++q){
   auto p=reinterpret_cast<uint8_t*>(privateState.data())+B*512+(q/4)*(4*K*6+1024);
   auto scores=reinterpret_cast<uint16_t*>(p)+(q%4)*K;auto ids=reinterpret_cast<uint32_t*>(p+4*K*2)+(q%4)*K;
   for(uint32_t i=0;i<K;++i){scores[i]=0xe000-i;ids[i]=i;}
   for(uint32_t i=0;i<N;++i)raw[q*H+R+i]=0xd000;
   for(uint32_t i=0;i<12;++i){uint32_t t=R+128+i*769+(q%2)*384;raw[q*H+t]=0xf000-i;bitmap[q*H/16+t/16]|=1U<<(2*(t%16));wanted[q/4].insert(t);}
   progress[q*32]=R+N;progress[q*32+16]=R+N;
 }
 for(uint32_t b=0;b<B;++b){for(uint32_t t=0;t<H;++t)addresses[b*H+t]=0x100000000ULL+(uint64_t(b)*H+t)*656;for(uint32_t i=0;i<2048;++i)lru[b*2048+i]=5000+i;uint32_t mapped=*wanted[b].begin();mapping[b*H+mapped]=R;wanted[b].erase(mapped);}
 void* rawD=upload(raw);void* bitmapD=upload(bitmap);void* progressD=upload(progress);void* mapD=upload(mapping);void* lruD=upload(lru);void* addrD=upload(addresses);
 void* locationsD=upload(std::vector<uint32_t>(B*8192,0));
 for(uint32_t q=0;q<Q;++q){privateState[q*16]=(uint64_t)rawD+q*H*2;privateState[q*16+1]=uint64_t(R)|(uint64_t(R)<<32);privateState[q*16+2]=0xd801;}
 void* privateD=upload(privateState);
 std::vector<rotating::Row> rows(B);std::vector<void*> sqs,queues,dbs,entities;
 for(uint32_t b=0;b<B;++b){
  void* sq=upload(std::vector<uint32_t>(8192*16,0));void* db=upload(std::vector<uint64_t>(8,0));void* queue=upload(std::vector<uint64_t>(8,0));std::vector<uint64_t> entity(24,0);entity[7]=(uint64_t)sq;entity[8]=(uint64_t)db;
  auto e=reinterpret_cast<uint32_t*>(entity.data());e[19]=8192;e[31]=8192;e[12]=1;e[20]=1;
  void* ent=upload(entity);sqs.push_back(sq);queues.push_back(queue);dbs.push_back(db);entities.push_back(ent);
  auto& r=rows[b];r.entity=(uint64_t)ent;r.queue=(uint64_t)queue;r.addresses=(uint64_t)addrD+b*H*8;r.output=0x200000000ULL+b*R*656ULL;r.slots=1;r.requests=1;r.limit=2048;r.prefetchLimit=512;r.queueLimit=8192;
  for(uint32_t q=0;q<4;++q){r.consumed[q]=R;r.snapshot[0][q]=R;}
 }
 void* rowsD=upload(rows);std::vector<rotating::Event> events(8);events[0].complete=1;void* eventsD=upload(events);
 rotating::Context ctx{};ctx.progress=(uint64_t)progressD;ctx.rows=(uint64_t)rowsD;ctx.mapping=(uint64_t)mapD;ctx.locations=(uint64_t)locationsD;ctx.events=(uint64_t)eventsD;ctx.hits=(uint64_t)bitmapD;ctx.lruSlots=(uint64_t)lruD;ctx.incremental=(uint64_t)privateD;ctx.windowLimit=64;
 ctx.shape.batch=B;ctx.shape.history=H;ctx.shape.resident=R;ctx.shape.serviceAics=1;ctx.shape.serviceLanes=1;ctx.shape.rounds=8;
 void* contextD=upload(std::vector<rotating::Context>{ctx});void* timingD=upload(std::vector<uint64_t>(8,0));
 ck(aclrtlaunch_window_model(1,stream,contextD,timingD));ck(aclrtSynchronizeStream(stream));
 auto times=read<uint64_t>(timingD,8);auto result=read<uint64_t>(privateD,privateState.size());
 for(uint32_t q=0;q<Q;++q){auto p=reinterpret_cast<uint8_t*>(result.data())+B*512+(q/4)*(4*K*6+1024);auto scores=reinterpret_cast<uint16_t*>(p)+(q%4)*K;auto ids=reinterpret_cast<uint32_t*>(p+4*K*2)+(q%4)*K;std::vector<uint16_t> expected;for(uint32_t i=0;i<K;++i)expected.push_back(0xe000-i);for(uint32_t i=0;i<N;++i)expected.push_back(raw[q*H+R+i]);std::sort(expected.begin(),expected.end(),std::greater<uint16_t>());expected.resize(K);std::vector<uint16_t> got(scores,scores+K);std::sort(got.begin(),got.end(),std::greater<uint16_t>());if(got!=expected)throw std::runtime_error("TopK scores q="+std::to_string(q));std::set<uint32_t> unique;for(uint32_t i=0;i<K;++i){uint32_t id=ids[i];uint16_t value=id<R?0xe000-id:raw[q*H+id];if(value!=scores[i]||!unique.insert(id).second)throw std::runtime_error("TopK IDs");}}
 auto rr=read<rotating::Row>(rowsD,B);auto ev=read<rotating::Event>(eventsD,8);
 for(uint32_t b=0;b<B;++b){auto sq=read<uint32_t>(sqs[b],8192*16);auto queue=read<uint64_t>(queues[b],8);auto db=read<uint64_t>(dbs[b],8);auto remaining=wanted[b];uint32_t count=rr[b].epochCount[1];if(rr[b].error||count!=remaining.size()||queue[1]!=count||db[0]!=count)throw std::runtime_error("post count/state b="+std::to_string(b)+" count="+std::to_string(count));for(uint32_t i=0;i<count;++i){uint64_t remote=uint64_t(sq[i*16+10])|(uint64_t(sq[i*16+11])<<32);uint64_t local=uint64_t(sq[i*16+14])|(uint64_t(sq[i*16+15])<<32);uint32_t token=(remote-0x100000000ULL)/656-b*H;if(!remaining.erase(token)||local!=rows[b].output+uint64_t(lru[b*2048+1+i])*656||sq[i*16+12]!=656)throw std::runtime_error("WQE payload b="+std::to_string(b));}printf("PASS b=%u TopK+dedup+mapping+slots+WQE+doorbell count=%u\n",b,count);}
 printf("CYCLES topk=%llu service=%llu total=%llu scan=%llu candidate=%llu post=%llu\n",(unsigned long long)(times[1]-times[0]),(unsigned long long)(times[2]-times[1]),(unsigned long long)(times[2]-times[0]),(unsigned long long)ev[1].scanTicks,(unsigned long long)ev[1].candidateTicks,(unsigned long long)ev[1].postTicks);
 // Same progress and TopK state: export must refresh compact IDs even when
 // AdvanceIncremental has no new score interval. No duplicate WQE is allowed.
 ck(aclrtlaunch_window_model(1,stream,contextD,timingD));ck(aclrtSynchronizeStream(stream));
 auto reused=read<rotating::Row>(rowsD,B);
 for(uint32_t b=0;b<B;++b){auto queue=read<uint64_t>(queues[b],8);if(reused[b].error||reused[b].epochCount[1]!=0||reused[b].requests!=rr[b].requests||queue[1]!=wanted[b].size())throw std::runtime_error("same-window replay posted duplicates");}
 printf("PASS unchanged progress: no duplicate fetch, slots and queue heads retained\n");
 ck(aclrtDestroyStream(stream));ck(aclrtResetDevice(0));ck(aclFinalize());return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
