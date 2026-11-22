#include <acl/acl.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>
#include <stdexcept>
#include "context.h"
#include "pipeline_types.h"
extern "C" uint32_t aclrtlaunch_test_service(uint32_t,void*,void*,uint32_t);
void ck(int v){if(v)throw std::runtime_error("ACL "+std::to_string(v));}
template<class T> uint64_t put(const std::vector<T>& v){void*p;ck(aclrtMalloc(&p,v.size()*sizeof(T),ACL_MEM_MALLOC_HUGE_FIRST));ck(aclrtMemcpy(p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));return(uint64_t)p;}
template<class T> void restore(uint64_t p,const std::vector<T>&v){ck(aclrtMemcpy((void*)p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));}
template<class T> std::vector<T> get(uint64_t p,size_t n){std::vector<T> v(n);ck(aclrtMemcpy(v.data(),n*sizeof(T),(void*)p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
int main(int argc,char**argv){try{
 uint32_t mode=argc>1?std::stoul(argv[1]):0;constexpr uint32_t H=131328,R=16384,K=2048;
 ck(aclInit(nullptr));ck(aclrtSetDevice(0));aclrtStream stream;ck(aclrtCreateStream(&stream));
 rotating::Context c{};c.shape.batch=1;c.shape.history=H;c.shape.resident=R;c.shape.rounds=4;c.shape.serviceLanes=1;c.shape.tokenBytes=656;c.shape.postThreads=256;c.shape.postBank=240;c.shape.batchedRows=0;c.shape.scanTokens=23808;c.shape.indexHeads=32;c.shape.splitQ=1;c.shape.serviceAics=1;c.shape.liCores=2;
 c.prefetchLimit=2048;c.windowLimit=mode?8192:1024;c.prefetchRankStart=2048;c.outputStride=R+512;
 std::vector<uint32_t> selected(8192),loc(8192),packed(8192),err(128),meta(128),lru(2048);std::iota(lru.begin(),lru.end(),10000);
 uint32_t counts[4]={11,0,94,87},begins[4]={0,0,11,59};
 for(uint32_t q=0;q<4;++q){err[q*32+1]=counts[q]+1;meta[q*32+6]=counts[q];for(uint32_t i=0;i<K;++i){selected[q*K+i]=1000+i;loc[q*K+i]=1000+i;}
 for(uint32_t i=0;i<counts[q];++i){uint32_t id=R+begins[q]+i;selected[q*K+i]=id;loc[q*K+i]=mode?UINT32_MAX:id;packed[q*K+i]=(id<<11)|i;}}
 c.selected=put(selected);c.locations=put(loc);c.sourceBlocks=put(packed);c.sourceOffsets=put(std::vector<uint32_t>(8192));c.mapping=put(std::vector<uint32_t>(H,UINT32_MAX));c.lruSlots=put(lru);
 constexpr uint32_t prior=H-16384,prefix=4*K*6+1024;
 std::vector<uint16_t> score(4*H,1),bitmap(4*H/8,0);
 std::vector<uint8_t> inc(512+prefix+(H+255)/256*32+4096);
 for(uint32_t q=0;q<4;++q){
  for(uint32_t i=0;i<K;++i){uint16_t v=20000+i;uint32_t id=(mode==2 && i==100)?((H-512)|0x40000000U):(i|0x80000000U);score[q*H+i]=v;
   memcpy(inc.data()+512+q*K*2+i*2,&v,2);memcpy(inc.data()+512+4*K*2+q*K*4+i*4,&id,4);}
  for(uint32_t j=0;j<counts[q];++j){uint32_t i=prior+begins[q]+j;score[q*H+i]=30000+j;bitmap[q*H/8+i/8]|=uint16_t(1u<<((i%8)*2));}
 }
 if(mode==2)for(uint32_t q=0;q<4;++q){score[q*H+H-1]=65500;bitmap[q*H/8+(H-1)/8]|=uint16_t(1u<<(((H-1)%8)*2));}
 auto scorep=put(score);
 for(uint32_t q=0;q<4;++q){uint64_t source=scorep+uint64_t(q)*H*2,cursor=uint64_t(prior)|(uint64_t(prior)<<32),cutoff=20000;
  memcpy(inc.data()+q*128,&source,8);memcpy(inc.data()+q*128+8,&cursor,8);memcpy(inc.data()+q*128+16,&cutoff,8);}
 c.incremental=put(inc);c.hits=put(bitmap);c.progress=put(std::vector<uint32_t>(128,H));c.queryEnds=put(std::vector<uint32_t>(4,H));
 if(mode==2){c.queryEnds=put(std::vector<uint32_t>{H-1,H-2,H-3,H-4});c.tailBounds=put(std::vector<uint32_t>{H-512,H-256});}
 c.hotValid=put(std::vector<uint32_t>(R,1));c.reservedHot=put(std::vector<uint32_t>(R,UINT32_MAX));

 c.changes=put(std::vector<uint32_t>(10240,UINT32_MAX));c.changeSlots=put(std::vector<uint32_t>(10240,UINT32_MAX));std::vector<rotating::Event> events(5);for(auto&e:events)e.complete=1;c.events=put(events);
 auto sq=put(std::vector<uint32_t>(8192*16));auto doorbell=put(std::vector<uint64_t>(16));auto queue=put(std::vector<uint64_t>(8));
 std::vector<uint64_t> entity(24);entity[7]=sq;entity[8]=doorbell;((uint32_t*)entity.data())[19]=8192;
 std::vector<uint64_t> addresses(H);for(uint32_t i=0;i<H;++i)addresses[i]=0x100000000ULL+uint64_t(i)*656;
 rotating::Row row{};row.slots=84;row.requests=84;row.addresses=put(addresses);c.rows=put(std::vector<rotating::Row>{row});
 KvafExecuteBatchGetKernelArgs args{};args.channel_entity=put(entity);args.queue_state=queue;args.output=0x120000000ULL;c.args=put(std::vector<KvafExecuteBatchGetKernelArgs>(2,args));
 pipeline_ab::Args a{};a.context=put(std::vector<rotating::Context>{c});a.queryErrors=put(std::vector<uint32_t>(128));a.queryBitmaps=put(std::vector<uint32_t>(2048));a.times=put(std::vector<uint64_t>(256));
 c.rowStatus=put(std::vector<uint32_t>(32,0xdeadbeef));
 a.context=put(std::vector<rotating::Context>{c});a.maxEvents=10240;
 a.prefetchCount=0x80000000U;
 a.flags=put(std::vector<pipeline_ab::BatchFlag>(16));
 a.eventsSrc=put(std::vector<uint32_t>(10240,0xdeadbeef));a.eventsDst=put(std::vector<uint32_t>(10240,0xdeadbeef));

 std::vector<uint64_t> keys(H*16),logical(H);std::vector<uint32_t> scales(H);
 for(uint32_t i=0;i<H;++i){logical[i]=i*3+7;scales[i]=i;for(uint32_t j=0;j<16;++j)keys[i*16+j]=uint64_t(i)*16+j;}
 a.keys=put(keys);a.scales=put(scales);a.logical=put(logical);a.addresses=row.addresses;
 a.flags=put(std::vector<uint32_t>(32*32));a.eventsSrc=put(std::vector<uint32_t>(8192));a.prefetchCount=mode==4?0x80000000U:0;
 auto deviceArg=put(std::vector<pipeline_ab::Args>{a});
 for(uint32_t repeat=0;repeat<(mode==3?2U:1U);++repeat){
  if(repeat){
   auto retired=get<uint32_t>(c.mapping,H);for(uint32_t i=prior;i<prior+146;++i)if(retired[i]!=UINT32_MAX)throw std::runtime_error("mapping retirement failed");
   auto retiredMeta=get<uint64_t>(c.incremental,64);for(auto value:retiredMeta)if(value)throw std::runtime_error("TopK metadata retirement failed");
   restore(c.incremental,inc);restore(c.progress,std::vector<uint32_t>(128,H));restore(c.rows,std::vector<rotating::Row>{row});
   restore(a.keys,keys);restore(a.scales,scales);restore(a.logical,logical);restore(a.addresses,addresses);
   restore(queue,std::vector<uint64_t>(8));
  }
  ck(aclrtlaunch_test_service(4,stream,(void*)deviceArg,mode));ck(aclrtSynchronizeStream(stream));
 }
 auto v=get<uint32_t>(a.eventsSrc,128);int bad=0;for(int q=0;q<4;++q){printf("q%d:",q);for(int j=0;j<4;++j){printf(" %u",v[q*32+j]);bad+=v[q*32+j]!=j+1;}puts("");}printf("sync_bad=%d\n",bad);fflush(stdout);ck(aclrtDestroyStream(stream));ck(aclrtResetDevice(0));ck(aclFinalize());return bad?1:0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 2;}}
