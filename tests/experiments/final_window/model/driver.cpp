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
template<class T> std::vector<T> get(uint64_t p,size_t n){std::vector<T> v(n);ck(aclrtMemcpy(v.data(),n*sizeof(T),(void*)p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
int main(int argc,char**argv){try{
 uint32_t mode=argc>1?std::stoul(argv[1]):0;constexpr uint32_t H=131328,R=16384,K=2048;
 ck(aclInit(nullptr));ck(aclrtSetDevice(0));aclrtStream stream;ck(aclrtCreateStream(&stream));
 rotating::Context c{};c.shape.batch=1;c.shape.history=H;c.shape.resident=R;c.shape.rounds=4;c.shape.serviceLanes=1;c.shape.tokenBytes=656;c.shape.postThreads=256;c.shape.postBank=240;c.shape.batchedRows=1;c.shape.scanTokens=23808;c.shape.indexHeads=32;c.shape.splitQ=1;c.shape.serviceAics=1;c.shape.liCores=2;
 c.prefetchLimit=2048;c.windowLimit=1024;c.prefetchRankStart=2048;c.outputStride=R+512;
 std::vector<uint32_t> selected(8192),loc(8192),packed(8192),err(128),meta(128),lru(2048);std::iota(lru.begin(),lru.end(),10000);
 uint32_t counts[4]={11,0,94,87},begins[4]={0,0,11,59};
 for(uint32_t q=0;q<4;++q){err[q*32+1]=counts[q]+1;meta[q*32+6]=counts[q];for(uint32_t i=0;i<K;++i){selected[q*K+i]=1000+i;loc[q*K+i]=1000+i;}
 for(uint32_t i=0;i<counts[q];++i){uint32_t id=R+begins[q]+i;selected[q*K+i]=id;loc[q*K+i]=mode?UINT32_MAX:id;packed[q*K+i]=(id<<11)|i;}}
 c.selected=put(selected);c.locations=put(loc);c.sourceBlocks=put(packed);c.sourceOffsets=put(std::vector<uint32_t>(8192));c.mapping=put(std::vector<uint32_t>(H,UINT32_MAX));c.lruSlots=put(lru);
 constexpr uint32_t prior=H-16384,prefix=4*K*6+1024;
 std::vector<uint16_t> score(4*H,1),bitmap(4*H/8,0);
 std::vector<uint8_t> inc(512+prefix+(H+255)/256*32+4096);
 for(uint32_t q=0;q<4;++q){
  for(uint32_t i=0;i<K;++i){uint16_t v=20000+i;uint32_t id=i|0x80000000U;score[q*H+i]=v;
   memcpy(inc.data()+512+q*K*2+i*2,&v,2);memcpy(inc.data()+512+4*K*2+q*K*4+i*4,&id,4);}
  for(uint32_t j=0;j<counts[q];++j){uint32_t i=prior+begins[q]+j;score[q*H+i]=30000+j;bitmap[q*H/8+i/8]|=uint16_t(1u<<((i%8)*2));}
 }
 auto scorep=put(score);
 for(uint32_t q=0;q<4;++q){uint64_t source=scorep+uint64_t(q)*H*2,cursor=uint64_t(prior)|(uint64_t(prior)<<32),cutoff=20000;
  memcpy(inc.data()+q*128,&source,8);memcpy(inc.data()+q*128+8,&cursor,8);memcpy(inc.data()+q*128+16,&cutoff,8);}
 c.incremental=put(inc);c.hits=put(bitmap);c.progress=put(std::vector<uint32_t>(128,H));c.queryEnds=put(std::vector<uint32_t>(4,H));
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
 a.flags=put(std::vector<pipeline_ab::BatchFlag>(16));
 a.eventsSrc=put(std::vector<uint32_t>(10240,0xdeadbeef));a.eventsDst=put(std::vector<uint32_t>(10240,0xdeadbeef));
 if(mode==3){
  size_t bad=0;
  for(uint32_t scenario=0;scenario<3;++scenario){
   std::vector<uint32_t> e(128);
   // Active empty, inactive padding, then active incomplete terminal service.
   if(scenario!=1)for(unsigned q=0;q<4;++q)e[q*32+1]=1;
   if(scenario==2){e[3]=5;e[6]=8;e[1]=6;}
   e[12]=0xdeadbeef;
   a.queryErrors=put(e);c.queryEnds=put(std::vector<uint32_t>(4,scenario==1?0:H));
   a.context=put(std::vector<rotating::Context>{c});
   ck(aclrtlaunch_test_service(1,stream,(void*)put(std::vector<pipeline_ab::Args>{a}),3));
   ck(aclrtSynchronizeStream(stream));
   auto f=get<pipeline_ab::BatchFlag>(a.flags,16);auto st=get<uint32_t>(c.rowStatus,32);
   auto out=get<uint32_t>(a.queryErrors,128);uint32_t expected=scenario==2?16:0;
   bad+=st[0]!=expected || out[12]!=(scenario==0?1U:0U);
   for(unsigned i:{0U,5U,12U,13U,14U,15U})bad+=f[i].ready!=1 || f[i].error!=expected;
  }
  printf("handoff boundaries bad=%zu\n",bad);fflush(stdout);
  ck(aclrtDestroyStream(stream));ck(aclrtResetDevice(0));ck(aclFinalize());return bad?1:0;
 }
 ck(aclrtlaunch_test_service(mode?4:1,stream,(void*)put(std::vector<pipeline_ab::Args>{a}),mode));ck(aclrtSynchronizeStream(stream));
 auto rows=get<rotating::Row>(c.rows,1);auto mapping=get<uint32_t>(c.mapping,H);auto slots=get<uint32_t>(c.locations,8192);auto wq=get<uint32_t>(sq,8192*16);auto t=get<uint64_t>(a.times,16);auto db=get<uint64_t>(doorbell,1);
 size_t bad=0;std::vector<bool> seen(146);
 for(uint32_t i=0;i<146;++i){uint32_t v=mapping[prior+i];if(v<R+84||v>=R+230){bad++;continue;}seen[v-R-84]=true;}
 for(bool v:seen)bad+=!v;
 for(uint32_t i=0;i<146;++i){uint64_t remote=uint64_t(wq[i*16+10])|(uint64_t(wq[i*16+11])<<32);uint32_t token=(remote-0x100000000ULL)/656;if(token<prior||token>=prior+146){bad++;continue;}uint64_t local=uint64_t(wq[i*16+14])|(uint64_t(wq[i*16+15])<<32);if(mapping[token]<R||mapping[token]>=R+2048){bad++;continue;}bad+=local!=args.output+uint64_t(lru[mapping[token]-R])*656;}
 auto prefixout=get<uint8_t>(c.incremental,inc.size());
 if(mode)selected=get<uint32_t>(c.selected,8192);
 else for(uint32_t q=0;q<4;++q)for(uint32_t i=0;i<K;++i){memcpy(&selected[q*K+i],prefixout.data()+512+4*K*2+q*K*4+i*4,4);selected[q*K+i]&=0x3fffffffU;}
 for(uint32_t q=0;q<4;++q){std::vector<uint32_t> want;for(uint32_t i=counts[q];i<K;++i)want.push_back(i);for(uint32_t i=0;i<counts[q];++i)want.push_back(prior+begins[q]+i);
  std::vector<uint32_t> got(selected.begin()+q*K,selected.begin()+(q+1)*K);std::sort(want.begin(),want.end());std::sort(got.begin(),got.end());bad+=got!=want;
  if(mode)for(uint32_t i=0;i<K;++i){uint32_t id=selected[q*K+i];if(id>=H){bad++;continue;}if(id>=R&&(mapping[id]<R||mapping[id]>=R+2048)){bad++;continue;}uint32_t dest=id<R?id:lru[mapping[id]-R];bad+=slots[q*K+i]!=dest;}
 }
 bad+=rows[0].requests!=230 || db[0]!=146;
 if(mode==2){
  auto flags=get<pipeline_ab::BatchFlag>(a.flags,16);auto st=get<uint32_t>(c.rowStatus,32);
  auto es=get<uint32_t>(a.eventsSrc,10240),ed=get<uint32_t>(a.eventsDst,10240),changes=get<uint32_t>(c.changes,10240),errors=get<uint32_t>(a.queryErrors,128);
  bad+=errors[12]!=1 || flags[0].ready!=1 || flags[0].count!=230 || flags[0].error || flags[5].ready!=1;
  for(unsigned q=0;q<4;q++)bad+=flags[12+q].ready!=1 || flags[12+q].count || flags[12+q].error || flags[12+q].reserved[0];
  for(unsigned i=0;i<230;i++)bad+=es[i]!=changes[i] || ed[i]!=lru[i];
  for(unsigned i=230;i<2048;i++)bad+=changes[i]!=UINT32_MAX;
  for(auto x:st)bad+=x!=0;
  bad+=rows[0].finalRequests || rows[0].finalPending;
 }
 printf("mode=%u posted=%u bad=%zu\n",mode,rows[0].requests-84,bad);fflush(stdout);
 ck(aclrtDestroyStream(stream));ck(aclrtResetDevice(0));ck(aclFinalize());return bad?1:0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 2;}}
