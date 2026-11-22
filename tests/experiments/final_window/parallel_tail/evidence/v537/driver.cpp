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
 const uint32_t abcBegin=mode==11?H-333:H-512,abcCapacity=mode==11?384:512;
 if(mode==11)c.outputStride=R+abcCapacity;
 uint32_t counts[4]={11,0,94,87},begins[4]={0,0,11,59};if(mode==10)begins[2]=0;
 if(mode==16){const uint32_t nc[4]={94,87,0,11},nb[4]={11,59,0,0};std::copy(nc,nc+4,counts);std::copy(nb,nb+4,begins);}
 for(uint32_t q=0;q<4;++q){err[q*32+1]=counts[q]+1;meta[q*32+6]=counts[q];for(uint32_t i=0;i<K;++i){selected[q*K+i]=1000+i;loc[q*K+i]=1000+i;}
 for(uint32_t i=0;i<counts[q];++i){uint32_t id=R+begins[q]+i;selected[q*K+i]=id;loc[q*K+i]=mode?UINT32_MAX:id;packed[q*K+i]=(id<<11)|i;}}
 c.selected=put(selected);c.locations=put(loc);c.sourceBlocks=put(packed);c.sourceOffsets=put(std::vector<uint32_t>(8192));c.mapping=put(std::vector<uint32_t>(H,UINT32_MAX));c.lruSlots=put(lru);
 constexpr uint32_t prior=H-16384,prefix=4*K*6+1024;
 std::vector<uint16_t> score(4*H,1),bitmap(4*H/8,0);
 std::vector<uint8_t> inc(512+prefix+(H+255)/256*32+4096);
 for(uint32_t q=0;q<4;++q){
  for(uint32_t i=0;i<K;++i){uint16_t v=20000+i;uint32_t id=((mode==8 || mode==9) && i==1000)?(H-596+q):((mode==2 || mode==11) && i==100)?(abcBegin|0x40000000U):(i|0x80000000U);score[q*H+i]=v;
   memcpy(inc.data()+512+q*K*2+i*2,&v,2);memcpy(inc.data()+512+4*K*2+q*K*4+i*4,&id,4);}
  for(uint32_t j=0;j<counts[q];++j){uint32_t i=prior+begins[q]+j;score[q*H+i]=30000+j;bitmap[q*H/8+i/8]|=uint16_t(1u<<((i%8)*2));}
 }
 if((mode==2 || mode==11))for(uint32_t q=0;q<4;++q){score[q*H+H-1]=65500;bitmap[q*H/8+(H-1)/8]|=uint16_t(1u<<(((H-1)%8)*2));}
 auto scorep=put(score);
 for(uint32_t q=0;q<4;++q){uint64_t source=scorep+uint64_t(q)*H*2,cursor=uint64_t(prior)|(uint64_t(prior)<<32),cutoff=20000;
  memcpy(inc.data()+q*128,&source,8);memcpy(inc.data()+q*128+8,&cursor,8);memcpy(inc.data()+q*128+16,&cutoff,8);}
 c.incremental=put(inc);c.hits=put(bitmap);c.progress=put(std::vector<uint32_t>(128,H));c.queryEnds=put(std::vector<uint32_t>(4,H));
 if((mode==2 || mode==11)){c.queryEnds=put(std::vector<uint32_t>{H-1,H-2,H-3,H-4});c.tailBounds=put(std::vector<uint32_t>{abcBegin,abcBegin+(mode==11?123U:256U)});}
 c.hotValid=put(std::vector<uint32_t>(R,1));c.reservedHot=put(std::vector<uint32_t>(R,UINT32_MAX));

 rotating::MappingJournal journal{};std::fill(std::begin(journal.tokens),std::end(journal.tokens),UINT32_MAX);journal.count=84;
 std::vector<uint32_t> initialMapping(H,UINT32_MAX);
 for(uint32_t i=0;i<84;++i){journal.tokens[i]=H-596+i;initialMapping[journal.tokens[i]]=R+i;}
 restore(c.mapping,initialMapping);
 c.mappingJournal=put(std::vector<rotating::MappingJournal>{journal});
 c.changes=put(std::vector<uint32_t>(10240,UINT32_MAX));c.changeSlots=put(std::vector<uint32_t>(10240,UINT32_MAX));std::vector<rotating::Event> events(5);for(auto&e:events)e.complete=1;c.events=put(events);
 auto sq=put(std::vector<uint32_t>(4*8192*16));auto doorbell=put(std::vector<uint64_t>(64));auto queue=put(std::vector<uint64_t>(32));
 std::vector<uint64_t> entity(96);for(uint32_t q=0;q<4;++q){entity[q*24+7]=sq+uint64_t(q)*8192*64;entity[q*24+8]=doorbell+q*128;((uint32_t*)entity.data())[q*48+19]=8192;}
 std::vector<uint64_t> initialHeads(4),queueState(32);
 if(mode==12 || mode==15){
  const uint64_t offsets[4]={8190,8191,16370,32760};
  for(uint32_t q=0;q<4;++q){
   initialHeads[q]=(uint64_t(1)<<32)+offsets[q];queueState[q*8+1]=initialHeads[q];
   auto e=reinterpret_cast<uint32_t*>(entity.data())+q*48;
   e[20]=0x123400+q;e[6]=0x45670+q;e[7]=0x781+q;e[12]=0x400+q;
   for(uint32_t j=0;j<4;++j)e[22+j]=0xabc000+q*16+j;
  }
  restore(queue,queueState);
 }
 std::vector<uint64_t> addresses(H);for(uint32_t i=0;i<H;++i)addresses[i]=0x100000000ULL+uint64_t(i)*656;
 rotating::Row row{};row.slots=84;row.requests=84;row.addresses=put(addresses);c.rows=put(std::vector<rotating::Row>{row});
 KvafExecuteBatchGetKernelArgs args{};args.channel_entity=put(entity);args.queue_state=queue;args.output=0x120000000ULL;std::vector<KvafExecuteBatchGetKernelArgs> perQArgs(4,args);for(uint32_t q=0;q<4;++q){perQArgs[q].channel_entity+=q*24*8;perQArgs[q].queue_state+=q*64;}c.args=put(perQArgs);
 pipeline_ab::Args a{};a.context=put(std::vector<rotating::Context>{c});a.queryErrors=put(std::vector<uint32_t>(128));a.queryBitmaps=put(std::vector<uint32_t>(2048));a.times=put(std::vector<uint64_t>(256));
 c.rowStatus=put(std::vector<uint32_t>(32,0xdeadbeef));
 a.context=put(std::vector<rotating::Context>{c});a.maxEvents=10240;
 a.prefetchCount=0x80000000U;
 a.flags=put(std::vector<pipeline_ab::BatchFlag>(16));
 a.eventsSrc=put(std::vector<uint32_t>(10240,0xdeadbeef));a.eventsDst=put(std::vector<uint32_t>(10240,0xdeadbeef));

 std::vector<uint64_t> keys(H*16),logical(H);std::vector<uint32_t> scales(H);
 for(uint32_t i=0;i<H;++i){logical[i]=((mode==2 || mode==11)?(uint64_t(i)<<32):0ULL)|(i*3+7);scales[i]=i;for(uint32_t j=0;j<16;++j)keys[i*16+j]=uint64_t(i)*16+j;}
 if((mode==8 || mode==9)){
  for(uint32_t i=0;i<84;++i){uint32_t token=journal.tokens[i],slot=lru[i];
   std::swap(logical[token],logical[slot]);std::swap(scales[token],scales[slot]);std::swap(addresses[token],addresses[slot]);
   for(uint32_t j=0;j<16;++j)std::swap(keys[token*16+j],keys[slot*16+j]);
  }
  restore(row.addresses,addresses);
 }
 a.keys=put(keys);a.scales=put(scales);a.logical=put(logical);a.addresses=row.addresses;
 a.flags=put(std::vector<uint32_t>(32*32));a.eventsSrc=put(std::vector<uint32_t>(8192));a.prefetchCount=mode==4?0x80000000U:0;
 // Lease-scoped immutable WQE fields, refreshed with every output/entity binding.
 std::vector<uint32_t> templates(4*32);
 for(uint32_t q=0;q<4;++q){auto w=templates.data()+q*32;auto e=reinterpret_cast<const uint32_t*>(entity.data())+q*48;
  w[1]=6U<<8;w[2]=(e[20]&0xffffffU)|0x1000000U;w[3]=e[6]&0xfffffU;
  for(uint32_t j=0;j<4;++j)w[4+j]=e[22+j];w[8]=e[7];w[9]=0;
  w[12]=656;w[13]=e[12];w[14]=uint32_t(args.output);w[15]=uint32_t(args.output>>32);w[17]=e[19];
  w[24]=e[14];w[25]=e[15];w[26]=e[16];w[27]=e[17];
 }
 a.eventsDst=put(templates);
 auto deviceArg=put(std::vector<pipeline_ab::Args>{a});
 auto audit=[&](uint32_t completed)->size_t {
 auto generations=get<uint32_t>(a.flags,16*32);for(uint32_t q=0;q<4;++q)if(generations[q*32+1]!=completed)throw std::runtime_error("generation mismatch");
 auto rows=get<rotating::Row>(c.rows,1);auto mapping=get<uint32_t>(c.mapping,H);auto slots=get<uint32_t>(c.locations,8192);auto rawWq=get<uint32_t>(sq,4*8192*16);auto t=get<uint64_t>(a.times,16);auto db=get<uint64_t>(doorbell,64);std::vector<uint32_t> wq;uint32_t totalPosted=0;auto queuesAfter=get<uint64_t>(queue,32);
 for(uint32_t q=0;q<4;++q){
  if(db[q*16]<initialHeads[q] || db[q*16]-initialHeads[q]>2048)throw std::runtime_error("invalid per-Q posted count");
  if(queuesAfter[q*8+1]!=db[q*16])throw std::runtime_error("64-bit producer/doorbell mismatch");
  uint32_t count=uint32_t(db[q*16]-initialHeads[q]);
  for(uint32_t i=0;i<count;++i){
   uint64_t head=initialHeads[q]+i;uint32_t offset=(q*8192+(head&8191))*16;
   uint32_t expected=((head&8192)?0U:0x80000000U)|(i+1==count?0x30360000U:0x30150000U);
   if(rawWq[offset]!=expected)throw std::runtime_error("WQE owner/completion mismatch");
   for(uint32_t f=1;f<16;++f){
    if(f==10 || f==11 || f==14 || f==15)continue;
    if(rawWq[offset+f]!=templates[q*32+f])throw std::runtime_error("immutable WQE field mismatch");
   }
   wq.insert(wq.end(),rawWq.begin()+offset,rawWq.begin()+offset+16);
  }
  totalPosted+=count;
 }
 if(totalPosted!=146)throw std::runtime_error("total post count mismatch");printf("cq_per_core=1\n");
 auto finalJournal=get<rotating::MappingJournal>(c.mappingJournal,1)[0];if(finalJournal.count || (mode==4 && get<uint64_t>(a.times,32)[30]!=84))throw std::runtime_error("journal high-water/retirement mismatch");
 for(uint32_t i=0;i<2048;++i)if(finalJournal.tokens[i]!=journal.tokens[i])throw std::runtime_error("final path unnecessarily appended journal");
 for(uint32_t i=0;i<84;++i)if(mapping[journal.tokens[i]]!=UINT32_MAX)throw std::runtime_error("prior mapping not retired");
 size_t bad=0;size_t checkpoint=0;std::vector<bool> seen(146);
 for(uint32_t i=0;i<146;++i){uint64_t remote=uint64_t(wq[i*16+10])|(uint64_t(wq[i*16+11])<<32);uint64_t local=uint64_t(wq[i*16+14])|(uint64_t(wq[i*16+15])<<32);uint32_t token=(remote-0x100000000ULL)/656,physical=(local-args.output)/656;
 if(token>=H||physical<10000||physical>=12048){++bad;continue;}
 bad+=mapping[token]!=UINT32_MAX;mapping[token]=R+physical-10000;}

 for(uint32_t i=0;i<146;++i){uint32_t v=mapping[prior+i];if(v<R+84||v>=R+230){bad++;continue;}seen[v-R-84]=true;}
 for(bool v:seen)bad+=!v;
 for(uint32_t i=0;i<146;++i){uint64_t remote=uint64_t(wq[i*16+10])|(uint64_t(wq[i*16+11])<<32);uint32_t token=(remote-0x100000000ULL)/656;if(token<prior||token>=prior+146){bad++;continue;}uint64_t local=uint64_t(wq[i*16+14])|(uint64_t(wq[i*16+15])<<32);if(mapping[token]<R||mapping[token]>=R+2048){bad++;continue;}bad+=local!=args.output+uint64_t(lru[mapping[token]-R])*656;}
 printf("audit_wqe=%zu\n",bad-checkpoint);checkpoint=bad;
 auto prefixout=get<uint8_t>(c.incremental,inc.size());
 for(uint32_t q=0;q<4;++q)for(uint32_t i=0;i<K;++i){memcpy(&selected[q*K+i],prefixout.data()+512+4*K*2+q*K*4+i*4,4);selected[q*K+i]&=0x3fffffffU;}
 for(uint32_t q=0;q<4;++q){std::vector<uint32_t> want;for(uint32_t i=counts[q];i<K;++i)want.push_back((mode==8 || mode==9) && i==1000?H-596+q:(mode==2 || mode==11) && i==100?abcBegin:i);for(uint32_t i=0;i<counts[q];++i)want.push_back(prior+begins[q]+i);
  std::vector<uint32_t> got(selected.begin()+q*K,selected.begin()+(q+1)*K);std::sort(want.begin(),want.end());std::sort(got.begin(),got.end());bad+=got!=want;

 }
 printf("audit_topk=%zu\n",bad-checkpoint);checkpoint=bad;
 bad+=rows[0].requests!=230 || totalPosted!=146;

 printf("audit_counts=%zu\n",bad-checkpoint);checkpoint=bad;
 auto logicalOut=get<uint32_t>(a.eventsSrc,8192);auto finalKeys=get<uint64_t>(a.keys,H*16);auto finalScales=get<uint32_t>(a.scales,H);auto finalLogical=get<uint64_t>(a.logical,H);auto finalAddresses=get<uint64_t>(a.addresses,H);auto errors=get<uint32_t>(a.queryErrors,128);
 for(uint32_t q=0;q<4;++q)for(uint32_t i=0;i<K;++i){uint32_t token=selected[q*K+i],physical=token;if(token>=H){++bad;continue;}if((mode==2 || mode==11) && token==abcBegin)physical=R+abcBegin%abcCapacity;else if((mode==8 || mode==9) && token>=H-596 && token<H-512)physical=lru[token-(H-596)];else if(token>=R){uint32_t ordinal=mapping[token]-R;physical=ordinal<2048?lru[ordinal]:UINT32_MAX;}uint32_t expectedLogical=((mode==8 || mode==9) && token>=H-596 && token<H-512)?token*3+7:uint32_t(logical[token]);bad+=logicalOut[q*K+i]!=expectedLogical||slots[q*K+i]!=physical;}
 printf("audit_output=%zu\n",bad-checkpoint);checkpoint=bad;
 for(uint32_t token=prior;token<prior+146;++token){if(mapping[token]<R||mapping[token]>=R+2048){++bad;continue;}uint32_t physical=lru[mapping[token]-R];bad+=finalLogical[physical]!=logical[token]||finalLogical[token]!=logical[physical]||finalAddresses[physical]!=addresses[token]||finalAddresses[token]!=addresses[physical]||finalScales[physical]!=scales[token]||finalScales[token]!=scales[physical];for(uint32_t j=0;j<16;++j)bad+=finalKeys[physical*16+j]!=keys[token*16+j]||finalKeys[token*16+j]!=keys[physical*16+j];}
 printf("audit_index=%zu\n",bad-checkpoint);checkpoint=bad;
 for(auto error:errors)bad+=error!=0;
 printf("audit_errors=%zu\n",bad-checkpoint);checkpoint=bad;
 auto times=get<uint64_t>(a.times,256);for(uint32_t q=0;q<4;++q)for(uint32_t k=0;k<32;++k)printf("stamp%u=%llu\n",q*32+k,(unsigned long long)times[q*32+k]);
 auto ev=get<rotating::Event>(c.events,5)[3];uint64_t phases[5];memcpy(phases,ev.pad,sizeof(phases));
 printf("topk_ticks=%llu service_ticks=%llu scan_ticks=%llu candidate_ticks=%llu post_ticks=%llu\n",
  (unsigned long long)(phases[3]-phases[2]),(unsigned long long)(ev.end-ev.begin),
  (unsigned long long)ev.scanTicks,(unsigned long long)ev.candidateTicks,(unsigned long long)ev.postTicks);
 printf("mode=%u posted=%u bad=%zu\n",mode,rows[0].requests-84,bad);fflush(stdout);
 return bad;
 };
 for(uint32_t repeat=0;repeat<((mode==3 || mode==9 || mode==15)?2U:1U);++repeat){
  if(repeat){
   if(get<rotating::MappingJournal>(c.mappingJournal,1)[0].count)throw std::runtime_error("journal retirement failed");
   auto retiredPrior=get<uint32_t>(c.mapping,H);for(uint32_t i=0;i<84;++i)if(retiredPrior[journal.tokens[i]]!=UINT32_MAX)throw std::runtime_error("prior mapping retirement failed");
   for(uint32_t i=prior;i<prior+146;++i)if(retiredPrior[i]!=UINT32_MAX)throw std::runtime_error("mapping retirement failed");
   restore(c.mapping,initialMapping);
   restore(c.mappingJournal,std::vector<rotating::MappingJournal>{journal});
   auto retiredMeta=get<uint64_t>(c.incremental,64);for(auto value:retiredMeta)if(value)throw std::runtime_error("TopK metadata retirement failed");
   restore(c.incremental,inc);restore(c.progress,std::vector<uint32_t>(128,H));restore(c.rows,std::vector<rotating::Row>{row});
   restore(a.keys,keys);restore(a.scales,scales);restore(a.logical,logical);restore(a.addresses,addresses);
   if(mode==15){
    auto retained=get<uint64_t>(queue,32);
    for(uint32_t q=0;q<4;++q)initialHeads[q]=retained[q*8+1];
   }else restore(queue,std::vector<uint64_t>(32));
  }
  ck(aclrtlaunch_test_service(4,stream,(void*)deviceArg,mode));ck(aclrtSynchronizeStream(stream));
  if(audit(repeat+1))throw std::runtime_error("per-launch full audit failed");
  printf("verified_launch=%u\n",repeat+1);fflush(stdout);
 }
 ck(aclrtDestroyStream(stream));ck(aclrtResetDevice(0));ck(aclFinalize());return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 2;}}
