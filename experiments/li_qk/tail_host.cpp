#include <acl/acl.h>
#include "context.h"
#include "tail_plan.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" uint32_t aclrtlaunch_li_tail_select(uint32_t,void*,void*,uint32_t,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_li_tail_gather(uint32_t,void*,void*,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_li_tail_map(uint32_t,void*,void*,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_li_tail_resident_finish(uint32_t,void*,void*,uint32_t);
extern "C" uint32_t aclrtlaunch_li_tail_cold(uint32_t,void*,void*,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_li_tail_classify(uint32_t,void*,void*,uint32_t);
extern "C" uint32_t aclrtlaunch_li_tail_union(uint32_t,void*,void*,uint32_t);
static void Need(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
static void Check(uint32_t c){Need(!c,"ACL error "+std::to_string(c));}
struct Memory {
 struct Input{uint64_t address;std::vector<uint8_t> bytes;};std::vector<void*> pointers;std::vector<Input> inputs;
 template<class T> uint64_t Put(const std::vector<T>& v,bool readonly=false){void* p=nullptr;size_t n=v.size()*sizeof(T);Need(n>0,"empty buffer");
  Check(aclrtMalloc(&p,n,ACL_MEM_MALLOC_HUGE_FIRST));pointers.push_back(p);Check(aclrtMemcpy(p,n,v.data(),n,ACL_MEMCPY_HOST_TO_DEVICE));
  if(readonly){auto b=reinterpret_cast<const uint8_t*>(v.data());inputs.push_back({uint64_t(p),{b,b+n}});}return uint64_t(p);}
 template<class T> std::vector<T> Get(uint64_t p,size_t n){std::vector<T> v(n);Check(aclrtMemcpy(v.data(),n*sizeof(T),(void*)p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
 void Verify(){for(auto& i:inputs)Need(Get<uint8_t>(i.address,i.bytes.size())==i.bytes,"readonly input changed");}
 ~Memory(){for(auto p:pointers)aclrtFree(p);}
};
static uint16_t Threshold(const std::vector<uint16_t>& keys,uint32_t q,uint32_t h,uint32_t begin,uint32_t end){
 Need(end-begin>=2048,"at least K valid values required");std::vector<uint16_t> v(keys.begin()+uint64_t(q)*h+begin,keys.begin()+uint64_t(q)*h+end);
 std::nth_element(v.begin(),v.begin()+2047,v.end(),std::greater<uint16_t>());return v[2047];}
static void TopCheck(const std::vector<uint16_t>& keys,const std::vector<int32_t>& indices,uint32_t q,uint32_t h,uint32_t end){
 uint16_t threshold=Threshold(keys,q,h,0,end);std::set<int32_t> seen;
 for(uint32_t i=0;i<2048;++i){int32_t x=indices[q*2048+i];Need(x>=0 && uint32_t(x)<end,"TopK bounds");Need(seen.insert(x).second,"TopK duplicate");Need(keys[uint64_t(q)*h+x]>=threshold,"TopK below Kth");}
 for(uint32_t x=0;x<end;++x)if(keys[uint64_t(q)*h+x]>threshold)Need(seen.count(x),"TopK misses mandatory value");
}
static double Median(std::vector<double> v){std::sort(v.begin(),v.end());return v.size()%2?v[v.size()/2]:(v[v.size()/2-1]+v[v.size()/2])/2;}
int main(int argc,char** argv){try{
 if(argc==2 && std::string(argv[1])=="--capabilities"){std::cout<<"{\"run_mode\":\""<<LI_QK_RUN_MODE<<"\",\"scope\":\"post-score controls\"}\n";return 0;}
 Need(argc==14,"li_tail FIXTURE OUTPUT B H R P RP COLD_PAR FINAL_SPLIT STAGE WARMUP ITER DEVICE");
 auto num=[](const char* s){size_t n=0;uint64_t x=std::stoull(s,&n);Need(s[n]==0 && x<=UINT32_MAX,"integer required");return uint32_t(x);};
 uint32_t b=num(argv[3]),h=num(argv[4]),r=num(argv[5]),p=num(argv[6]),rp=num(argv[7]),cp=num(argv[8]),fs=num(argv[9]),warm=num(argv[11]),iterations=num(argv[12]);
 std::string stage=argv[10],out=argv[2];bool native=std::string(LI_QK_RUN_MODE)=="npu";
 Need(li_tail::Valid(b,h,r,p,rp) && cp<=1 && fs<=1,"invalid post-score geometry/control");
 Need(stage=="all" || stage=="cold" || stage=="final" || stage=="resident","unknown stage");
 Need(iterations>=1 && iterations<=1000 && warm<=100 && (!native || (warm>=1 && stage=="all")),"physical run needs warmup/all stage");
 Need(!std::filesystem::exists(out),"fresh output required");std::filesystem::create_directories(out);
 std::vector<uint16_t> keys(uint64_t(b)*4*h);std::ifstream input(std::string(argv[1])+"/oracle.bin",std::ios::binary|std::ios::ate);
 Need(input && uint64_t(input.tellg())==keys.size()*2,"fixture size");input.seekg(0);input.read(reinterpret_cast<char*>(keys.data()),keys.size()*2);Need(bool(input),"fixture read");
 // Zero is reserved for invalid local candidates; FP8/ReLU positive-weight
 // fixture sortable keys are strictly above zero. Do not silently generalize.
 Need(std::all_of(keys.begin(),keys.end(),[](uint16_t x){return x>0;}),"reserved zero key in fixture");
 Check(aclInit(nullptr));Check(aclrtSetDevice(num(argv[13])));aclrtStream stream=nullptr;Check(aclrtCreateStream(&stream));
 {Memory m;uint32_t queries=b*4,words=r/32;li_tail::Bindings a{};a.batch=b;a.history=h;a.resident=r;a.parts=fs?p:1;a.residentParts=rp;a.coldParts=p;
  auto keyGuard=keys;keyGuard.resize(keys.size()+128,0x5a5a);a.scores=m.Put(keyGuard,true);
  a.selected=m.Put(std::vector<int32_t>(queries*2048+128,-7));a.residentSelected=m.Put(std::vector<int32_t>(queries*2048+128,-7));
  uint32_t candidateCount=queries*std::max(p,rp)*2048;
  a.localIndices=m.Put(std::vector<int32_t>(candidateCount+128,-7));a.candidateScores=m.Put(std::vector<uint16_t>(candidateCount+128,0x5a5a));
  a.candidateTokens=m.Put(std::vector<int32_t>(candidateCount+128,-7));a.mergeIndices=m.Put(std::vector<int32_t>(queries*2048+128,-7));
  std::vector<uint32_t> ends(queries),lru(b*2048),mapping(uint64_t(b)*h,UINT32_MAX);
  for(uint32_t q=0;q<queries;++q)ends[q]=h-3+q%4;
  for(uint32_t owner=0;owner<b;++owner){for(uint32_t i=0;i<2048;++i)lru[owner*2048+i]=(i*7+owner*13)%r;
   for(uint32_t x=0;x<r;++x)mapping[uint64_t(owner)*h+x]=x;
   for(uint32_t x=r;x<h;x+=8)mapping[uint64_t(owner)*h+x]=r+(x%2048);}
  std::vector<uint16_t> prefix(uint64_t(b)*(4*2048*3+512)+128,0x5a5a);
  if(stage=="cold" || stage=="final")for(uint32_t q=0;q<queries;++q)prefix[uint64_t(q/4)*(4*2048*3+512)+4*2048*3+(q%4)*128]=Threshold(keys,q,h,0,r);
  a.prefix=m.Put(prefix);std::vector<uint8_t> hits(uint64_t(b)*h+128,0);std::fill(hits.end()-128,hits.end(),0x5a);a.hits=m.Put(hits);
  a.locations=m.Put(std::vector<uint32_t>(queries*2048+128,0x5a5a5a5a));
  std::vector<uint32_t> bitmap(queries*words+128,0),errors(queries*32+128,0),united(b*words+128,0);
  std::fill(bitmap.end()-128,bitmap.end(),0x5a5a5a5a);std::fill(errors.end()-128,errors.end(),0x5a5a5a5a);std::fill(united.end()-128,united.end(),0x5a5a5a5a);
  a.queryBitmaps=m.Put(bitmap);a.queryErrors=m.Put(errors);a.unionBitmap=m.Put(united);
  rotating::Context ctx{};ctx.shape.batch=b;ctx.shape.history=h;ctx.shape.resident=r;ctx.shape.indexHeads=32;
  ctx.queryEnds=m.Put(ends,true);ctx.lruSlots=m.Put(lru,true);ctx.mapping=m.Put(mapping,true);ctx.hits=a.hits;ctx.locations=a.locations;
  a.context=m.Put(std::vector<rotating::Context>{ctx},true);ctx.queryEnds=0;a.plainContext=m.Put(std::vector<rotating::Context>{ctx},true);
  uint64_t bindings=m.Put(std::vector<li_tail::Bindings>{a},true);
  struct Launch{std::string kernel,phase;uint32_t first,cores;};std::vector<Launch> launches;
  auto wave=[&](uint32_t jobs,const std::string& kernel,const std::string& phase,auto fn){for(uint32_t first=0;first<jobs;first+=64){uint32_t n=std::min(64U,jobs-first);Check(fn(n,first));launches.push_back({kernel,phase,first,n});}};
  auto select=[&](bool resident){uint32_t parts=resident?rp:a.parts;std::string phase=resident?"resident":"final";
   wave(queries*parts,"li_tail_select",phase+"_local",[&](uint32_t n,uint32_t f){return aclrtlaunch_li_tail_select(n,stream,(void*)bindings,f,resident,0);});
   if(parts>1){wave(queries*parts,"li_tail_gather",phase+"_gather",[&](uint32_t n,uint32_t f){return aclrtlaunch_li_tail_gather(n,stream,(void*)bindings,f,resident);});
    wave(queries,"li_tail_select",phase+"_merge",[&](uint32_t n,uint32_t f){return aclrtlaunch_li_tail_select(n,stream,(void*)bindings,f,resident,1);});
    wave(queries,"li_tail_map",phase+"_map",[&](uint32_t n,uint32_t f){return aclrtlaunch_li_tail_map(n,stream,(void*)bindings,f,resident);});}};
  auto call=[&](){launches.clear();if(stage=="all" || stage=="resident"){select(true);if(rp>1)wave(queries,"li_tail_resident_finish","resident_theta_hot",[&](uint32_t n,uint32_t f){return aclrtlaunch_li_tail_resident_finish(n,stream,(void*)bindings,f);});}
   if(stage=="all" || stage=="cold")wave(cp?b*p*2:b,"li_tail_cold","cold",[&](uint32_t n,uint32_t f){return aclrtlaunch_li_tail_cold(n,stream,(void*)bindings,f,cp);});
   if(stage=="all" || stage=="final"){select(false);wave(queries,"li_tail_classify","final_classify",[&](uint32_t n,uint32_t f){return aclrtlaunch_li_tail_classify(n,stream,(void*)bindings,f);});
    wave(b,"li_tail_union","four_q_union",[&](uint32_t n,uint32_t f){return aclrtlaunch_li_tail_union(n,stream,(void*)bindings,f);});}};
  aclrtEvent start=nullptr,end=nullptr;if(native){Check(aclrtCreateEvent(&start));Check(aclrtCreateEvent(&end));}
  for(uint32_t i=0;i<warm;++i){call();Check(aclrtSynchronizeStream(stream));}
  std::vector<double> samples;
  for(uint32_t i=0;i<iterations;++i){if(native)Check(aclrtRecordEvent(start,stream));call();if(native)Check(aclrtRecordEvent(end,stream));Check(aclrtSynchronizeStream(stream));
   if(native){float ms=0;Check(aclrtEventElapsedTime(&ms,start,end));Need(ms>0,"invalid device time");samples.push_back(ms*1000);}}
  if(native){Check(aclrtDestroyEvent(start));Check(aclrtDestroyEvent(end));}
  auto residentSelected=m.Get<int32_t>(a.residentSelected,queries*2048+128),selected=m.Get<int32_t>(a.selected,queries*2048+128);
  auto gotPrefix=m.Get<uint16_t>(a.prefix,prefix.size());auto gotHits=m.Get<uint8_t>(a.hits,hits.size());
  bool res=stage=="all" || stage=="resident",fin=stage=="all" || stage=="final",cold=stage=="all" || stage=="cold";
  // Production ProcessResidentTopK writes selected IDs into its prefix record.
  // Read that record for the oracle, without adding a device copy to the path.
  if(res && rp==1)for(uint32_t q=0;q<queries;++q)for(uint32_t i=0;i<2048;++i){
   uint64_t off=uint64_t(q/4)*(4*2048*3+512)+4*2048+(q%4)*2048*2+i*2;
   residentSelected[q*2048+i]=int32_t(uint32_t(gotPrefix[off])|(uint32_t(gotPrefix[off+1])<<16));}
  for(uint32_t q=0;q<queries;++q){uint16_t theta=Threshold(keys,q,h,0,r);uint64_t off=uint64_t(q/4)*(4*2048*3+512)+4*2048*3+(q%4)*128;
   Need(gotPrefix[off]==theta,"resident theta mismatch");if(res)TopCheck(keys,residentSelected,q,h,r);if(fin)TopCheck(keys,selected,q,h,ends[q]);
   for(uint32_t byte=0;byte<h/4;++byte){uint8_t expected=0;
    if(res && byte<512)for(uint32_t j=0;j<4;++j){uint32_t token=lru[q/4*2048+byte*4+j];if(keys[uint64_t(q)*h+token]>theta)expected|=1U<<(2*j);}
    if(cold && byte>=r/4)for(uint32_t j=0;j<4;++j){uint32_t token=byte*4+j;if(token<ends[q] && keys[uint64_t(q)*h+token]>theta)expected|=1U<<(2*j);}
    Need(gotHits[uint64_t(q)*h/4+byte]==expected,"Hot/Cold packed predicate mismatch");}}
  if(fin){auto locations=m.Get<uint32_t>(a.locations,queries*2048+128),gotBits=m.Get<uint32_t>(a.queryBitmaps,bitmap.size()),gotErrors=m.Get<uint32_t>(a.queryErrors,errors.size()),gotUnion=m.Get<uint32_t>(a.unionBitmap,united.size());
   for(uint32_t q=0;q<queries;++q)for(uint32_t i=0;i<2048;++i){uint32_t token=selected[q*2048+i],slot=mapping[uint64_t(q/4)*h+token];if(slot>=r && slot-r<2048)slot=lru[q/4*2048+slot-r];else if(slot>=r)slot=UINT32_MAX;
    Need(locations[q*2048+i]==slot,"Final location mismatch");if(slot<r)bitmap[q*words+slot/32]|=1U<<(slot%32);}
   for(uint32_t owner=0;owner<b;++owner)for(uint32_t x=0;x<words;++x)united[owner*words+x]=bitmap[owner*4*words+x]|bitmap[(owner*4+1)*words+x]|bitmap[(owner*4+2)*words+x]|bitmap[(owner*4+3)*words+x];
   Need(gotBits==bitmap && gotErrors==errors && gotUnion==united,"Final bitmap/error/union mismatch");
   for(uint32_t x=queries*2048;x<locations.size();++x)Need(locations[x]==0x5a5a5a5a,"locations guard");}
  for(uint32_t x=queries*2048;x<selected.size();++x)Need(selected[x]==-7 && residentSelected[x]==-7,"TopK guards");
  for(uint32_t x=b*h;x<gotHits.size();++x)Need(gotHits[x]==0x5a,"predicate guard");
  for(uint32_t x=prefix.size()-128;x<prefix.size();++x)Need(gotPrefix[x]==0x5a5a,"theta guard");
  for(auto address:{a.localIndices,a.candidateTokens}){auto v=m.Get<int32_t>(address,candidateCount+128);for(uint32_t x=candidateCount;x<v.size();++x)Need(v[x]==-7,"candidate ID guard");}
  auto ck=m.Get<uint16_t>(a.candidateScores,candidateCount+128);for(uint32_t x=candidateCount;x<ck.size();++x)Need(ck[x]==0x5a5a,"candidate score guard");m.Verify();
  std::ofstream report(out+"/report.json");report<<std::setprecision(12)<<"{\"status\":\"PASS\",\"run_mode\":\""<<LI_QK_RUN_MODE<<"\",\"stage\":\""<<stage<<"\",\"batch\":"<<b<<",\"history\":"<<h<<",\"resident\":"<<r<<",\"parts\":"<<p<<",\"resident_parts\":"<<rp<<",\"cold_parallel\":"<<cp<<",\"final_split\":"<<fs<<",\"tail_p50_us\":";
  if(native)report<<Median(samples);else report<<"null";report<<",\"samples_us\":[";for(size_t i=0;i<samples.size();++i){if(i)report<<",";report<<samples[i];}
  report<<"],\"oracle\":true,\"guards\":true,\"inputs_unchanged\":true,\"static_mapping_classification\":"<<(fin?"true":"false")<<",\"fetch_launches\":0,\"launches\":[";
  for(size_t i=0;i<launches.size();++i){if(i)report<<",";auto l=launches[i];report<<"{\"kernel\":\""<<l.kernel<<"\",\"phase\":\""<<l.phase<<"\",\"first\":"<<l.first<<",\"cores\":"<<l.cores<<"}";}
  report<<"]}\n";Need(bool(report),"report write");std::cout<<"LI_TAIL_ORACLE_PASS\n";
 }
 Check(aclrtDestroyStream(stream));Check(aclrtResetDevice(num(argv[13])));Check(aclFinalize());return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
