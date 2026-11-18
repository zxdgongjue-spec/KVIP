#include <acl/acl.h>
#include "context.h"
#include "radix_plan.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" uint32_t aclrtlaunch_li_qk_topk(uint32_t,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_li_radix_high(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_li_radix_high_reduce(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_li_radix_low(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_li_radix_low_reduce(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_li_radix_emit(uint32_t,void*,void*);
static void Need(bool v,const std::string& s){if(!v)throw std::runtime_error(s);}
static void Check(uint32_t c){Need(c==0,"ACL error "+std::to_string(c));}
struct Memory {
 struct Input{uint64_t address;std::vector<uint8_t> bytes;};std::vector<void*> ptrs;std::vector<Input> inputs;
 template<class T> uint64_t Put(const std::vector<T>& v,bool ro=false){void* p=nullptr;size_t n=v.size()*sizeof(T);
  Check(aclrtMalloc(&p,n,ACL_MEM_MALLOC_HUGE_FIRST));ptrs.push_back(p);Check(aclrtMemcpy(p,n,v.data(),n,ACL_MEMCPY_HOST_TO_DEVICE));
  if(ro){auto b=reinterpret_cast<const uint8_t*>(v.data());inputs.push_back({uint64_t(p),{b,b+n}});}return uint64_t(p);}
 template<class T> std::vector<T> Get(uint64_t p,size_t n){std::vector<T> v(n);Check(aclrtMemcpy(v.data(),n*sizeof(T),(void*)p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
 void Verify(){for(auto& i:inputs)Need(Get<uint8_t>(i.address,i.bytes.size())==i.bytes,"readonly input changed");}
 ~Memory(){for(auto p:ptrs)aclrtFree(p);}
};
int main(int argc,char** argv){try{
 if(argc==2 && std::string(argv[1])=="--capabilities"){std::cout<<"{\"run_mode\":\""<<LI_QK_RUN_MODE<<"\",\"scope\":\"exact post-score radix; no QK/Fetch\"}\n";return 0;}
 Need(argc==10,"li_radix FIXTURE OUTPUT B H P original|sharded WARMUP ITERATIONS DEVICE");
 auto num=[](const char* s){size_t used=0;uint64_t n=std::stoull(s,&used);Need(s[used]==0 && n<=UINT32_MAX,"integer required");return uint32_t(n);};
 uint32_t b=num(argv[3]),h=num(argv[4]),p=num(argv[5]),warm=num(argv[7]),iters=num(argv[8]),queries=b*4;
 std::string route=argv[6],out=argv[2];bool native=std::string(LI_QK_RUN_MODE)=="npu";
 Need(li_radix::Valid(b,h,p) && h>=4096,"B1/B16 H4096..128K aligned256 P1..32 required");
 Need((route=="original" && p==1) || route=="sharded","original P1 or sharded required");
 Need(iters>=1 && iters<=1000 && warm<=100 && (!native || warm>=1),"valid sampling required");
 Need(!std::filesystem::exists(out),"fresh output required");std::filesystem::create_directories(out);
 std::vector<uint16_t> keys(uint64_t(queries)*h);std::ifstream f(std::string(argv[1])+"/oracle.bin",std::ios::binary|std::ios::ate);
 Need(f && uint64_t(f.tellg())==keys.size()*2,"fixture size");f.seekg(0);f.read(reinterpret_cast<char*>(keys.data()),keys.size()*2);Need(bool(f),"fixture read");
 Check(aclInit(nullptr));Check(aclrtSetDevice(num(argv[9])));aclrtStream stream=nullptr;Check(aclrtCreateStream(&stream));
 {Memory m;auto guarded=keys;guarded.resize(keys.size()+128,0x5a5a);li_radix::Bindings a{};a.batch=b;a.history=h;a.parts=p;a.scores=m.Put(guarded,true);
  std::vector<uint32_t> ends(queries);for(uint32_t q=0;q<queries;++q)ends[q]=h-3+q%4;a.ends=m.Put(ends,true);
  a.selected=m.Put(std::vector<int32_t>(queries*2048+128,-7));
  a.high=m.Put(std::vector<uint32_t>(queries*p*256+128,0x5a5a5a5a));a.low=m.Put(std::vector<uint32_t>(queries*p*256+128,0x5a5a5a5a));
  a.params=m.Put(std::vector<uint32_t>(queries*32+128,0x5a5a5a5a));a.offsets=m.Put(std::vector<uint32_t>(queries*p*8+128,0x5a5a5a5a));
  uint64_t bindings=m.Put(std::vector<li_radix::Bindings>{a},true);rotating::Context ctx{};ctx.shape.batch=b;ctx.shape.history=ctx.shape.resident=h;ctx.shape.indexHeads=32;ctx.queryEnds=a.ends;
  uint64_t context=m.Put(std::vector<rotating::Context>{ctx},true);uint32_t jobs=queries*p,workers=std::min(64U,jobs);
  struct Launch{const char* name;uint32_t cores;};std::vector<Launch> launches;
  auto call=[&](){launches.clear();if(route=="original"){
    Check(aclrtlaunch_li_qk_topk(queries,stream,(void*)a.scores,(void*)a.selected,(void*)context,h,32,queries));launches.push_back({"li_qk_topk",queries});
   }else{
    Check(aclrtlaunch_li_radix_high(workers,stream,(void*)bindings));launches.push_back({"li_radix_high",workers});
    Check(aclrtlaunch_li_radix_high_reduce(queries,stream,(void*)bindings));launches.push_back({"li_radix_high_reduce",queries});
    Check(aclrtlaunch_li_radix_low(workers,stream,(void*)bindings));launches.push_back({"li_radix_low",workers});
    Check(aclrtlaunch_li_radix_low_reduce(queries,stream,(void*)bindings));launches.push_back({"li_radix_low_reduce",queries});
    Check(aclrtlaunch_li_radix_emit(workers,stream,(void*)bindings));launches.push_back({"li_radix_emit",workers});
   }};
  aclrtEvent start=nullptr,stop=nullptr;if(native){Check(aclrtCreateEvent(&start));Check(aclrtCreateEvent(&stop));}
  for(uint32_t i=0;i<warm;++i){call();Check(aclrtSynchronizeStream(stream));}std::vector<double> samples;
  for(uint32_t i=0;i<iters;++i){if(native)Check(aclrtRecordEvent(start,stream));call();if(native)Check(aclrtRecordEvent(stop,stream));Check(aclrtSynchronizeStream(stream));
   if(native){float ms=0;Check(aclrtEventElapsedTime(&ms,start,stop));Need(ms>0,"invalid device time");samples.push_back(ms*1000.0);}}
  if(native){Check(aclrtDestroyEvent(start));Check(aclrtDestroyEvent(stop));}
  auto selected=m.Get<int32_t>(a.selected,queries*2048+128);auto params=m.Get<uint32_t>(a.params,queries*32+128);
  auto high=m.Get<uint32_t>(a.high,queries*p*256+128),low=m.Get<uint32_t>(a.low,queries*p*256+128),offsets=m.Get<uint32_t>(a.offsets,queries*p*8+128);
  for(uint32_t q=0;q<queries;++q){std::vector<uint16_t> values(keys.begin()+uint64_t(q)*h,keys.begin()+uint64_t(q)*h+ends[q]);
   std::nth_element(values.begin(),values.begin()+2047,values.end(),std::greater<uint16_t>());uint16_t tau=values[2047];std::set<int32_t> seen;
   for(uint32_t i=0;i<2048;++i){int32_t x=selected[q*2048+i];Need(x>=0 && uint32_t(x)<ends[q],"TopK bounds");Need(seen.insert(x).second,"duplicate token");Need(keys[uint64_t(q)*h+x]>=tau,"below Kth");}
   for(uint32_t x=0;x<ends[q];++x)if(keys[uint64_t(q)*h+x]>tau)Need(seen.count(x),"missing mandatory GT token");
   if(route=="sharded"){
    Need(params[q*32+2]==tau,"global threshold mismatch");uint32_t totalGT=0,totalEQ=0;
    for(uint32_t part=0;part<p;++part){uint32_t hist[256]={},lob[256]={};uint32_t begin=li_radix::Begin(h,p,part),end=std::min(ends[q],li_radix::Begin(h,p,part+1));
     for(uint32_t x=begin;x<end;++x){uint16_t key=keys[uint64_t(q)*h+x];++hist[key>>8];if((key>>8)==(tau>>8))++lob[key&255];}
     uint32_t hc=0,lc=0;for(uint32_t j=0;j<256;++j){hc+=hist[j];lc+=lob[j];Need(high[(q*p+part)*256+j]==hc,"high cumulative histogram mismatch q="+std::to_string(q)+" p="+std::to_string(part)+" bin="+std::to_string(j)+" actual="+std::to_string(high[(q*p+part)*256+j])+" expected="+std::to_string(hc));Need(low[(q*p+part)*256+j]==lc,"low cumulative histogram mismatch q="+std::to_string(q)+" p="+std::to_string(part)+" bin="+std::to_string(j)+" actual="+std::to_string(low[(q*p+part)*256+j])+" expected="+std::to_string(lc));}
     uint32_t gt=0;for(uint32_t j=(tau>>8)+1;j<256;++j)gt+=hist[j];for(uint32_t j=(tau&255)+1;j<256;++j)gt+=lob[j];
     uint32_t quota=std::min(lob[tau&255],2048-params[q*32+3]-totalEQ);
     Need(offsets[(q*p+part)*8]==totalGT && offsets[(q*p+part)*8+1]==totalEQ && offsets[(q*p+part)*8+2]==quota,"prefix offset mismatch");totalGT+=gt;totalEQ+=quota;
    }Need(totalGT==params[q*32+3] && totalGT+totalEQ==2048,"GT/EQ accounting mismatch");
   }
  }
  for(uint32_t i=queries*2048;i<selected.size();++i)Need(selected[i]==-7,"output guard");
  for(auto pair:{std::pair<uint64_t,uint32_t>{a.high,queries*p*256},{a.low,queries*p*256},{a.params,queries*32},{a.offsets,queries*p*8}}){auto v=m.Get<uint32_t>(pair.first,pair.second+128);for(uint32_t i=pair.second;i<v.size();++i)Need(v[i]==0x5a5a5a5a,"metadata guard");}
  m.Verify();std::ofstream dump(out+"/topk.bin",std::ios::binary);dump.write(reinterpret_cast<char*>(selected.data()),queries*2048*4);
  std::ofstream report(out+"/report.json");report<<std::setprecision(12)<<"{\"status\":\"PASS\",\"run_mode\":\""<<LI_QK_RUN_MODE<<"\",\"route\":\""<<route<<"\",\"batch\":"<<b<<",\"history\":"<<h<<",\"queries\":4,\"parts\":"<<p<<",\"aiv_budget\":64,\"k\":2048,\"score_launches\":0,\"fetch_launches\":0,\"radix_p50_us\":";
  if(native){auto v=samples;std::sort(v.begin(),v.end());report<<(v.size()%2?v[v.size()/2]:(v[v.size()/2-1]+v[v.size()/2])/2);}else report<<"null";
  report<<",\"samples_us\":[";for(uint32_t i=0;i<samples.size();++i){if(i)report<<",";report<<samples[i];}
  report<<"],\"warmup\":"<<warm<<",\"iterations\":"<<iters<<",\"oracle\":true,\"guards\":true,\"inputs_unchanged\":true,\"launches\":[";
  for(uint32_t i=0;i<launches.size();++i){if(i)report<<",";report<<"{\"kernel\":\""<<launches[i].name<<"\",\"cores\":"<<launches[i].cores<<"}";}
  report<<"],\"scope\":\"isolated numeric TopK; immutable GM score; no resident/Cold/Service/Fetch\"}\n";Need(bool(report) && bool(dump),"write output");std::cout<<"LI_RADIX_ORACLE_PASS\n";
 }
 Check(aclrtDestroyStream(stream));Check(aclrtResetDevice(num(argv[9])));Check(aclFinalize());return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
