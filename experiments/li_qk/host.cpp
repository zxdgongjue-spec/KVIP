// Non-fused LI reference and Q/K task controls; no transport library/launches.
#include <acl/acl.h>
#include "context.h"
#include "plan.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <string>

extern "C" uint32_t aclrtlaunch_li_qk_score(uint32_t,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_li_qk_scatter(uint32_t,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_li_qk_topk(uint32_t,void*,void*,void*,void*,uint32_t,uint32_t);
static void Check(uint32_t c){if(c)throw std::runtime_error("ACL error "+std::to_string(c));}
static void Require(bool c,const std::string& msg){if(!c)throw std::runtime_error(msg);}
template<class T> static std::vector<T> Read(const std::string& path,size_t count) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    Require(bool(f)&&uint64_t(f.tellg())==count*sizeof(T),"input size: "+path);
    f.seekg(0);std::vector<T> data(count);f.read(reinterpret_cast<char*>(data.data()),count*sizeof(T));
    Require(bool(f),"input read: "+path);return data;
}
struct Memory {
    struct Input {uint64_t address;std::vector<uint8_t> bytes;};
    std::vector<void*> allocations;std::vector<Input> inputs;
    template<class T> uint64_t Put(const std::vector<T>& v,bool readonly=false) {
        void* p=nullptr;size_t n=v.size()*sizeof(T);Require(n>0,"empty allocation");
        Check(aclrtMalloc(&p,n,ACL_MEM_MALLOC_HUGE_FIRST));allocations.push_back(p);
        Check(aclrtMemcpy(p,n,v.data(),n,ACL_MEMCPY_HOST_TO_DEVICE));
        if(readonly){const auto* b=reinterpret_cast<const uint8_t*>(v.data());inputs.push_back({uint64_t(p),{b,b+n}});}
        return uint64_t(p);
    }
    template<class T> std::vector<T> Get(uint64_t p,size_t n) {
        std::vector<T> v(n);Check(aclrtMemcpy(v.data(),n*sizeof(T),(void*)p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;
    }
    void CheckInputs(){for(const auto& i:inputs)Require(Get<uint8_t>(i.address,i.bytes.size())==i.bytes,"input changed");}
    ~Memory(){for(auto p:allocations)aclrtFree(p);}
};
struct Wave {uint64_t binding,score,tasks;uint32_t rows,first;};
static void Score(const std::vector<Wave>& waves,const li_qk::Plan& p,aclrtStream stream) {
    for(const auto& w:waves)Check(aclrtlaunch_li_qk_score(w.rows,stream,(void*)w.binding,w.rows,p.stride,p.heads,p.queryGroup));
}
// T1/Qparts1: one MIX1:2 group scans Q4 x the complete H, then a separate
// numeric TopK kernel selects K=2048 per real query. No fused streaming tail.
static void SerialLI(const Wave& w,const li_qk::Plan& p,uint64_t indices,uint64_t context,aclrtStream stream) {
    Require(p.historyParts==1 && p.queryParts==1 && w.rows==1,"serial baseline geometry");
    Check(aclrtlaunch_li_qk_score(1,stream,(void*)w.binding,1,p.stride,p.heads,4));
    Check(aclrtlaunch_li_qk_topk(4,stream,(void*)w.score,(void*)indices,(void*)context,p.history,p.heads));
}
static void SplitLI(const std::vector<Wave>& waves,const li_qk::Plan& p,uint64_t canonical,
                    uint64_t indices,uint64_t context,bool alias,aclrtStream stream) {
    Score(waves,p,stream);
    if(!alias)for(const auto& w:waves)Check(aclrtlaunch_li_qk_scatter(w.rows,stream,(void*)w.score,(void*)canonical,
        (void*)w.tasks,w.rows,p.stride,p.history,p.physicalGroup));
    Check(aclrtlaunch_li_qk_topk(4,stream,(void*)canonical,(void*)indices,(void*)context,p.history,p.heads));
}
static double Median(std::vector<double> v){std::sort(v.begin(),v.end());return v.size()%2?v[v.size()/2]:(v[v.size()/2-1]+v[v.size()/2])/2;}
static void Run(const std::string& fixture,const std::string& output,const li_qk::Plan& p,
                uint32_t warmup,uint32_t iterations,aclrtStream stream) {
    Memory m;uint32_t h=p.history,g=p.heads;
    auto query=Read<uint8_t>(fixture+"/query.bin",4ULL*g*128);
    auto key=Read<uint8_t>(fixture+"/key.bin",uint64_t(h)*128);key.resize(key.size()+128,0x5a);
    auto weight=Read<uint16_t>(fixture+"/weight.bin",4*g);
    auto qs=Read<float>(fixture+"/qscale.bin",4*g),ks=Read<float>(fixture+"/kscale.bin",h);
    auto oracle=Read<uint16_t>(fixture+"/oracle.bin",4ULL*h);
    uint64_t keyGm=m.Put(key,true),ksGm=m.Put(ks,true);
    std::vector<Wave> waves;
    for(uint32_t first=0;first<p.tasks.size();first+=p.capacity) {
        uint32_t rows=std::min(p.capacity,uint32_t(p.tasks.size())-first);
        // Physical Q2 storage for a logical Q1 tail preserves the existing
        // input row stride and Cube/Fixpipe alignment. qlen stays logical Q1.
        std::vector<uint8_t> q(uint64_t(rows)*p.physicalGroup*g*128,0);
        std::vector<uint16_t> w(uint64_t(rows)*p.physicalGroup*g,0);
        std::vector<float> z(w.size(),0);std::vector<uint32_t> qlen(rows,p.queryGroup),klen(rows);
        std::vector<int32_t> pages(uint64_t(rows)*p.stride/128,0);
        std::vector<li_qk::Task> tasks;
        for(uint32_t r=0;r<rows;++r) {
            auto t=p.tasks[first+r];tasks.push_back(t);klen[r]=t.end-t.begin;
            uint64_t from=uint64_t(t.queryBegin)*g,to=uint64_t(r)*p.physicalGroup*g;
            std::copy_n(query.begin()+from*128,p.queryGroup*g*128,q.begin()+to*128);
            std::copy_n(weight.begin()+from,p.queryGroup*g,w.begin()+to);
            std::copy_n(qs.begin()+from,p.queryGroup*g,z.begin()+to);
            for(uint32_t x=0;x<klen[r]/128;++x)pages[uint64_t(r)*p.stride/128+x]=t.begin/128+x;
        }
        uint64_t score=m.Put(std::vector<uint16_t>(uint64_t(rows)*p.physicalGroup*p.stride+128,0x5a5a));
        std::vector<uint64_t> binding(16,0);
        binding[0]=m.Put(q,true);binding[1]=keyGm;binding[2]=m.Put(w,true);binding[3]=m.Put(z,true);binding[4]=ksGm;
        binding[5]=m.Put(qlen,true);binding[6]=m.Put(klen,true);binding[7]=m.Put(pages,true);
        binding[9]=m.Put(std::vector<int32_t>(uint64_t(rows)*p.physicalGroup*2048+128,-7));binding[12]=score;
        waves.push_back({m.Put(binding,true),score,m.Put(tasks,true),rows,first});
    }
    bool baseline=p.historyParts==1 && p.queryParts==1;
    // Q2 groups on unsplit H are already contiguous Q0,Q1,Q2,Q3 rows.
    // Charge pack only when the representation actually needs a device copy.
    bool alias=p.historyParts==1 && p.physicalGroup==p.queryGroup && waves.size()==1;
    uint64_t canonical=alias?waves[0].score:m.Put(std::vector<uint16_t>(4ULL*h+128,0x5a5a));
    uint64_t indices=m.Put(std::vector<int32_t>(4*2048+128,-7));
    rotating::Context ctx{};ctx.shape.batch=1;ctx.shape.history=ctx.shape.resident=h;ctx.shape.indexHeads=g;
    uint64_t context=m.Put(std::vector<rotating::Context>{ctx},true);
    const bool native=std::string(LI_QK_RUN_MODE)=="npu";
    aclrtEvent start=nullptr,end=nullptr;if(native){Check(aclrtCreateEvent(&start));Check(aclrtCreateEvent(&end));}
    std::vector<double> scoreUs,liUs,hostScoreUs,hostLiUs;
    auto call=[&](bool complete,bool measured) {
        if(native && measured)Check(aclrtRecordEvent(start,stream));
        auto began=std::chrono::steady_clock::now();
        if(!complete)Score(waves,p,stream);
        else if(baseline)SerialLI(waves[0],p,indices,context,stream);
        else SplitLI(waves,p,canonical,indices,context,alias,stream);
        if(native && measured)Check(aclrtRecordEvent(end,stream));
        Check(aclrtSynchronizeStream(stream));
        if(measured) {
            (complete?hostLiUs:hostScoreUs).push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-began).count());
            if(native){float ms=0;Check(aclrtEventElapsedTime(&ms,start,end));Require(ms>0,"invalid device event time");(complete?liUs:scoreUs).push_back(ms*1000.0);}
        }
    };
    for(uint32_t i=0;i<warmup;++i){call(false,false);call(true,false);}
    for(uint32_t i=0;i<iterations;++i){if(i%2){call(true,true);call(false,true);}else{call(false,true);call(true,true);}}
    if(native){Check(aclrtDestroyEvent(start));Check(aclrtDestroyEvent(end));}
    auto flat=m.Get<uint16_t>(canonical,4ULL*h+128);
    for(uint64_t x=0;x<4ULL*h;++x)Require(flat[x]==oracle[x],"canonical score mismatch at "+std::to_string(x));
    for(uint64_t x=4ULL*h;x<flat.size();++x)Require(flat[x]==0x5a5a,"canonical guard changed");
    for(const auto& wave:waves) {
        auto raw=m.Get<uint16_t>(wave.score,uint64_t(wave.rows)*p.physicalGroup*p.stride+128);
        for(uint32_t r=0;r<wave.rows;++r) {
            auto t=p.tasks[wave.first+r];uint32_t length=t.end-t.begin;
            for(uint32_t q=0;q<p.physicalGroup;++q)for(uint32_t x=0;x<p.stride;++x) {
                uint16_t expected=q<p.queryGroup && x<length?oracle[uint64_t(t.queryBegin+q)*h+t.begin+x]:0x5a5a;
                Require(raw[(uint64_t(r)*p.physicalGroup+q)*p.stride+x]==expected,"task score/padding mismatch");
            }
        }
        for(uint64_t x=uint64_t(wave.rows)*p.physicalGroup*p.stride;x<raw.size();++x)Require(raw[x]==0x5a5a,"task guard changed");
    }
    auto selected=m.Get<int32_t>(indices,4*2048+128);
    for(uint32_t q=0;q<4;++q) {
        std::vector<uint16_t> sorted(oracle.begin()+uint64_t(q)*h,oracle.begin()+uint64_t(q+1)*h);
        std::nth_element(sorted.begin(),sorted.begin()+2047,sorted.end(),std::greater<uint16_t>());uint16_t theta=sorted[2047];
        std::set<int32_t> got;
        for(uint32_t k=0;k<2048;++k){int32_t x=selected[q*2048+k];Require(x>=0&&uint32_t(x)<h,"TopK index bounds");Require(got.insert(x).second,"TopK duplicate");Require(oracle[uint64_t(q)*h+x]>=theta,"TopK below threshold");}
        for(uint32_t x=0;x<h;++x)if(oracle[uint64_t(q)*h+x]>theta)Require(got.count(x),"TopK missing mandatory score");
    }
    for(uint32_t i=4*2048;i<selected.size();++i)Require(selected[i]==-7,"TopK guard changed");
    m.CheckInputs();
    std::ofstream f(output+"/score-keys.bin",std::ios::binary);f.write(reinterpret_cast<char*>(flat.data()),4ULL*h*2);Require(bool(f),"score file write");
    std::ofstream k(output+"/topk.bin",std::ios::binary);k.write(reinterpret_cast<char*>(selected.data()),4*2048*4);Require(bool(k),"TopK file write");
    std::ofstream report(output+"/report.json");report<<std::setprecision(10);
    report<<"{\"status\":\"PASS\",\"run_mode\":\""<<LI_QK_RUN_MODE<<"\",\"batch\":1,\"queries\":4,\"history\":"<<h<<",\"heads\":"<<g
      <<",\"history_parts\":"<<p.historyParts<<",\"query_parts\":"<<p.queryParts<<",\"logical_query_group\":"<<p.queryGroup<<",\"physical_query_tile\":"<<p.physicalGroup
      <<",\"capacity\":"<<p.capacity<<",\"tasks\":"<<p.tasks.size()<<",\"score_waves\":"<<waves.size()<<",\"score_launches\":"<<waves.size()
      <<",\"scatter_launches\":"<<(alias?0:waves.size())<<",\"canonical_alias\":"<<(alias?"true":"false")<<",\"topk_launches\":1,\"fetch_launches\":0,\"final_topk_aivs\":4,\"k\":2048,\"warmup\":"<<warmup<<",\"iterations\":"<<iterations
      <<",\"score_p50_us\":";if(native)report<<Median(scoreUs);else report<<"null";
    report<<",\"li_p50_us\":";if(native)report<<Median(liUs);else report<<"null";
    auto array=[&](const char* name,const std::vector<double>& v){report<<",\""<<name<<"\":[";for(size_t i=0;i<v.size();++i){if(i)report<<",";report<<v[i];}report<<"]";};
    array("score_samples_us",scoreUs);array("li_samples_us",liUs);array("host_score_samples_us",hostScoreUs);array("host_li_samples_us",hostLiUs);
    report<<",\"guards\":true,\"inputs_unchanged\":true,\"topk_threshold_oracle\":true,\"scope\":\"standalone numeric LI; score and score+canonical+global TopK; no Fetch/Service/Index\"}\n";
    Require(bool(report),"report write");std::cout<<"LI_QK_ORACLE_PASS T="<<p.historyParts<<" Qparts="<<p.queryParts<<"\n";
}
int main(int argc,char** argv){try {
    if(argc==2 && std::string(argv[1])=="--capabilities"){
        std::cout<<"{\"run_mode\":\""<<LI_QK_RUN_MODE<<"\",\"scope\":\"standalone LI QK controls\",\"fetch_launches\":0,\"queries\":4,\"batch\":1,\"k\":2048}\n";return 0;
    }
    Require(argc==12,"li_qk FIXTURE OUTPUT H HEADS C V T QPARTS WARMUP ITERATIONS DEVICE");
    auto number=[](const char* s){size_t used=0;uint64_t n=std::stoull(s,&used);Require(s[used]==0&&n<=UINT32_MAX,"integer required");return uint32_t(n);};
    auto p=li_qk::MakePlan(number(argv[3]),number(argv[4]),number(argv[5]),number(argv[6]),number(argv[7]),number(argv[8]));
    uint32_t warmup=number(argv[9]),iterations=number(argv[10]);Require(warmup>=1&&warmup<=100&&iterations>=1&&iterations<=1000,"warmup 1..100, iterations 1..1000");
    Require(!std::filesystem::exists(argv[2]),"fresh output required");std::filesystem::create_directories(argv[2]);
    Check(aclInit(nullptr));Check(aclrtSetDevice(number(argv[11])));aclrtStream stream=nullptr;Check(aclrtCreateStream(&stream));
    Run(argv[1],argv[2],p,warmup,iterations,stream);Check(aclrtDestroyStream(stream));Check(aclrtResetDevice(number(argv[11])));Check(aclFinalize());return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
