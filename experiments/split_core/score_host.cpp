// Score-only research driver; no Init/Service/TopK/transport launches.
#include <acl/acl.h>
#include "score_schedule.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

extern "C" uint32_t aclrtlaunch_split_score_only(uint32_t,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t);
static void Check(uint32_t c) { if(c)throw std::runtime_error("ACL/launch error "+std::to_string(c)); }
template<class T> std::vector<T> Read(const std::string& path,size_t count) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || uint64_t(f.tellg())!=count*sizeof(T))throw std::runtime_error("input size: "+path);
    f.seekg(0);std::vector<T> v(count);f.read(reinterpret_cast<char*>(v.data()),count*sizeof(T));
    if(!f)throw std::runtime_error("read: "+path);return v;
}
struct Memory {
    std::vector<void*> owned;
    template<class T> uint64_t Put(const std::vector<T>& v) {
        void* p=nullptr;size_t bytes=v.size()*sizeof(T);Check(aclrtMalloc(&p,bytes,ACL_MEM_MALLOC_HUGE_FIRST));
        owned.push_back(p);Check(aclrtMemcpy(p,bytes,v.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE));return uint64_t(p);
    }
    template<class T> std::vector<T> Get(uint64_t p,size_t count) {
        std::vector<T> v(count);Check(aclrtMemcpy(v.data(),count*sizeof(T),(void*)p,count*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;
    }
    ~Memory(){for(auto p:owned)aclrtFree(p);}
};
struct Wave { uint64_t binding,score;uint32_t rows,begin; };
static void Run(const std::string& fixture,const std::string& output,const kvip_score::Plan& plan,
                uint32_t iterations,aclrtStream stream) {
    Memory m;uint32_t b=plan.batch,h=plan.history,g=plan.heads,qg=plan.queryGroup;
    auto query=Read<uint8_t>(fixture+"/query.bin",uint64_t(b)*4*g*128);
    auto key=Read<uint8_t>(fixture+"/key.bin",uint64_t(b)*h*128);
    auto weight=Read<uint16_t>(fixture+"/weight.bin",uint64_t(b)*4*g);
    auto qs=Read<float>(fixture+"/qscale.bin",weight.size());
    auto ks=Read<float>(fixture+"/kscale.bin",uint64_t(b)*h);
    auto oracle=Read<uint16_t>(fixture+"/oracle.bin",uint64_t(b)*4*h);
    key.resize(key.size()+128,0x5a);uint64_t keyGm=m.Put(key),ksGm=m.Put(ks);
    std::vector<Wave> waves;
    for(uint32_t start=0;start<plan.tasks.size();start+=plan.capacity) {
        uint32_t rows=std::min(plan.capacity,uint32_t(plan.tasks.size())-start);
        std::vector<uint8_t> vq(uint64_t(rows)*qg*g*128);
        std::vector<uint16_t> vw(uint64_t(rows)*qg*g);
        std::vector<float> vqs(vw.size());
        std::vector<uint32_t> qlen(rows,qg),klen(rows);
        std::vector<int32_t> pages(uint64_t(rows)*plan.rowStride/128,0);
        for(uint32_t r=0;r<rows;++r) {
            auto t=plan.tasks[start+r];klen[r]=t.end-t.begin;
            uint64_t from=(uint64_t(t.owner)*4+t.queryBegin)*g;
            std::copy_n(query.begin()+from*128,qg*g*128,vq.begin()+uint64_t(r)*qg*g*128);
            std::copy_n(weight.begin()+from,qg*g,vw.begin()+uint64_t(r)*qg*g);
            std::copy_n(qs.begin()+from,qg*g,vqs.begin()+uint64_t(r)*qg*g);
            for(uint32_t p=0;p<klen[r]/128;++p)
                pages[uint64_t(r)*plan.rowStride/128+p]=t.owner*h/128+t.begin/128+p;
        }
        uint64_t score=m.Put(std::vector<uint16_t>(uint64_t(rows)*qg*plan.rowStride+128,0x5a5a));
        std::vector<uint64_t> binding(16,0);binding[0]=m.Put(vq);binding[1]=keyGm;
        binding[2]=m.Put(vw);binding[3]=m.Put(vqs);binding[4]=ksGm;
        binding[5]=m.Put(qlen);binding[6]=m.Put(klen);binding[7]=m.Put(pages);
        binding[9]=m.Put(std::vector<int32_t>(uint64_t(rows)*qg*2048+128,0x5a5a5a5a));
        binding[12]=score;waves.push_back({m.Put(binding),score,rows,start});
    }
    const bool native=std::string(SCORE_RUN_MODE)=="npu";
    std::vector<float> samples;std::vector<double> wall;
    for(uint32_t i=0;i<iterations;++i) {
        aclrtEvent first=nullptr,last=nullptr;
        if(native){Check(aclrtCreateEvent(&first));Check(aclrtCreateEvent(&last));Check(aclrtRecordEvent(first,stream));}
        auto begin=std::chrono::steady_clock::now();
        for(auto w:waves)Check(aclrtlaunch_split_score_only(w.rows,stream,(void*)w.binding,w.rows,plan.rowStride,g,qg));
        if(native)Check(aclrtRecordEvent(last,stream));
        Check(aclrtSynchronizeStream(stream));
        wall.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
        if(native){float ms=0;Check(aclrtEventElapsedTime(&ms,first,last));samples.push_back(ms);aclrtDestroyEvent(first);aclrtDestroyEvent(last);}
    }
    std::vector<uint16_t> flat(uint64_t(b)*4*h);
    for(auto wave:waves) {
        auto raw=m.Get<uint16_t>(wave.score,uint64_t(wave.rows)*qg*plan.rowStride+128);
        for(uint32_t r=0;r<wave.rows;++r) {
            auto t=plan.tasks[wave.begin+r];uint32_t length=t.end-t.begin;
            for(uint32_t q=0;q<qg;++q) {
                auto source=raw.begin()+(uint64_t(r)*qg+q)*plan.rowStride;
                std::copy_n(source,length,flat.begin()+(uint64_t(t.owner)*4+t.queryBegin+q)*h+t.begin);
                for(uint32_t x=length;x<plan.rowStride;++x)if(source[x]!=0x5a5a)throw std::runtime_error("row padding overwritten");
            }
        }
        for(size_t x=uint64_t(wave.rows)*qg*plan.rowStride;x<raw.size();++x)
            if(raw[x]!=0x5a5a)throw std::runtime_error("score allocation guard overwritten");
    }
    auto keyAfter=m.Get<uint8_t>(keyGm,key.size());if(keyAfter!=key)throw std::runtime_error("key input/guard changed");
    for(size_t i=0;i<flat.size();++i)if(flat[i]!=oracle[i])
        throw std::runtime_error("score key mismatch at "+std::to_string(i)+": "+std::to_string(flat[i])+" != "+std::to_string(oracle[i]));
    std::ofstream f(output+"/score-keys.bin",std::ios::binary);f.write((char*)flat.data(),flat.size()*2);
    if(!f)throw std::runtime_error("score output write");
    std::ofstream report(output+"/score-check.json");
    report<<"{\"status\":\"PASSED\",\"version\":\""<<SCORE_VERSION<<"\",\"run_mode\":\""<<SCORE_RUN_MODE
        <<"\",\"batch\":"<<b<<",\"history\":"<<h<<",\"heads\":"<<g<<",\"producers\":"<<plan.producers
        <<",\"query_group\":"<<qg<<",\"waves\":"<<waves.size()<<",\"iterations\":"<<iterations
        <<",\"oracle_keys\":"<<flat.size()<<",\"guards\":true,\"score_ms\":";
    if(native){std::sort(samples.begin(),samples.end());report<<samples[samples.size()/2];}else report<<"null";
    report<<",\"host_wall_ms\":[";for(size_t i=0;i<wall.size();++i){if(i)report<<",";report<<wall[i];}
    report<<"],\"scope\":\"score-only, no transport or streaming completion\"}\n";
    if(!report)throw std::runtime_error("report write");std::cout<<"SCORE_ORACLE_PASSED keys="<<flat.size()<<"\n";
}
int main(int argc,char** argv) {
    try {
        if(argc!=12)throw std::invalid_argument("score_routes FIXTURE OUTPUT B H G C V P QGROUP ITERATIONS DEVICE");
        auto number=[](const char* s){size_t used=0;auto v=std::stoull(s,&used);if(s[used] || v>UINT32_MAX)throw std::invalid_argument("invalid integer");return uint32_t(v);};
        auto p=kvip_score::MakePlan(number(argv[3]),number(argv[4]),number(argv[5]),number(argv[6]),number(argv[7]),number(argv[8]),number(argv[9]));
        uint32_t iterations=number(argv[10]);if(!iterations||iterations>100)throw std::invalid_argument("iterations 1..100");
        if(std::filesystem::exists(argv[2]))throw std::invalid_argument("output must be new");
        std::filesystem::create_directories(argv[2]);
        Check(aclInit(nullptr));Check(aclrtSetDevice(number(argv[11])));aclrtStream stream=nullptr;Check(aclrtCreateStream(&stream));
        Run(argv[1],argv[2],p,iterations,stream);
        Check(aclrtDestroyStream(stream));Check(aclrtResetDevice(number(argv[11])));Check(aclFinalize());return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
