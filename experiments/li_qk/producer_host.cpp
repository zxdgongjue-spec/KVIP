// Reuse fixture/ACL/oracle helpers without altering the existing LI entrypoint.
#define main li_controls_unused_main
#include "host.cpp"
#undef main
#include "producer_plan.h"
extern "C" uint32_t aclrtlaunch_li_qk_score_producer(uint32_t,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_li_producer_high_tail(uint32_t,void*,void*);
static void ProducerRun(const std::string& fixture,const std::string& output,uint32_t route,uint32_t warmup,uint32_t iterations,aclrtStream stream){
    constexpr uint32_t h=131072,g=32,p=16,shard=8192;
    Memory m;auto query=Read<uint8_t>(fixture+"/query.bin",4*g*128);
    auto key=Read<uint8_t>(fixture+"/key.bin",uint64_t(h)*128);
    auto weight=Read<uint16_t>(fixture+"/weight.bin",4*g);
    auto qs=Read<float>(fixture+"/qscale.bin",4*g),ks=Read<float>(fixture+"/kscale.bin",h);
    auto oracle=Read<uint16_t>(fixture+"/oracle.bin",4*h);
    auto keyGm=m.Put(key,true),ksGm=m.Put(ks,true);
    std::vector<uint8_t> q(p*4*g*128);std::vector<uint16_t> w(p*4*g);std::vector<float> z(p*4*g);
    std::vector<int32_t> pages(p*shard/128);std::vector<li_qk::Task> tasks;
    for(uint32_t r=0;r<p;++r){std::copy(query.begin(),query.end(),q.begin()+r*query.size());std::copy(weight.begin(),weight.end(),w.begin()+r*weight.size());std::copy(qs.begin(),qs.end(),z.begin()+r*qs.size());
        for(uint32_t x=0;x<shard/128;++x)pages[r*shard/128+x]=r*shard/128+x;tasks.push_back({0,4,r*shard,(r+1)*shard,0});}
    auto scratch=m.Put(std::vector<uint16_t>(4*h+128,0x5a5a));
    std::vector<uint64_t> bind(16,0);bind[0]=m.Put(q,true);bind[1]=keyGm;bind[2]=m.Put(w,true);bind[3]=m.Put(z,true);bind[4]=ksGm;
    bind[5]=m.Put(std::vector<uint32_t>(p,4),true);bind[6]=m.Put(std::vector<uint32_t>(p,shard),true);bind[7]=m.Put(pages,true);
    bind[9]=m.Put(std::vector<int32_t>(p*4*2048+128,-7));bind[12]=scratch;auto binding=m.Put(bind,true);
    auto canonical=m.Put(std::vector<uint16_t>(4*h+128,0x5a5a)),ids=m.Put(std::vector<int32_t>(4*2048+128,-7));
    rotating::Context ctx{};ctx.shape.batch=1;ctx.shape.history=ctx.shape.resident=h;ctx.shape.indexHeads=g;auto context=m.Put(std::vector<rotating::Context>{ctx},true);
    li_producer::Bindings a{};a.canonical=canonical;a.tasks=m.Put(tasks,true);a.selected=ids;a.history=h;a.route=route;
    if(route==2)a.high=m.Put(std::vector<uint32_t>(4*p*256+128,0x5a5a5a5a));
    auto sidecar=m.Put(std::vector<li_producer::Bindings>{a},true);
    bool native=std::string(LI_QK_RUN_MODE)=="npu";aclrtEvent start=nullptr,end=nullptr;
    if(native){Check(aclrtCreateEvent(&start));Check(aclrtCreateEvent(&end));}
    std::vector<double> samples,scoreSamples,tailSamples;
    auto score=[&](){if(!route)Check(aclrtlaunch_li_qk_score(p,stream,(void*)binding,p,shard,g,4));else Check(aclrtlaunch_li_qk_score_producer(p,stream,(void*)binding,(void*)sidecar,p,shard,g,4));};
    auto tail=[&](){if(!route)Check(aclrtlaunch_li_qk_scatter(p,stream,(void*)scratch,(void*)canonical,(void*)a.tasks,p,shard,h,4));if(route==2)Check(aclrtlaunch_li_producer_high_tail(4,stream,(void*)sidecar));else Check(aclrtlaunch_li_qk_topk(4,stream,(void*)canonical,(void*)ids,(void*)context,h,g,4));};
    auto call=[&](uint32_t phase,bool measured){if(native&&measured)Check(aclrtRecordEvent(start,stream));if(phase!=2)score();if(phase!=1)tail();if(native&&measured)Check(aclrtRecordEvent(end,stream));Check(aclrtSynchronizeStream(stream));
        if(native&&measured){float ms;Check(aclrtEventElapsedTime(&ms,start,end));Require(ms>0,"device time");(phase==1?scoreSamples:phase==2?tailSamples:samples).push_back(ms*1000.0);}};
    for(uint32_t i=0;i<warmup;++i)call(0,false);
    for(uint32_t i=0;i<iterations;++i)call(0,true);
    if(native){for(uint32_t phase=1;phase<=2;++phase){for(uint32_t i=0;i<warmup;++i)call(phase,false);for(uint32_t i=0;i<iterations;++i)call(phase,true);}Check(aclrtDestroyEvent(start));Check(aclrtDestroyEvent(end));}
    auto flat=m.Get<uint16_t>(canonical,4*h+128);Require(std::equal(oracle.begin(),oracle.end(),flat.begin()),"canonical oracle");for(uint32_t x=4*h;x<flat.size();++x)Require(flat[x]==0x5a5a,"canonical guard");
    auto raw=m.Get<uint16_t>(scratch,4*h+128);if(route)Require(std::all_of(raw.begin(),raw.end(),[](uint16_t x){return x==0x5a5a;}),"direct must not write scratch");
    if(route==2){auto high=m.Get<uint32_t>(a.high,4*p*256+128);for(uint32_t q=0;q<4;++q)for(uint32_t part=0;part<p;++part){std::vector<uint32_t> count(256);for(uint32_t x=part*shard;x<(part+1)*shard;++x)++count[oracle[q*h+x]>>8];uint32_t sum=0;for(uint32_t b=0;b<256;++b){sum+=count[b];Require(high[(q*p+part)*256+b]==sum,"high histogram oracle");}}for(uint32_t x=4*p*256;x<high.size();++x)Require(high[x]==0x5a5a5a5a,"high guard");}
    auto selected=m.Get<int32_t>(ids,4*2048+128);for(uint32_t q=0;q<4;++q){std::vector<uint16_t> sorted(oracle.begin()+q*h,oracle.begin()+(q+1)*h);std::nth_element(sorted.begin(),sorted.begin()+2047,sorted.end(),std::greater<uint16_t>());uint16_t tau=sorted[2047];std::set<uint32_t> got;
        for(uint32_t k=0;k<2048;++k){int32_t t=selected[q*2048+k];Require(t>=0&&uint32_t(t)<h&&got.insert(t).second,"TopK bounds/unique");Require(oracle[q*h+t]>=tau,"TopK below threshold");}for(uint32_t t=0;t<h;++t)if(oracle[q*h+t]>tau)Require(got.count(t),"mandatory GT");}
    for(uint32_t x=4*2048;x<selected.size();++x)Require(selected[x]==-7,"TopK guard");m.CheckInputs();
    std::ofstream f(output+"/score-keys.bin",std::ios::binary);f.write((char*)flat.data(),4*h*2);std::ofstream k(output+"/topk.bin",std::ios::binary);k.write((char*)selected.data(),4*2048*4);
    std::ofstream report(output+"/report.json");report<<std::setprecision(10)<<"{\"status\":\"PASS\",\"run_mode\":\""<<LI_QK_RUN_MODE<<"\",\"batch\":1,\"queries\":4,\"history\":131072,\"heads\":32,\"parts\":16,\"route\":"<<route<<",\"score_aics\":16,\"score_aivs\":32,\"tail_aivs\":4,\"score_launches\":1,\"scatter_launches\":"<<(route?0:1)<<",\"tail_launches\":1,\"full_li_launches\":"<<(route?2:3)<<",\"warmup\":"<<warmup<<",\"iterations\":"<<iterations<<",\"oracle\":true,\"guards\":true,\"inputs_unchanged\":true,\"fetch_launches\":0";
    auto metric=[&](const char* name,const std::vector<double>& values){report<<",\""<<name<<"_p50_us\":";if(values.empty())report<<"null";else report<<Median(values);report<<",\""<<name<<"_samples_us\":[";for(size_t i=0;i<values.size();++i){if(i)report<<",";report<<values[i];}report<<"]";};metric("li",samples);metric("score",scoreSamples);metric("tail",tailSamples);report<<"}\n";std::cout<<"PRODUCER_ORACLE_PASS route="<<route<<"\n";
}
int main(int argc,char** argv){try{if(argc==2&&std::string(argv[1])=="--capabilities"){std::cout<<"{\"run_mode\":\""<<LI_QK_RUN_MODE<<"\",\"scope\":\"B1 H128K P16 producer controls\"}\n";return 0;}Require(argc==7,"li_producer FIXTURE OUTPUT ROUTE WARMUP ITERATIONS DEVICE");uint32_t route=std::stoul(argv[3]),warm=std::stoul(argv[4]),it=std::stoul(argv[5]);Require(route<=2&&it>=1&&it<=1000&&warm<=100,"route0/1/2 and samples");Require(warm||std::string(LI_QK_RUN_MODE)=="sim","NPU warmup required");Require(!std::filesystem::exists(argv[2]),"fresh output");std::filesystem::create_directories(argv[2]);Check(aclInit(nullptr));Check(aclrtSetDevice(std::stoul(argv[6])));aclrtStream s;Check(aclrtCreateStream(&s));ProducerRun(argv[1],argv[2],route,warm,it,s);Check(aclrtDestroyStream(s));Check(aclrtResetDevice(std::stoul(argv[6])));Check(aclFinalize());return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
