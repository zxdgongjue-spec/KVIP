// NextGen V0: CPU-only compatibility and frozen P6 planner migration.
// This program does not load any NPU binary and does not certify device execution.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "b16_mtp_cpp_planner.cpp"
#include "runtime_shape.h"
namespace rotating {
#include "sequence_schedule.h"
}

namespace {
void require(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "V0_FAIL %s\n", why); std::exit(1); }
}
using Lengths = std::array<uint32_t,16>;
using Queries = std::array<uint32_t,16>;
using Visibility = std::array<uint32_t,64>;
Lengths fill(uint32_t value) { Lengths x{}; x.fill(value); return x; }
Lengths tiers(std::initializer_list<uint32_t> pattern) {
    require(pattern.size()>0 && 16%pattern.size()==0,"tier size");
    Lengths x{}; size_t i=0;
    for (size_t round=0;round<16/pattern.size();++round)
        for (auto v:pattern) x[i++]=v*1024;
    return x;
}
Visibility visibility(const Lengths& ks,const Queries& qs) {
    Visibility v{};
    for (size_t b=0;b<16;++b) {
        require(qs[b]<=4 && ks[b]>=qs[b],"input geometry");
        for (uint32_t q=0;q<qs[b];++q) v[b*4+q]=ks[b]-qs[b]+q+1;
    }
    return v;
}
void verify_plan(const MtpPacked& p,const Lengths& ks,const Visibility& vis) {
    require(p.n>=32 && p.n<=256 && p.offsets[0]==0 && p.offsets[32]==p.n,"task count/offsets");
    require(std::equal(vis.begin(),vis.end(),p.visible),"visibility");
    std::array<std::vector<std::array<uint32_t,3>>,32> rows;
    for (uint32_t c=0;c<32;++c) {
        require(p.offsets[c]<=p.offsets[c+1] && p.offsets[c+1]<=p.n,"core offsets");
        for (uint32_t i=p.offsets[c];i<p.offsets[c+1];++i) {
            const auto* t=p.tasks[i];
            require(t[0]<16 && t[1]<2 && t[2]<t[3] && t[3]<=ks[t[0]]/128,"task bounds");
            rows[2*t[0]+t[1]].push_back({t[2],t[3],i});
        }
    }
    require(p.row_offsets[0]==0 && p.row_offsets[64]==2*p.n,"row offsets");
    for (uint32_t g=0;g<32;++g) {
        auto& row=rows[g]; std::sort(row.begin(),row.end());
        require(!row.empty() && row.size()<=8,"shard count");
        uint32_t end=0;
        for (const auto& shard:row) {
            require(shard[0]==end,"coverage gap or overlap"); end=shard[1];
        }
        require(end==ks[g/2]/128,"physical coverage");
        for (uint32_t lane=0;lane<2;++lane) {
            uint32_t q=2*g+lane, lo=p.row_offsets[q], hi=p.row_offsets[q+1];
            require(lo<=hi && hi<=2*p.n && hi-lo==row.size(),"row gather length");
            for(uint32_t j=lo;j<hi;++j)
                require(p.row_ids[j]==2*row[j-lo][2]+lane,"row gather ownership/order");
        }
    }
}
void run(const char* name,const Lengths& ks,const Queries& qs,bool planner_supported) {
    auto v=visibility(ks,qs);
    uint32_t h=*std::max_element(ks.begin(),ks.end());
    rotating::RuntimeShape s{};
    s.batch=16;s.history=((h+255)/256)*256;s.resident=16384;s.indexHeads=32;
    s.aicCount=32;s.splitQ=1;s.rounds=8;s.tokenBytes=656;
    const auto rc=rotating::PrepareShape(&s);
    // Exact shape helper, then a source-derived mirror of host launch.cpp and
    // device HasUniformLengths. Not an execution trace or a device guarantee.
    bool host=rc==0 && s.batch<=16 && s.finalWorkers==4 &&
        s.batch*4<=s.aicCount*2 && rotating::UseSequenceSplit(s);
    bool uniform=true;uint32_t qsum=0;
    for(size_t b=0;b<16;++b){uniform=uniform && qs[b]==4 && ks[b]==s.history;qsum+=qs[b];}
    bool device=rc==0 && rotating::UseSequenceSplit(s) && uniform;
    std::printf("ROUTE_MODEL %s H=%u Q=%u shape_rc=%u host_parallel=%d device_split=%d mismatch=%d\n",
        name,s.history,qsum,rc,int(host),int(device),int(host&&!device));
    MtpPacked count{},adaptive{};
    int a=mtp_plan(ks.data(),v.data(),0,&count), b=mtp_plan(ks.data(),v.data(),1,&adaptive);
    if(!planner_supported){
        require(a==-1 && b==-1,"unsupported planner input must reject");
        std::printf("PLANNER_EXPECTED_REJECT %s rc=%d/%d\n",name,a,b);return;
    }
    require(a==0 && b==0,"planner return");verify_plan(count,ks,v);verify_plan(adaptive,ks,v);
    require(adaptive.predicted<=count.predicted+1e-8,"model cost regression");
    std::array<uint64_t,2000> samples{};
    require(mtp_bench(ks.data(),v.data(),1000,samples.data())==0,"planner CPU benchmark");
    std::sort(samples.begin(),samples.begin()+1000);std::sort(samples.begin()+1000,samples.end());
    std::printf("PLANNER_PASS %s label=%u mode=%u gate=%u tasks=%u count_max_tiles=%u adaptive_max_tiles=%u count_model=%.3f adaptive_model=%.3f cpu_count_p50_us=%.3f cpu_adaptive_p50_us=%.3f\n",
      name,adaptive.label,adaptive.mode,adaptive.gate,adaptive.n,count.features[2],adaptive.features[2],
      count.predicted,adaptive.predicted,(samples[499]+samples[500])/2000.0,
      (samples[1499]+samples[1500])/2000.0);
}
}
int main(){
    auto q4=fill(4);
    run("V01_uniform32k",fill(32768),q4,true);
    run("V02_uniform128k",fill(131072),q4,true);
    auto ratio=fill(32768);std::fill_n(ratio.begin(),8,131072);
    run("V03_ratio4",ratio,q4,true);
    run("V04_four_tier",tiers({16,32,64,128}),q4,true);
    auto rare=fill(32768);rare[0]=rare[1]=196608;
    run("V05_two192k",rare,q4,true);
    auto high=fill(131072);std::fill_n(high.begin(),8,524288);
    run("V06_eight512k",high,q4,true);
    auto extreme=fill(65536);extreme[0]=extreme[1]=524288;
    run("V07_two512k",extreme,q4,true);
    auto single=fill(32768);single[0]=524288;
    run("V08_one512k",single,q4,true);
    auto q56=q4;for(size_t i=0;i<16;i+=2)q56[i]=3;
    run("V09_q56",high,q56,true);
    auto q60=q4;for(size_t i=0;i<16;i+=4)q60[i]=3;
    run("V10_q60",extreme,q60,true);
    auto q0=q4;q0[0]=0;
    run("V11_q0",ratio,q0,false);
    auto q1=q4;q1[0]=1;
    run("V12_q1",ratio,q1,false);
    auto q55=q56;q55[1]=3;
    run("V13_q55",ratio,q55,false);
    auto q2=q4;for(size_t i=0;i<4;++i)q2[i]=2;
    run("V14_q56_q2",ratio,q2,true);
    std::puts("NEXTGEN_V0_HOST_CONTRACT_PASS");
}
