// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
extern "C" uint32_t aclrtlaunch_incremental_topk_kernel(uint32_t,void*,void*,void*,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
static void check(int code) { if(code) throw std::runtime_error("ACL error " + std::to_string(code)); }
template<class T> void* upload(const std::vector<T>& v) {
    void* p=nullptr; check(aclrtMalloc(&p,v.size()*sizeof(T),ACL_MEM_MALLOC_HUGE_FIRST));
    check(aclrtMemcpy(p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));return p;
}
template<class T> std::vector<T> read(void* p,size_t n) {
    std::vector<T> v(n);check(aclrtMemcpy(v.data(),n*sizeof(T),p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;
}
int main(int argc,char** argv) { try {
    uint32_t device=0,blocks=1,rows=8,added=12288,warmup=4,iterations=50,seed=43; bool sim=false; std::string inputFile; uint32_t mode=0,percent=100;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];if(a=="--sim"){sim=true;continue;}
        if(i+1==argc)throw std::runtime_error("missing value: "+a);
        if(a=="--mode"){std::string m=argv[++i];if(m!="dense"&&m!="bitmap"&&m!="compact")throw std::runtime_error("invalid mode");mode=m=="bitmap"?1:m=="compact"?2:0;continue;}
        if(a=="--input"){inputFile=argv[++i];continue;}
        auto v=std::stoul(argv[++i]);
        if(a=="--device")device=v;else if(a=="--blocks")blocks=v;else if(a=="--rows")rows=v;
        else if(a=="--new-scores")added=v;else if(a=="--warmup")warmup=v;
        else if(a=="--iterations")iterations=v;else if(a=="--seed")seed=v;else if(a=="--candidate-percent")percent=v;
        else throw std::runtime_error("unknown option: "+a);
    }
    if(!percent||percent>100)throw std::runtime_error("candidate-percent:1..100");
    if(!blocks||blocks>64||!rows||rows>64||!added||added>16384||added%256||!iterations)
        throw std::runtime_error("blocks/rows:1..64, new-scores:256..16384 multiple of256, iterations>0");
    constexpr uint32_t K=2048,oldHistory=16384;
    uint32_t n=blocks*rows,width=K+added;
    // Finite, sortable BF16 keys; same representation as LI. Previous TopK is
    // selected from 16K history, rather than arbitrary 2048 random elements.
    std::mt19937 rng(seed);std::uniform_int_distribution<uint32_t> random(0x4000,0xc600);
    std::vector<uint16_t> full;
    if(!inputFile.empty()){
        full.resize(size_t(n)*(oldHistory+added)); std::ifstream file(inputFile,std::ios::binary);
        file.read(reinterpret_cast<char*>(full.data()),full.size()*2);
        if(!file || file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("input must contain rows*blocks*(16384+new-scores) uint16 keys");
    }
    std::vector<uint16_t> source(n*width);std::vector<uint32_t> oldIds(n*K);
    for(uint32_t row=0;row<n;++row) {
        std::vector<uint16_t> history(oldHistory);for(uint32_t i=0;i<oldHistory;++i)history[i]=full.empty()?random(rng):full[size_t(row)*(oldHistory+added)+i];
        std::vector<uint32_t> ids(oldHistory);std::iota(ids.begin(),ids.end(),0);
        std::partial_sort(ids.begin(),ids.begin()+K,ids.end(),[&](auto a,auto b){return history[a]>history[b];});
        for(uint32_t i=0;i<K;++i){source[row*width+i]=history[ids[i]];oldIds[row*K+i]=ids[i];}
        for(uint32_t i=K;i<width;++i)source[row*width+i]=full.empty()?random(rng):full[size_t(row)*(oldHistory+added)+oldHistory+i-K];
    }
    const uint32_t candidates=(added*percent+50)/100,padded=(candidates+255)/256*256;
    std::vector<uint32_t> masks(n*added/32,0);
    // Randomly placed eligible entries. All omitted scores are below the old
    // TopK minimum, so full-input TopK remains an independent exact reference.
    if(mode){
        for(uint32_t row=0;row<n;++row){
            std::vector<uint32_t> order(added);std::iota(order.begin(),order.end(),0);
            std::shuffle(order.begin(),order.end(),rng);
            for(uint32_t i=0;i<candidates;++i)masks[row*(added/32)+order[i]/32]|=1U<<(order[i]%32);
            const uint16_t threshold=source[row*width+K-1];
            if(!threshold)throw std::runtime_error("fixture requires positive sortable old threshold");
            for(uint32_t i=0;i<added;++i){auto& score=source[row*width+K+i];
                if(!(masks[row*(added/32)+i/32]&(1U<<(i%32))))score=threshold-1;
                else score=std::max(score,threshold);}
        }
    }
    auto deviceScores=source;auto deviceIds=oldIds;
    if(mode==2){
        deviceScores.assign(n*(K+padded),0);deviceIds.assign(n*(K+padded),0);
        for(uint32_t row=0;row<n;++row){uint32_t out=row*(K+padded);
            for(uint32_t i=0;i<K;++i){deviceScores[out+i]=source[row*width+i];deviceIds[out+i]=oldIds[row*K+i];}
            uint32_t rank=K;
            for(uint32_t i=0;i<added;++i)if(masks[row*(added/32)+i/32]&(1U<<(i%32))){
                deviceScores[out+rank]=source[row*width+K+i];deviceIds[out+rank++]=oldHistory+i;}
        }
    }
    std::printf("CASE mode=%s candidates=%u/%u percent=%u padded=%u blocks=%u rows/AIV=%u\n",mode==1?"bitmap":mode==2?"compact":"dense",candidates,added,percent,padded,blocks,rows);
    check(aclInit(nullptr));check(aclrtSetDevice(device));aclrtStream stream;check(aclrtCreateStream(&stream));
    void* s=upload(deviceScores);void* old=upload(deviceIds);void* mask=upload(masks);void* values=upload(std::vector<uint16_t>(n*K));
    void* ids=upload(std::vector<uint32_t>(n*K));void* ticks=upload(std::vector<uint64_t>(n*5));
    auto launch=[&](uint32_t profile){check(aclrtlaunch_incremental_topk_kernel(blocks,stream,s,old,values,ids,ticks,mask,rows,added,profile,mode,candidates));};
    auto verify=[&](){
        auto actual=read<uint16_t>(values,n*K);auto actualIds=read<uint32_t>(ids,n*K);
        for(uint32_t row=0;row<n;++row) {
            std::vector<uint16_t> expected(source.begin()+row*width,source.begin()+(row+1)*width);
            std::sort(expected.begin(),expected.end(),std::greater<uint16_t>());expected.resize(K);
            std::vector<uint16_t> got(actual.begin()+row*K,actual.begin()+(row+1)*K);
            std::sort(got.begin(),got.end(),std::greater<uint16_t>());
            if(got!=expected)throw std::runtime_error("TopK score multiset mismatch row="+std::to_string(row));
            std::vector<int> pos(oldHistory+added,-1);
            for(uint32_t i=0;i<K;++i)pos[oldIds[row*K+i]]=i;
            for(uint32_t i=0;i<added;++i)pos[oldHistory+i]=K+i;
            for(uint32_t i=0;i<K;++i){uint32_t id=actualIds[row*K+i];
                if(id>=pos.size()||pos[id]<0||source[row*width+pos[id]]!=actual[row*K+i])
                    throw std::runtime_error("TopK ID/value/uniqueness mismatch row="+std::to_string(row));
                pos[id]=-1;
            }
        }
        std::printf("CORRECT rows=%u TopK scores and original token IDs PASS\n",n);
    };
    if(!sim) {
        launch(0);check(aclrtSynchronizeStream(stream));verify();
        aclmdlRI graph;check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));launch(0);
        check(aclmdlRICaptureEnd(stream,&graph));
        aclrtEvent start,end;check(aclrtCreateEvent(&start));check(aclrtCreateEvent(&end));
        std::vector<double> times;
        for(uint32_t i=0;i<warmup+iterations;++i){
            check(aclrtRecordEvent(start,stream));check(aclmdlRIExecuteAsync(graph,stream));check(aclrtRecordEvent(end,stream));
            check(aclrtSynchronizeStream(stream));float ms;check(aclrtEventElapsedTime(&ms,start,end));
            if(i>=warmup)times.push_back(ms*1000.0);
        }
        verify();std::sort(times.begin(),times.end());
        auto percentile=[&](double q){double at=q*(times.size()-1);size_t low=at;return times[low]+(times[std::min(low+1,times.size()-1)]-times[low])*(at-low);};
        std::printf("GRAPH uninstrumented blocks=%u rows/AIV=%u old=%u new=%u mean_us=%.2f p50_us=%.2f p99_us=%.2f\n",
            blocks,rows,K,added,std::accumulate(times.begin(),times.end(),0.)/times.size(),percentile(.5),percentile(.99));
        check(aclrtDestroyEvent(start));check(aclrtDestroyEvent(end));check(aclmdlRIDestroy(graph));
    }
    launch(1);check(aclrtSynchronizeStream(stream));verify();auto trace=read<uint64_t>(ticks,n*5);
    std::printf("SEPARATE PROFILE native GetSystemCycle ticks; no assumed clock conversion\nblock,row,load_ticks,select_ticks,map_ticks,store_ticks,total_ticks\n");
    for(uint32_t row=0;row<n;++row){auto t=trace.data()+row*5;
        std::printf("%u,%u,%llu,%llu,%llu,%llu,%llu\n",row/rows,row%rows,
            (unsigned long long)(t[1]-t[0]),(unsigned long long)(t[2]-t[1]),
            (unsigned long long)(t[3]-t[2]),(unsigned long long)(t[4]-t[3]),(unsigned long long)(t[4]-t[0]));
    }
    for(uint32_t b=0;b<blocks;++b)std::printf("BLOCK %u span_ticks=%llu\n",b,
        (unsigned long long)(trace[((b+1)*rows-1)*5+4]-trace[b*rows*5]));
    for(void* p:{s,old,values,ids,ticks,mask})check(aclrtFree(p));
    check(aclrtDestroyStream(stream));check(aclrtResetDevice(device));check(aclFinalize());return 0;
} catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
