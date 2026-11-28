// LB-P5b1: exact-equivalent fast path for the frozen B16/Q4 planner.
// No CANN, NPU, dynamic allocation, H2D or production integration.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>

namespace {
constexpr int B=16,C=32,Q=4,K=2048,TILE=128,MIN=16,MAX=256;
enum Label { Count, Split2, Split4, Split8, Fill, Equal, General };
struct Task { uint32_t b,g,lo,hi; };
static_assert(sizeof(Task)==16);
int size(Task t){return int(t.hi-t.lo);}
bool less_task(Task a,Task b){
    if(size(a)!=size(b))return size(a)>size(b);
    return std::tie(a.b,a.g,a.lo,a.hi)<std::tie(b.b,b.g,b.lo,b.hi);
}
bool eq(Task a,Task b){return a.b==b.b&&a.g==b.g&&a.lo==b.lo&&a.hi==b.hi;}
struct Plan {
    // Only the live prefix is initialized/copied; inactive slots are never read.
    std::array<Task,MAX> t;
    std::array<int,MAX> next;
    std::array<int,C> head{},tail{},load{},trunks{},nt{};
    std::array<uint32_t,C> shards{};
    uint32_t candidates=0,max_shards=0,merge_rows=0,max_load=0,max_trunks=0,max_nt=0;
    int n=0;
    Plan(){head.fill(-1);tail.fill(-1);}
    Plan(const Plan& x):Plan(){*this=x;}
    Plan& operator=(const Plan& x){
        if(this==&x)return *this;
        n=x.n;std::copy_n(x.t.begin(),n,t.begin());std::copy_n(x.next.begin(),n,next.begin());
        head=x.head;tail=x.tail;load=x.load;trunks=x.trunks;nt=x.nt;shards=x.shards;
        candidates=x.candidates;max_shards=x.max_shards;merge_rows=x.merge_rows;
        max_load=x.max_load;max_trunks=x.max_trunks;max_nt=x.max_nt;
        return *this;
    }
    bool push(int c,Task x){
        if(n>=MAX||c<0||c>=C||x.b>=B||x.g>=2||x.lo>=x.hi)return false;
        t[n]=x;next[n]=-1;
        if(tail[c]>=0)next[tail[c]]=n;else head[c]=n;
        tail[c]=n++;load[c]+=size(x);trunks[c]+=(size(x)+63)/64;++nt[c];
        max_load=std::max(max_load,uint32_t(load[c]));
        max_trunks=std::max(max_trunks,uint32_t(trunks[c]));
        max_nt=std::max(max_nt,uint32_t(nt[c]));
        uint32_t s=++shards[x.b*2+x.g];
        max_shards=std::max(max_shards,s);if(s==2)merge_rows+=2;
        candidates+=2*std::min(K,size(x)*TILE);
        return true;
    }
};
bool same(const Plan&a,const Plan&b){
    if(a.n!=b.n||a.max_load!=b.max_load||a.max_shards!=b.max_shards||a.nt!=b.nt||a.load!=b.load)return false;
    for(int c=0;c<C;c++){
        int j=b.head[c];
        for(int i=a.head[c];i>=0;i=a.next[i],j=b.next[j])
            if(j<0||!eq(a.t[i],b.t[j]))return false;
        if(j>=0)return false;
    }return true;
}
using Lens=std::array<int,B>;
int ceildiv(int a,int b){return (a+b-1)/b;}
int sum(const Lens& l){int s=0;for(int x:l)s+=x;return s;}
bool read_lens(const uint32_t* in,Lens& l){
    if(!in)return false;
    for(int b=0;b<B;b++){
        if(in[b]<K||in[b]>524288||in[b]%TILE)return false;
        l[b]=int(in[b]/TILE);
    }return true;
}
std::array<Task,C> original(const Lens&l){
    std::array<Task,C> a{};
    for(int i=0;i<C;i++)a[i]={uint32_t(i/2),uint32_t(i%2),0,uint32_t(l[i/2])};
    return a;
}
Plan base_plan(const Lens&l){
    Plan p;auto a=original(l);for(int i=0;i<C;i++)p.push(i,a[i]);return p;
}
Plan split_plan(const Lens&l,int cap,bool equal){
    std::array<Task,MAX> ts;int n=0,total=sum(l);
    for(auto t:original(l)){
        int k=size(t),s=equal?(k>=2*MIN?2:1):(k*B>total?std::max(1,std::min(cap,k/MIN)):1);
        for(int j=0;j<s;j++)ts[n++]={t.b,t.g,uint32_t(k*j/s),uint32_t(k*(j+1)/s)};
    }
    std::sort(ts.begin(),ts.begin()+n,less_task);Plan p;
    for(int j=0;j<n;j++){
        int c=int(std::min_element(p.load.begin(),p.load.end())-p.load.begin());
        p.push(c,ts[j]);
    }return p;
}
bool paired(const Lens&l,Plan&p){
    int lo=*std::min_element(l.begin(),l.end()),hi=*std::max_element(l.begin(),l.end());
    if(lo==hi)return false;
    int ns=0,nl=0;std::array<int,B> ss{},ll{};
    for(int b=0;b<B;b++){if(l[b]==lo)ss[ns++]=b;else if(l[b]==hi)ll[nl++]=b;else return false;}
    if(ns!=8||nl!=8)return false;
    int target=(lo+hi+1)/2;if(hi-target<MIN)return false;
    for(int i=0;i<16;i++){
        uint32_t b=ll[i/2],s=ss[i/2],g=i%2;
        p.push(i,{b,g,0,uint32_t(target)});
        p.push(i+16,{s,g,0,uint32_t(lo)});
        p.push(i+16,{b,g,uint32_t(target),uint32_t(hi)});
    }return true;
}
bool eight_eight(const Lens&l){
    int lo=*std::min_element(l.begin(),l.end()),hi=*std::max_element(l.begin(),l.end());
    return lo!=hi&&std::count(l.begin(),l.end(),lo)==8&&std::count(l.begin(),l.end(),hi)==8;
}
bool capacity(const std::array<Task,C>&a,int target,int cap,Plan&p){
    std::array<int,C> free;free.fill(target);
    for(auto t:a)if(size(t)<=target){
        int c=int(std::max_element(free.begin(),free.end())-free.begin());
        if(free[c]<size(t)||!p.push(c,t))return false;
        free[c]-=size(t);
    }
    for(auto t:a)if(size(t)>target){
        int length=size(t),begin=0,num=0;uint32_t used=0;
        while(begin<length){
            if(num>=cap)return false;
            int rem=length-begin,best=-1,largest=-1;
            for(int c=0;c<C;c++)if(!(used&(uint32_t(1)<<c))&&free[c]>=MIN){
                if(largest<0||free[c]>free[largest])largest=c;
                if(free[c]>=rem&&(best<0||free[c]<free[best]))best=c;
            }
            if(largest<0)return false;
            int c=best>=0?best:largest;
            int take=best>=0?rem:std::min(free[c],rem-MIN);
            if(take<MIN||!p.push(c,{t.b,t.g,uint32_t(begin),uint32_t(begin+take)}))return false;
            free[c]-=take;used|=uint32_t(1)<<c;num++;begin+=take;
        }
    }
    return std::all_of(p.nt.begin(),p.nt.end(),[](int n){return n>0;});
}
struct Features { uint32_t tasks,max_tiles,max_tasks,max_trunks,candidates,max_shards,merge_rows; };
Features features(const Plan&p){
    return {uint32_t(p.n),p.max_load,p.max_nt,p.max_trunks,p.candidates,p.max_shards,p.merge_rows};
}
double estimate(Features f,const Lens&l){
    double v=25.28977509043643;
    v+=0.16685339451843886*f.max_tiles;
    v+=0.015531602670248825*(double(sum(l))/B);
    v+=1.7811570487347388*f.max_trunks;
    v+=8.100773020100553*int(f.max_shards>1);
    v+=1.1815789344768906*(int(f.max_shards)-1);
    v+=12.399691064711648*(double(f.merge_rows)/(B*Q));
    return v;
}
struct Candidates {
    std::array<Plan,7> p{};
    std::array<int,7> label{};
    int n=0;
    bool contains(const Plan&q)const{for(int i=0;i<n;i++)if(same(p[i],q))return true;return false;}
    void add(int k,const Plan&q){if(!contains(q)){p[n]=q;label[n++]=k;}}
};
Candidates generate(const Lens&l,bool general){
    Candidates a;a.add(Count,base_plan(l));
    for(int cap:{2,4,8})a.add(cap==2?Split2:cap==4?Split4:Split8,split_plan(l,cap,false));
    if(eight_eight(l)){
        Plan p;if(paired(l,p))a.add(Fill,p);
        a.add(Equal,split_plan(l,2,true));
    }
    if(!general)return a;
    int lower=ceildiv(sum(l),B),upper=*std::max_element(l.begin(),l.end());
    // Preserve Python's first-occurrence tie break and dedup across targets.
    std::array<Plan,12> seen;int ns=0;Plan best;bool have=false;
    std::tuple<double,int,int,int> key{};
    auto ordered=original(l);std::sort(ordered.begin(),ordered.end(),less_task);
    // Capacity uses cap solely as an early-abort condition. For a fixed
    // target, once a smaller cap succeeds, a larger cap yields the same
    // deterministic queue. Failed attempts are reusable only at same cap.
    std::array<int,12> attempted_target{},attempted_cap{};
    std::array<bool,12> succeeded{};int attempts=0;
    for(int cap:{2,4,8})for(int pct:{0,5,10,20}){
        int target=std::max(ceildiv(lower*(100+pct),100),ceildiv(upper,cap));
        if(target>=upper)continue;
        bool skip=false;
        for(int j=0;j<attempts;j++)if(attempted_target[j]==target&&
            (succeeded[j]||attempted_cap[j]==cap)){skip=true;break;}
        if(skip)continue;
        Plan p;bool ok=capacity(ordered,target,cap,p);
        attempted_target[attempts]=target;attempted_cap[attempts]=cap;succeeded[attempts++]=ok;
        if(!ok||a.contains(p))continue;
        bool duplicate=false;for(int i=0;i<ns;i++)if(same(seen[i],p)){duplicate=true;break;}
        if(duplicate)continue;
        seen[ns++]=p;
        auto k=std::make_tuple(estimate(features(p),l),p.n,target,cap);
        if(!have||k<key){best=p;key=k;have=true;}
    }
    if(have) a.add(General,best);
    return a;
}
// A sufficient reject condition for the *frozen* model, not a new admission rule.
int gate(const Lens&l,int mode){
    const int upper=*std::max_element(l.begin(),l.end()),lower=ceildiv(sum(l),B);
    // All feasible queues have max_tiles >= ceil(total tiles / 32).
    if(upper-lower<128)return 1;
    // At least one split group implies >=2 merge rows, >=2 max shards;
    // local rounds on the heaviest core >= ceil(lower/64).
    const Features base{32,uint32_t(upper),1,uint32_t(ceildiv(upper,64)),0,1,0};
    const Features optimistic{0,uint32_t(lower),0,uint32_t(ceildiv(lower,64)),0,2,2};
    const double t0=estimate(base,l),tmin=estimate(optimistic,l);
    const double required=std::max(2.,.05*t0)+(mode==1?0.:5.);
    // Conservative floating-point guard: never reject near equality.
    if(t0-tmin+1e-8<required)return 2;
    return 0;
}
int choose(const Candidates&a,const Lens&l,int mode){
    auto f0=features(a.p[0]);double base=estimate(f0,l),best=base;int sel=0;
    for(int i=1;i<a.n;i++){
        if(mode!=2&&a.label[i]==General)continue;
        auto f=features(a.p[i]);double t=estimate(f,l);
        if(f.max_shards>1&&int(f0.max_tiles)-int(f.max_tiles)>=128&&
           base-t>=std::max(2.,.05*base)+(mode==0?0.:5.)&&t<best){best=t;sel=i;}
    }return sel;
}
} // namespace

// Flat task ABI and row-owned candidate gather table; fixed upper bounds.
// Byte layout is explicitly verified by the Python harness.
extern "C" {
struct B16Packed {
    uint32_t label,n;
    uint32_t offsets[33],tasks[256][4],row_offsets[65],row_ids[512];
    uint32_t features[7];
    double prediction;
};
}
namespace {
bool pack(const Plan&p,const Lens&l,int label,B16Packed&out){
    out.label=label;out.n=p.n;
    std::array<std::array<int,8>,C> ids{};std::array<int,C> counts{};
    uint32_t j=0;out.offsets[0]=0;
    for(int c=0;c<C;c++){
        for(int i=p.head[c];i>=0;i=p.next[i]){
            Task t=p.t[i];int g=t.b*2+t.g;
            if(t.b>=B||t.g>=2||t.hi>uint32_t(l[t.b])||size(t)<MIN||counts[g]>=8||j>=MAX)return false;
            out.tasks[j][0]=t.b;out.tasks[j][1]=t.g;out.tasks[j][2]=t.lo;out.tasks[j][3]=t.hi;
            ids[g][counts[g]++]=j++;
        }out.offsets[c+1]=j;
        if(!p.nt[c])return false;
    }
    if(j!=uint32_t(p.n))return false;
    uint32_t r=0;out.row_offsets[0]=0;
    for(int g=0;g<C;g++){
        int n=counts[g];if(n==0)return false;
        std::sort(ids[g].begin(),ids[g].begin()+n,[&](int a,int b){return out.tasks[a][2]<out.tasks[b][2];});
        uint32_t end=0;
        for(int i=0;i<n;i++){auto t=out.tasks[ids[g][i]];if(t[2]!=end)return false;end=t[3];}
        if(end!=uint32_t(l[g/2]))return false;
        for(int lane=0;lane<2;lane++){
            for(int i=0;i<n;i++)out.row_ids[r++]=2*ids[g][i]+lane;
            out.row_offsets[2*g+lane+1]=r;
        }
    }
    if(r!=2*j)return false;
    Features f=features(p);uint32_t fs[]={f.tasks,f.max_tiles,f.max_tasks,f.max_trunks,f.candidates,f.max_shards,f.merge_rows};
    std::copy(fs,fs+7,out.features);out.prediction=estimate(f,l);return true;
}
}
extern "C" {
// out has seven slots; selected has three labels (V0,V1,V1_general).
int b16_all(const uint32_t*in,B16Packed*out,uint32_t*selected){
    Lens l{};if(!out||!selected||!read_lens(in,l))return -1;
    auto a=generate(l,true);
    for(int i=0;i<a.n;i++)if(!pack(a.p[i],l,a.label[i],out[i]))return -2;
    for(int i=0;i<3;i++)selected[i]=a.label[choose(a,l,i)];
    return a.n;
}
// mode 0 baseline, 1 V0, 2 V1 general. Includes validation and packing.
__attribute__((noinline)) int b16_plan(const uint32_t*in,int mode,B16Packed*out){
    Lens l{};if(!out||mode<0||mode>2||!read_lens(in,l))return -1;
    if(mode==0||gate(l,mode))return pack(base_plan(l),l,Count,*out)?0:-2;
    auto a=generate(l,mode==2);int i=choose(a,l,mode==1?0:2);
    return pack(a.p[i],l,a.label[i],*out)?0:-2;
}
__attribute__((noinline)) int b16_cache_lookup(const uint32_t*in,const uint32_t*key,const B16Packed*cached,const B16Packed**out){
    if(!in||!key||!cached||!out)return -1;
    *out=nullptr;
    if(!std::equal(in,in+B,key))return 1;
    *out=cached;return 0;
}
// Native loop: interleave modes, steady_clock, exact key lookup only (no copy/H2D).
// output is [4][samples], nanoseconds. Single caller, no device access.
int b16_bench(const uint32_t*in,int samples,uint64_t*out){
    if(!out||samples<1||samples>100000)return -1;
    B16Packed cached{},tmp{};if(b16_plan(in,2,&cached))return -2;
    std::array<uint32_t,B> key{};std::copy(in,in+B,key.begin());
    volatile uint64_t check=0;
    for(int i=0;i<samples+32;i++)for(int j=0;j<4;j++){
        int mode=(i+j)%4;const B16Packed*res=&tmp;
        auto t=std::chrono::steady_clock::now();
        int rc=mode==3?b16_cache_lookup(in,key.data(),&cached,&res):b16_plan(in,mode,&tmp);
        auto e=std::chrono::steady_clock::now();
        if(rc||!res)return -3;
        check=check+res->n+res->label;
        if(i>=32)out[mode*samples+i-32]=std::chrono::duration_cast<std::chrono::nanoseconds>(e-t).count();
    }
    return check?0:-4;
}
}

extern "C" int b16_fast_gate(const uint32_t*in,int mode){
    Lens l{};if(mode<1||mode>2||!read_lens(in,l))return -1;
    return gate(l,mode);
}
// Same-process alternating old/new benchmark; reference function from pinned SO.
// Compare active descriptors, not unused tail bytes or structure padding.
using PlanFn=int(*)(const uint32_t*,int,B16Packed*);
static bool packed_equal(const B16Packed&a,const B16Packed&b){
    if(a.label!=b.label||a.n!=b.n||a.n>256)return false;
    return std::equal(a.offsets,a.offsets+33,b.offsets)&&
        std::memcmp(a.tasks,b.tasks,a.n*sizeof(a.tasks[0]))==0&&
        std::equal(a.row_offsets,a.row_offsets+65,b.row_offsets)&&
        std::equal(a.row_ids,a.row_ids+2*a.n,b.row_ids)&&
        std::equal(a.features,a.features+7,b.features)&&a.prediction==b.prediction;
}
extern "C" int b16_fast_bench(const uint32_t*in,int samples,PlanFn reference,uint64_t*out){
    if(!in||!reference||!out||samples<1||samples>100000)return -1;
    B16Packed old{},fast{};
    for(int i=0;i<samples+32;i++)for(int j=0;j<3;j++){
        int mode=(i+j)%3;
        for(int k=0;k<2;k++){
            bool use_fast=bool((i+j+k)&1);auto fn=use_fast?b16_plan:reference;
            auto* dst=use_fast?&fast:&old;
            auto t=std::chrono::steady_clock::now();
            int rc=fn(in,mode,dst);
            auto e=std::chrono::steady_clock::now();
            if(rc)return -2;
            if(i>=32)out[(2*mode+int(use_fast))*samples+i-32]=
                std::chrono::duration_cast<std::chrono::nanoseconds>(e-t).count();
        }
        if(!packed_equal(old,fast))return -3;
    }
    return 0;
}
