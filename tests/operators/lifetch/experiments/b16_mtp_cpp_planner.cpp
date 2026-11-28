// Final experimental MTP-aware planner, CPU only.
// Reuse the exact frozen Q4 candidate generator; add effective Q2 work.
#include "b16_cpp_planner_fast.cpp"
namespace {
constexpr int Effective=7;
using Visible=std::array<uint32_t,B*Q>;
struct Geometry {
    Lens l{};
    Visible v{};
    std::array<int,C> ends{};
    int total=0,upper=0,lower=0;
};
bool geometry(const uint32_t*ks,const uint32_t*vis,Geometry&g){
    if(!vis||!read_lens(ks,g.l))return false;
    int valid=0;
    for(int b=0;b<B;b++){
        int qn=0;
        for(int q=0;q<Q;q++){
            uint32_t x=vis[b*Q+q];
            if(x){
                if(q!=qn||x<K||x>uint32_t(g.l[b]*TILE)||
                    (q&&x!=vis[b*Q+q-1]+1))return false;
                qn++;
            }
            g.v[b*Q+q]=x;
        }
        if(qn<2)return false;
        valid+=qn;
    }
    if(valid<56||valid>64)return false;
    for(int i=0;i<C;i++){
        int e=ceildiv(int(std::max(g.v[2*i],g.v[2*i+1])),TILE);
        g.ends[i]=e;g.total+=e;g.upper=std::max(g.upper,e);
    }
    g.lower=std::max(ceildiv(g.total,C),ceildiv(g.upper,8));
    return true;
}
bool effective_capacity(const Geometry&g,const std::array<int,C>&order,int target,Plan&p){
    std::array<int,C> load{};
    for(int id:order){
        const int end=g.ends[id],physical=g.l[id/2];
        uint32_t b=id/2,group=id%2;
        if(!end){
            int c=0;
            for(int i=1;i<C;i++)
                if(std::tie(load[i],p.nt[i],i)<std::tie(load[c],p.nt[c],c))c=i;
            if(!p.push(c,{b,group,0,uint32_t(physical)}))return false;
            continue;
        }
        int lo=0,n=0;uint32_t used=0;
        while(lo<end){
            if(n==8)return false;
            int rem=end-lo,fit=-1,largest=-1;
            for(int i=0;i<C;i++)if(!(used&(uint32_t(1)<<i))&&load[i]<target){
                int free=target-load[i];
                if(largest<0||std::make_tuple(-free,p.nt[i],i)<
                    std::make_tuple(load[largest]-target,p.nt[largest],largest))largest=i;
                if(free>=rem&&(fit<0||std::make_tuple(free-rem,p.nt[i],i)<
                    std::make_tuple(target-load[fit]-rem,p.nt[fit],fit)))fit=i;
            }
            int c=fit>=0?fit:largest;
            if(c<0)return false;
            int take=std::min(rem,target-load[c]),hi=lo+take;
            if(!p.push(c,{b,group,uint32_t(lo),uint32_t(hi==end?physical:hi)}))return false;
            load[c]+=take;lo=hi;used|=uint32_t(1)<<c;n++;
        }
    }
    return true;
}
struct MFeatures {
    uint32_t n=0,active=0,max_tiles=0,total_tiles=0,cores=0,candidates=0,
        max_trunks=0,max_shards=0,merge_rows=0;
    std::array<uint32_t,C> loads{};
};
MFeatures mf(const Plan&p,const Geometry&g,bool clip){
    MFeatures f;f.n=p.n;f.max_shards=p.max_shards;
    for(int c=0;c<C;c++){
        int loads=0,rounds[2]={0,0};
        for(int i=p.head[c];i>=0;i=p.next[i]){
            auto t=p.t[i];int id=t.b*2+t.g;
            int end=clip?std::min(int(t.hi),g.ends[id]):int(t.hi);
            int work=std::max(0,end-int(t.lo));
            if(work){++f.active;loads+=work;}
            for(int lane=0;lane<2;lane++){
                int len=std::max(0,std::min(int(t.hi)*TILE,int(g.v[2*id+lane]))-int(t.lo)*TILE);
                f.candidates+=std::min(K,len);
                // Proxy, not exact hardware scan rounds.
                rounds[lane]+=ceildiv(len,8192);
            }
        }
        f.loads[c]=loads;f.total_tiles+=loads;f.cores+=loads>0;
        f.max_tiles=std::max(f.max_tiles,uint32_t(loads));
        f.max_trunks=std::max(f.max_trunks,uint32_t(std::max(rounds[0],rounds[1])));
    }
    for(int id=0;id<C;id++)if(p.shards[id]>1)
        f.merge_rows+=int(g.v[2*id]!=0)+int(g.v[2*id+1]!=0);
    return f;
}
Plan effective(const Geometry&g){
    std::array<int,C> order{};
    for(int i=0;i<C;i++)order[i]=i;
    std::sort(order.begin(),order.end(),[&](int a,int b){
        return std::make_tuple(-g.ends[a],a)<std::make_tuple(-g.ends[b],b);
    });
    std::array<Plan,9> plans;
    std::array<MFeatures,9> fs{};
    std::array<int,9> targets{};
    int n=0;uint32_t best=UINT32_MAX;
    for(int pct:{100,102,105,110,115,120,130,150,200}){
        int target=ceildiv(g.lower*pct,100);
        if(n&&targets[n-1]==target)continue;
        Plan p;if(!effective_capacity(g,order,target,p))continue;
        auto f=mf(p,g,true);
        plans[n]=p;fs[n]=f;targets[n++]=target;best=std::min(best,f.max_tiles);
    }
    if(!n)return base_plan(g.l);
    int sel=-1;
    for(int i=0;i<n;i++)if(100*fs[i].max_tiles<=105*best){
        auto key=[&](int j){return std::make_tuple(fs[j].n,fs[j].candidates,fs[j].max_tiles,targets[j]);};
        if(sel<0||key(i)<key(sel))sel=i;
    }
    return plans[sel];
}
// Frozen engineering proxy, not a fit to the MTP holdout.
// Keep Q4 coefficients for critical path and scan proxy. Budget 45 us for
// incremental cold planning, 40 us for merge metadata H2D, 0.30 us/extra
// task; device clip adds 1 + 0.005*max effective tiles.
// The constant mean term is identical across modes, avoiding a reward for
// reducing total work without reducing critical path.
double cost(const MFeatures&f,const Geometry&g,int label,int mode){
    double v=25.28977509043643+0.16685339451843886*f.max_tiles+
        0.015531602670248825*(double(sum(g.l))/B)+1.7811570487347388*f.max_trunks;
    if(label!=Count){
        v+=8.100773020100553+1.1815789344768906*(int(f.max_shards)-1)+
            12.399691064711648*(double(f.merge_rows)/(B*Q));
        v+=40.+.30*std::max(0,int(f.n)-32);
    }
    if(mode)v+=1.+.005*f.max_tiles;
    if(label!=Count||mode)v+=45.;
    return v;
}
struct Choice {
    Plan p;MFeatures f;int label=Count,mode=0,gate=0;double predicted=0,base=0;
};
Choice select_plan(const Geometry&g,bool baseline){
    Choice out;out.p=base_plan(g.l);out.f=mf(out.p,g,false);
    out.base=out.predicted=cost(out.f,g,Count,0);
    if(baseline)return out;
    // Model-only lower bound: sum of per-shard ceil(length/8192) is at
    // least ceil(full visible length/8192) for each logical row.
    // There are 64 Vector lanes. This is NOT a hardware time bound.
    int scans=0;for(auto v:g.v)scans+=ceildiv(int(v),8192);
    double optimistic=25.28977509043643+0.16685339451843886*g.lower+
        0.015531602670248825*(double(sum(g.l))/B)+
        1.7811570487347388*ceildiv(scans,B*Q)+45.;
    if(out.base-optimistic+1e-8<std::max(5.,.05*out.base)){
        out.gate=1;return out;
    }
    // Audit A4-01: every non-Count label pays at least the fixed Merge
    // and H2D terms, even when the physical plan happens to have no split.
    // The remaining terms are nonnegative. Count+clip is evaluated exactly
    // and separately because it does not pay those fixed Merge terms.
    const double required=std::max(5.,.05*out.base);
    const double split_lower=optimistic+8.100773020100553+40.;
    if(out.base-split_lower+1e-8<required){
        const double count_clip=cost(mf(out.p,g,true),g,Count,1);
        if(out.base-count_clip+1e-8<required){
            out.gate=1;return out;
        }
    }
    auto consider=[&](const Plan&p,int label,int mode){
        auto f=mf(p,g,mode!=0);
        double t=cost(f,g,label,mode);
        if(out.base-t>=std::max(5.,.05*out.base)&&t<out.predicted){
            out.p=p;out.f=f;out.label=label;out.mode=mode;out.predicted=t;
        }
    };
    auto a=generate(g.l,true);
    for(int i=0;i<a.n;i++)for(int mode:{0,1})
        consider(a.p[i],a.label[i],mode);
    auto e=effective(g);consider(e,Effective,1);
    return out;
}
}
extern "C" {
struct MtpPacked {
    uint32_t label,mode,n,merge,gate;
    uint32_t offsets[33],tasks[256][4],visible[64],row_offsets[65],row_ids[512];
    uint32_t features[9],loads[32];
    double predicted,base;
};
}
namespace {
bool pack_mtp(const Choice&ch,const Geometry&g,MtpPacked&out){
    auto&p=ch.p;out.label=ch.label;out.mode=ch.mode;out.n=p.n;
    out.merge=ch.label!=Count;out.gate=ch.gate;
    std::copy(g.v.begin(),g.v.end(),out.visible);
    std::array<std::array<int,8>,C> ids{};
    std::array<int,C> counts{};
    uint32_t j=0;out.offsets[0]=0;
    for(int c=0;c<C;c++){
        for(int i=p.head[c];i>=0;i=p.next[i]){
            auto t=p.t[i];int id=t.b*2+t.g;
            if(t.b>=B||t.g>=2||t.lo>=t.hi||t.hi>uint32_t(g.l[t.b])||j>=MAX||counts[id]>=8)return false;
            out.tasks[j][0]=t.b;out.tasks[j][1]=t.g;out.tasks[j][2]=t.lo;out.tasks[j][3]=t.hi;
            ids[id][counts[id]++]=j++;
        }
        out.offsets[c+1]=j;
    }
    if(j!=uint32_t(p.n))return false;
    uint32_t r=0;out.row_offsets[0]=0;
    for(int id=0;id<C;id++){
        int n=counts[id];if(!n)return false;
        std::sort(ids[id].begin(),ids[id].begin()+n,[&](int x,int y){return out.tasks[x][2]<out.tasks[y][2];});
        uint32_t end=0;
        for(int k=0;k<n;k++){
            auto t=out.tasks[ids[id][k]];if(t[2]!=end)return false;end=t[3];
        }
        if(end!=uint32_t(g.l[id/2]))return false;
        for(int lane=0;lane<2;lane++){
            for(int k=0;k<n;k++)out.row_ids[r++]=2*ids[id][k]+lane;
            out.row_offsets[2*id+lane+1]=r;
        }
        if(!out.merge&&(n!=1||ids[id][0]!=id))return false;
    }
    if(r!=2*j)return false;
    auto f=ch.f;
    uint32_t fs[]={f.n,f.active,f.max_tiles,f.total_tiles,f.cores,f.candidates,f.max_trunks,f.max_shards,f.merge_rows};
    std::copy(fs,fs+9,out.features);std::copy(f.loads.begin(),f.loads.end(),out.loads);
    out.predicted=ch.predicted;out.base=ch.base;
    return true;
}
Choice specific(const Plan&p,const Geometry&g,int label,int mode){
    Choice ch;ch.p=p;ch.f=mf(p,g,mode!=0);ch.label=label;ch.mode=mode;
    ch.predicted=cost(ch.f,g,label,mode);ch.base=cost(mf(base_plan(g.l),g,false),g,Count,0);
    return ch;
}
}
extern "C" {
// mode=0 baseline, mode=1 automatic selection. All paths build full descriptors.
__attribute__((noinline)) int mtp_plan(const uint32_t*ks,const uint32_t*vis,int mode,MtpPacked*out){
    Geometry g;if(!out||mode<0||mode>1||!geometry(ks,vis,g))return -1;
    return pack_mtp(select_plan(g,mode==0),g,*out)?0:-2;
}
// Diagnostic all candidates, no gate. Caller allocates 15 entries.
int mtp_all(const uint32_t*ks,const uint32_t*vis,MtpPacked*out){
    Geometry g;if(!out||!geometry(ks,vis,g))return -1;
    auto a=generate(g.l,true);int n=0;
    for(int i=0;i<a.n;i++)for(int m:{0,1})
        if(!pack_mtp(specific(a.p[i],g,a.label[i],m),g,out[n++]))return -2;
    if(!pack_mtp(specific(effective(g),g,Effective,1),g,out[n++]))return -2;
    return n;
}
// Native CPU timing with rotating baseline/auto; no ctypes/H2D in interval.
int mtp_bench(const uint32_t*ks,const uint32_t*vis,int samples,uint64_t*out){
    if(!out||samples<1||samples>100000)return -1;
    MtpPacked p{};volatile uint64_t check=0;
    for(int i=0;i<samples+32;i++)for(int j=0;j<2;j++){
        int m=(i+j)%2;
        auto t=std::chrono::steady_clock::now();
        int rc=mtp_plan(ks,vis,m,&p);
        auto e=std::chrono::steady_clock::now();
        if(rc)return -2;
        check=check+p.n+p.label;
        if(i>=32)out[m*samples+i-32]=std::chrono::duration_cast<std::chrono::nanoseconds>(e-t).count();
    }
    return check?0:-3;
}
}
