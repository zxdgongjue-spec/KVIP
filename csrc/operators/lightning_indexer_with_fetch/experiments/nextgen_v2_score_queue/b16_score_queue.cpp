// NextGen V2a: isolated port of LB-P4b3a to frozen 79c6a225.
// Score only: no production Context, Rotation, Fetch or TopK publication.
// Keep physical Q4, Q2 Cube shape and aligned K backing unchanged.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
struct QLITilingData {
    uint32_t bSize,tSize,gSize,s1Size,s2Size,sparseCount,usedCoreNum,blockSize,maxBlockNumPerBatch,sparseMode;
    uint32_t keyStride0=16384,keyDequantScaleStride0=128;
    uint32_t resident,splitQ,aicCount;
};
#include "indexer.h"

namespace b16_score {
using namespace AscendC;
using namespace QLICommon;
using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,
    LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;
constexpr uint32_t kMagic=0x42313631U;
constexpr uint32_t kAics=32, kBatch=16, kQ=4, kTile=128, kMaxTasks=256;
struct Task { uint32_t batch,group,begin,end; }; // half-open tile range
static_assert(sizeof(Task)==16,"B16 task ABI");
__aicore__ inline uint32_t VisibleEnd(const __gm__ uint32_t* visible,
                                     uint32_t b,uint32_t g,uint32_t end) {
    uint32_t x=visible[b*kQ+2*g],y=visible[b*kQ+2*g+1];
    uint32_t limit=((x>y?x:y)+kTile-1)/kTile;
    return end<limit?end:limit;
}

// All tasks share the original buffers/events. No reinitialization between
// tasks, no global barrier, no reuse of a logical Score destination.
class QueueKernel : public QLIKernel::QuantLightningIndexerKernel<Type> {
public:
    __aicore__ inline void RunQueue(const __gm__ Task* tasks, uint32_t first,
        uint32_t last, GM_ADDR scores, uint32_t history,
        const __gm__ uint32_t* visible)
    {
        if (first==last) return;
        if ASCEND_IS_AIV {
            vectorService.AllocEventID();
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(ConstInfo::CROSS_VC_EVENT);
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(ConstInfo::CROSS_VC_EVENT+1);
        } else {
            matmulService.AllocEventID();
        }
        uint32_t loop=0, qScaleLoop=0;
        for (uint32_t i=first;i<last;++i) {
            const uint32_t b=tasks[i].batch, g=tasks[i].group;
            const uint32_t lo=tasks[i].begin, hi=VisibleEnd(visible,b,g,tasks[i].end);
            if(lo>=hi) continue;
            splitCoreInfo=SplitCoreInfo{};
            splitCoreInfo.isCoreEnable=true;
            splitCoreInfo.bN2Start=splitCoreInfo.bN2End=b;
            splitCoreInfo.gS1Start=splitCoreInfo.gS1End=g;
            splitCoreInfo.s2Start=lo;
            splitCoreInfo.s2End=hi-1;
            tempLoopInfo=QLIKernel::TempLoopInfo{};
            CalcGS1LoopParams(b);
            CalcS2LoopParams(b,g);
            // Q2: one row per AIV, native per-group row offsets 0 and H.
            // Each task writes [batch, group*2:group*2+2, lo*128:hi*128].
            if ASCEND_IS_AIV {
                GlobalTensor<uint16_t> dst;
                const uint64_t off=(uint64_t(b)*kQ+2U*g)*history;
                dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(scores)+off);
                vectorService.InitVecWorkspaceTensor(dst);
            }
            // Bound Scale prefetch to this clipped task, not the whole request.
            tempLoopInfo.validS2Len = hi * kTile;
            RunInfo runInfo{};
            uint32_t kScaleLoop=0;
            for(uint32_t t=lo;t<hi;++t) {
                if ((t-lo)%QLICommon::KScaleTiles==0) ++kScaleLoop;
                CalcRunInfo(loop,t,runInfo,qScaleLoop,kScaleLoop);
                // Never enter FinishBaseBlock: it owns production rotation
                // and final publication, which are absent from this probe.
                if ASCEND_IS_AIC {
                    matmulService.ComputeMm1(runInfo);
                } else {
                    vectorService.ProcessVec1(runInfo,0,false,true);
                }
                ++loop;
            }
            ++qScaleLoop;
        }
        if ASCEND_IS_AIV {
            vectorService.FreeEventID();
        } else {
            matmulService.FreeEventID();
            // Drain both lanes' two ping-pong credits, including AIV1.
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(ConstInfo::CROSS_VC_EVENT);
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(ConstInfo::CROSS_VC_EVENT+1);
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(ConstInfo::CROSS_VC_EVENT+ConstInfo::AIV0_AIV1_OFFSET);
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(ConstInfo::CROSS_VC_EVENT+ConstInfo::AIV0_AIV1_OFFSET+1);
        }
    }
};
__aicore__ inline void Status(__gm__ uint32_t* status,uint32_t lane,
                            uint32_t code,uint32_t tasks,uint32_t tiles) {
    // Separate 64B slots; bypass-DCache publication follows the existing
    // C7 task probe. Only AIV writes diagnostics in this first version.
    auto p=status+lane*16U;
    WriteGmByPassDCache<uint32_t>(p+1,tasks);
    WriteGmByPassDCache<uint32_t>(p+2,tiles);
    WriteGmByPassDCache<uint32_t>(p,code);
    DataSyncBarrier<MemDsbT::DDR>();
}
} // namespace b16_score
#endif

// Binding order: Q,K,W,Qscale,Kscale,Qlen,Klen,PA table,dummy index,
// logical Score, queue offsets[33], Task[N], control[5], AIV status[64][16], visible[64].
extern "C" __global__ __aicore__ void v9_b16_score_clip_kernel(GM_ADDR bindings)
{
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using namespace AscendC;
    using namespace b16_score;
    auto a=reinterpret_cast<__gm__ const uint64_t*>(bindings);
    for(uint32_t i=0;i<15;++i) if(!a[i]) return;
    uint32_t lane=0, core=0;
    if ASCEND_IS_AIV { lane=GetBlockIdx(); core=lane/2U; }
    else { core=GetBlockIdx(); }
    if(core>=kAics) return;
    auto ctrl=reinterpret_cast<__gm__ const uint32_t*>(a[12]);
    auto status=reinterpret_cast<__gm__ uint32_t*>(a[13]);
    uint32_t h=ctrl[2], count=ctrl[3], trace=ctrl[4];
    if(ctrl[0]!=kMagic || ctrl[1]!=2 || (h<8192U || h>524288U || h%256U!=0) ||
       count==0 || count>kMaxTasks || trace>1) return;
    auto offsets=reinterpret_cast<__gm__ const uint32_t*>(a[10]);
    auto tasks=reinterpret_cast<__gm__ const Task*>(a[11]);
    auto qlen=reinterpret_cast<__gm__ const uint32_t*>(a[5]);
    auto klen=reinterpret_cast<__gm__ const uint32_t*>(a[6]);
    auto visible=reinterpret_cast<__gm__ const uint32_t*>(a[14]);
    uint32_t first=offsets[core], last=offsets[core+1], tiles=0,active=0;
    bool valid=offsets[0]==0 && offsets[kAics]==count && first<=last && last<=count;
    for(uint32_t i=first;valid && i<last;++i) {
        uint32_t b=tasks[i].batch, g=tasks[i].group, lo=tasks[i].begin, hi=tasks[i].end;
        valid=b<kBatch && g<2 && lo<hi && hi<=h/kTile;
        if(valid) valid=qlen[b]==4 && klen[b]>0 && klen[b]<=h &&
            klen[b]%kTile==0 && hi<=klen[b]/kTile;
        if(valid) {
            uint32_t x=visible[b*kQ+2*g],y=visible[b*kQ+2*g+1];
            valid=x<=klen[b] && y<=klen[b] &&
                (x==0||x>=2048U) && (y==0||y>=2048U);
        }
        if(valid) {
            uint32_t end=VisibleEnd(visible,b,g,hi);
            if(end>lo) {tiles+=end-lo;++active;}
        }
    }
    if(!valid) {
        if ASCEND_IS_AIV { if(trace) Status(status,lane,10,0,0); }
        return;
    }
    // Both AIC and its AIV lanes decide before allocating pipeline events.
    if(active==0) {
        if ASCEND_IS_AIV { if(trace) Status(status,lane,2,0,0); }
        return;
    }
    TPipe pipe;
    QueueKernel op;
    op.runtimeShape_=rotating::RuntimeShape{};
    op.sequenceSplit_=false;
    op.sequenceTopKOnly_=false;
    // Deliberately independent of production PrepareShape/History admission.
    // tSize=64, splitQ=1, aicCount=32 -> native S1Base=2, MBase=64.
    QLITilingData tiling{16,64,32,4,h,2048,32,128,h/128,0,16384,128,8192,1,32};
#define P(i) reinterpret_cast<GM_ADDR>(a[i])
    op.Init(P(0),P(1),P(2),P(3),P(4),P(5),P(6),P(7),P(8),P(9),&tiling,&pipe);
    op.RunQueue(tasks,first,last,P(9),h,visible);
    pipe.Destroy();
    if ASCEND_IS_AIV { if(trace) Status(status,lane,1,active,tiles); }
#undef P
#endif
}
