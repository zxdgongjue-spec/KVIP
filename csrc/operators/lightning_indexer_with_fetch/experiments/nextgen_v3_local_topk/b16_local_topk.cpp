// NextGen V3a: independent P6 TopK port; frozen new LI headers.
// LB-P2a: independent AIV local TopK over validated logical Score.
// LB-P4b1: native per-query exclusive visibility. Positive sortable scores only.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "indexer_types.h"
#include "topk.h"
namespace b16_local {
using namespace AscendC;
constexpr uint32_t C=32,B=16,Q=4,K=2048,TILE=128,TRUNK=8192;
constexpr uint32_t MAGIC=0x42313632U;
struct Task { uint32_t batch,group,begin,end; };
static_assert(sizeof(Task)==16,"task ABI");
class Local {
public:
    __aicore__ inline void Init(TPipe& pipe) {
        op.Init(K,TRUNK);
        pipe.InitBuffer(mrgBuf,(K+TRUNK)*sizeof(uint16_t));
        pipe.InitBuffer(idxBuf,(K+64)*sizeof(uint32_t));
        pipe.InitBuffer(valueBuf,K*sizeof(uint16_t));
        pipe.InitBuffer(tmpBuf,op.GetSharedTmpBufferSize());
        mrg=mrgBuf.Get<uint16_t>(); idx=idxBuf.Get<uint32_t>();
        values=valueBuf.Get<uint16_t>(); tmp=tmpBuf.Get<uint32_t>();
        op.InitBuffers(tmp,idx);
    }
    __aicore__ inline void Run(GlobalTensor<uint16_t> score,
        GlobalTensor<uint16_t> outScore, GlobalTensor<uint32_t> outIdx,
        uint32_t begin,uint32_t end) {
        const uint32_t len=end>begin?end-begin:0;
        // All LB-P1 plans are tile aligned. Short shards keep all valid
        // positions; sentinel tails must never enter a subsequent merge.
        if(len<K) {
            Duplicate(values,uint16_t(0xffff),K);
            Duplicate(idx.ReinterpretCast<int32_t>(),int32_t(-1),K);
            PipeBarrier<PIPE_V>();
            if(len) {
                CreateVecIndex(idx.ReinterpretCast<int32_t>(),int32_t(begin),len);
                PipeBarrier<PIPE_V>();
                const uint32_t tailBlock=len/8U*8U;
                uint64_t mask[1];mask[0]=(~0ULL)<<(len%8U);
                Duplicate(idx.ReinterpretCast<int32_t>()[tailBlock],int32_t(-1),mask,1,1,0);
                const uint32_t bulkTail=tailBlock+64U;
                if(bulkTail<K) {
                    PipeBarrier<PIPE_V>();
                    Duplicate(idx.ReinterpretCast<int32_t>()[bulkTail],int32_t(-1),K-bulkTail);
                }
                DataCopyExtParams cp{1,static_cast<uint32_t>(len*sizeof(uint16_t)),0,0,0};
                DataCopyPadExtParams<uint16_t> pad{true,0,static_cast<uint8_t>((16-len%16)%16),uint16_t(0xffff)};
                SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                DataCopyPad(values,score[begin],cp,pad);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }
            PipeBarrier<PIPE_ALL>();
        } else {
            const uint32_t loops=(len+TRUNK-1)/TRUNK;
            for(uint32_t j=0;j<loops;++j) {
                const uint32_t remaining=len-j*TRUNK;
                const uint32_t chunk=remaining<TRUNK?remaining:TRUNK;
                const uint32_t prefix=j?K:0;
                uint32_t n=prefix+chunk;
                if(n<K) n=K;
                n=(n+255U)/256U*256U;
                if(j) {
                    PipeBarrier<PIPE_V>();
                    DataCopy(mrg,values,K);
                }
                // Same minimum-padding convention as previous P2b.
                Duplicate(mrg[prefix],uint16_t(0),n-prefix);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                if(chunk%16U==0) {
                    DataCopy(mrg[prefix],score[begin+j*TRUNK],chunk);
                } else {
                    DataCopyExtParams cp{1,static_cast<uint32_t>(chunk*sizeof(uint16_t)),0,0,0};
                    DataCopyPadExtParams<uint16_t> pad{true,0,static_cast<uint8_t>((16-chunk%16)%16),0};
                    DataCopyPad(mrg[prefix],score[begin+j*TRUNK],cp,pad);
                }
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                op(mrg,idx,values,n,j,loops,true);
                PipeBarrier<PIPE_V>();
            }
            topkb32gather::IndicesAddOffset(idx,begin,K);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(outScore,values,K);
        DataCopy(outIdx,idx,K);
        // Candidate buffers are reused by the next task only after completion.
        SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
        PipeBarrier<PIPE_ALL>();
    }
private:
    topk::LITopk<uint16_t> op;
    TBuf<TPosition::VECCALC> mrgBuf,idxBuf,valueBuf,tmpBuf;
    LocalTensor<uint16_t> mrg,values;
    LocalTensor<uint32_t> idx,tmp;
};
}
#endif
// a[0]=logical Score[B,Q,H], a[1]=QueueOffsets[33], a[2]=Task[N],
// a[3]=control[4] (magic,version,H,N), a[4]=candidate scores[N,2,K],
// a[5]=candidate original indices[N,2,K], a[6]=key lengths[B].
// Candidate slots are logical task IDs, not physical core IDs.
extern "C" __global__ __aicore__ void v9_b16_local_topk_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        using namespace b16_local;
        using namespace AscendC;
        auto a=reinterpret_cast<__gm__ const uint64_t*>(bindings);
        for(uint32_t i=0;i<8;++i) if(!a[i]) return;
        const uint32_t lane=GetBlockIdx(),core=lane/2U,sub=lane%2U;
        if(core>=C) return;
        auto ctrl=reinterpret_cast<__gm__ const uint32_t*>(a[3]);
        const uint32_t h=ctrl[2],count=ctrl[3];
        if(ctrl[0]!=MAGIC||ctrl[1]!=2|| (h<8192U||h>524288U||h%256U!=0)||count==0||count>256) return;
        auto off=reinterpret_cast<__gm__ const uint32_t*>(a[1]);
        auto tasks=reinterpret_cast<__gm__ const Task*>(a[2]);
        auto klen=reinterpret_cast<__gm__ const uint32_t*>(a[6]);
        auto visible=reinterpret_cast<__gm__ const uint32_t*>(a[7]);
        const uint32_t first=off[core],last=off[core+1];
        if(off[0]!=0||off[C]!=count||first>last||last>count||first==last) return;
        for(uint32_t i=first;i<last;++i) {
            uint32_t b=tasks[i].batch,g=tasks[i].group,lo=tasks[i].begin,hi=tasks[i].end;
            if(b>=B||g>=2||lo>=hi||hi>h/TILE||hi*TILE>klen[b]) return;
            uint32_t v=visible[b*Q+2*g+sub];
            if(v>klen[b]||(v!=0&&v<K)) return;
        }
        TPipe pipe;
        Local op;
        op.Init(pipe);
        for(uint32_t i=first;i<last;++i) {
            uint32_t b=tasks[i].batch,g=tasks[i].group,lo=tasks[i].begin,hi=tasks[i].end;
            const uint64_t row=(uint64_t(b)*Q+2*g+sub)*h;
            const uint64_t dst=(uint64_t(i)*2+sub)*K;
            GlobalTensor<uint16_t> score,outScore;
            GlobalTensor<uint32_t> outIdx;
            score.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(a[0])+row);
            outScore.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(a[4])+dst);
            outIdx.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a[5])+dst);
            uint32_t v=visible[b*Q+2*g+sub];
            uint32_t end=hi*TILE<v?hi*TILE:v;
            op.Run(score,outScore,outIdx,lo*TILE,end);
        }
        pipe.Destroy();
    }
#endif
}
