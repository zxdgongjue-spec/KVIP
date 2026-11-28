// NextGen V3a: independent P6 TopK port; frozen new LI headers.
// LB-P2b: exact query-owned merge of task-owned candidates.
// Independent MIX launch, AIC returns immediately; one logical query per AIV.
// Initial admission: Q4/B16, tile-aligned ranges, all rows >= K, <=8 shards,
// positive sortable scores (zero is minimum padding). No production changes.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "indexer_types.h"
#include "topk.h"
namespace b16_merge {
using namespace AscendC;
constexpr uint32_t B=16,Q=4,K=2048,TILE=128,TRUNK=8192,MAX_SHARDS=8;
constexpr uint32_t MAGIC=0x42313633U;
struct Task { uint32_t batch,group,begin,end; };
static_assert(sizeof(Task)==16,"task ABI");
}
#endif
// Binding: candidate scores, candidate indices, Tasks, row offsets[65],
// candidate row IDs[2*N] sorted by S2 begin for each query,
// control[4]={magic,version,H,N}, lengths[16], final indices[64,K].
extern "C" __global__ __aicore__ void v9_b16_global_topk_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        using namespace AscendC;
        using namespace b16_merge;
        auto a=reinterpret_cast<__gm__ const uint64_t*>(bindings);
        for(uint32_t i=0;i<9;++i) if(!a[i]) return;
        const uint32_t row=GetBlockIdx();
        if(row>=B*Q) return;
        auto ctrl=reinterpret_cast<__gm__ const uint32_t*>(a[5]);
        const uint32_t h=ctrl[2],n=ctrl[3];
        if(ctrl[0]!=MAGIC||ctrl[1]!=2||(h<8192U||h>524288U||h%256U!=0)||n==0||n>256) return;
        auto tasks=reinterpret_cast<__gm__ const Task*>(a[2]);
        auto offsets=reinterpret_cast<__gm__ const uint32_t*>(a[3]);
        auto ids=reinterpret_cast<__gm__ const uint32_t*>(a[4]);
        auto lengths=reinterpret_cast<__gm__ const uint32_t*>(a[6]);
        const uint32_t first=offsets[row],last=offsets[row+1];
        if(offsets[0]!=0||offsets[B*Q]!=2*n||first>=last||last>2*n||
           last-first>MAX_SHARDS) return;
        const uint32_t b=row/Q,q=row%Q,shards=last-first;
        if(lengths[b]<K||lengths[b]>h||lengths[b]%TILE) return;
        auto visible=reinterpret_cast<__gm__ const uint32_t*>(a[8]);
        const uint32_t v=visible[row];
        if(v>lengths[b]||(v!=0&&v<K)) return;
        uint32_t end=0;
        for(uint32_t j=first;j<last;++j) {
            uint32_t id=ids[j];
            if(id>=2*n) return;
            const uint32_t t=id/2;
            const uint32_t lo=tasks[t].begin,hi=tasks[t].end;
            if(tasks[t].batch!=b||tasks[t].group!=q/2||id%2!=q%2||
               lo!=end||lo>=hi||hi>lengths[b]/TILE) return;
            end=hi;
        }
        if(end!=lengths[b]/TILE) return;
        GlobalTensor<uint16_t> cs;
        GlobalTensor<uint32_t> ci,out;
        cs.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(a[0]));
        ci.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a[1]));
        out.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a[7])+uint64_t(row)*K);
        TPipe pipe;
        TBuf<TPosition::VECCALC> idxBuf;
        pipe.InitBuffer(idxBuf,(K+64)*sizeof(uint32_t));
        LocalTensor<uint32_t> idx=idxBuf.Get<uint32_t>();
        if(v==0) {
            Duplicate(idx.ReinterpretCast<int32_t>(),int32_t(-1),K);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
            DataCopy(out,idx,K);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
            pipe.Destroy();
            return;
        }
        if(shards==1) {
            // A complete unsplit row already has the exact local result.
            DataCopy(idx,ci[uint64_t(ids[first])*K],K);
            PipeBarrier<PIPE_ALL>();
            DataCopy(out,idx,K);
            PipeBarrier<PIPE_ALL>();
            pipe.Destroy();
            return;
        }
        topk::LITopk<uint16_t> op;
        op.Init(K,TRUNK);
        TBuf<TPosition::VECCALC> mrgBuf,valueBuf,tmpBuf,mapBuf;
        pipe.InitBuffer(mrgBuf,(K+TRUNK)*sizeof(uint16_t));
        pipe.InitBuffer(valueBuf,K*sizeof(uint16_t));
        pipe.InitBuffer(tmpBuf,op.GetSharedTmpBufferSize());
        pipe.InitBuffer(mapBuf,MAX_SHARDS*K*sizeof(uint32_t));
        LocalTensor<uint16_t> mrg=mrgBuf.Get<uint16_t>(),values=valueBuf.Get<uint16_t>();
        LocalTensor<uint32_t> tmp=tmpBuf.Get<uint32_t>(),map=mapBuf.Get<uint32_t>();
        op.InitBuffers(tmp,idx);
        const uint32_t loops=(shards+3)/4;
        for(uint32_t loop=0;loop<loops;++loop) {
            uint32_t lo=loop*4,hi=lo+4<shards?lo+4:shards;
            uint32_t prefix=loop?K:0;
            if(loop) {
                PipeBarrier<PIPE_V>();
                DataCopy(mrg,values,K);
            }
            // Keep fixed K-sized candidate slots. Short shard tails are
            // masked to minimum score 0, NOT the local output 0xffff.
            Duplicate(mrg[prefix],uint16_t(0),(hi-lo)*K);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            for(uint32_t rank=lo;rank<hi;++rank) {
                uint32_t id=ids[first+rank],t=id/2;
                const uint32_t begin=tasks[t].begin*TILE;
                const uint32_t end=tasks[t].end*TILE<v?tasks[t].end*TILE:v;
                const uint32_t length=end>begin?end-begin:0;
                const uint32_t valid=length<K?length:K;
                if(valid) {
                    if(valid%16U==0) {
                        DataCopy(mrg[prefix+(rank-lo)*K],cs[uint64_t(id)*K],valid);
                    } else {
                        DataCopyExtParams cp{1,static_cast<uint32_t>(valid*sizeof(uint16_t)),0,0,0};
                        DataCopyPadExtParams<uint16_t> pad{true,0,static_cast<uint8_t>((16-valid%16)%16),0};
                        DataCopyPad(mrg[prefix+(rank-lo)*K],cs[uint64_t(id)*K],cp,pad);
                    }
                }
                DataCopy(map[rank*K],ci[uint64_t(id)*K],K);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            op(mrg,idx,values,prefix+(hi-lo)*K,loop,loops,true);
            PipeBarrier<PIPE_V>();
        }
        // Candidate positions -> original sequence positions.
        __VEC_SCOPE__ {
            topkb32gather::FindValueOutputVFImpl(
                reinterpret_cast<__ubuf__ uint32_t*>(idx.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(map.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(idx.GetPhyAddr()),
                static_cast<uint16_t>(K/64));
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(out,idx,K);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
        pipe.Destroy();
    }
#endif
}
