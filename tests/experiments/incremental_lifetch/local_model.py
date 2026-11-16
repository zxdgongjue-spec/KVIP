"""Isolate the exact incremental implementation for CANN model analysis."""
from pathlib import Path
import shutil
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Build a local model harness from the actual integrated methods, not a rewritten TopK.")
parser.add_argument("output", type=Path)
parser.add_argument("--serial", action="store_true", help="Disable cross-Q overlap with the same allocation")
parser.add_argument("--revision", help="Read methods/helpers from an earlier git commit")
args = parser.parse_args()
repo=Path(__file__).resolve().parents[3];root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
dst=root/'tests/experiments/incremental_topk';dst.mkdir(parents=True)
def source(name):
    path='tests/experiments/incremental_lifetch/'+name
    return subprocess.check_output(['git','show',args.revision+':'+path],cwd=repo,text=True) if args.revision else (repo/path).read_text()
for f in ('CMakeLists.txt','main.cpp'):shutil.copy(repo/'tests/experiments/incremental_topk'/f,dst/f)
shutil.copytree(repo/'csrc',root/'csrc',dirs_exist_ok=True,ignore=shutil.ignore_patterns('build*','_custom_opp','__pycache__'))
methods=source('methods.h')
flush=methods[methods.index('__aicore__ inline void FlushIncremental'):methods.index('// Skipped windows')]
end=methods.find('// A window has only one producer-side opportunity.')
if end<0:end=methods.index('__aicore__ inline bool AdvanceWindow')
advance=methods[methods.index('// Snapshot the next Q') if '// Snapshot the next Q' in methods else methods.index('__aicore__ inline void AdvanceIncremental'):end]
helpers=source('vector_helpers.h').replace('../../../common/lifetch/incremental_compact.h','../../../csrc/operators/common/lifetch/incremental_compact.h')
(dst/'helpers.h').write_text(helpers)
batched='incrementalMetadataActive_' in methods
code='''#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "indexer_types.h"
#include "topk.h"
#include "helpers.h"
using namespace AscendC;
namespace rotating {
struct Context {uint64_t hits;};
struct SequenceSlice {uint32_t shard,end;};
__aicore__ inline SequenceSlice SequenceSliceAt(uint32_t token,uint32_t resident,uint32_t half){uint32_t i=(token-resident)/half;return{i%2,resident+(i+1)*half};}
}
namespace kvaf_device {
__aicore__ inline void CleanCacheLines(__gm__ uint8_t* p,uint32_t n){GlobalTensor<uint8_t> g;g.SetGlobalBuffer(p);for(uint32_t i=0;i<n;i+=64) DataCacheCleanAndInvalid<uint8_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(g[i]);}
}
class Adapter {
public:
struct Shape {uint32_t history,resident;} runtimeShape_;
uint32_t trunkLen_=8192,incrementalCalls_=0,incrementalScanned_=0,incrementalCandidates_=0;
GM_ADDR prefixOutput_;GM_ADDR rotationContext_;GM_ADDR metadata_;
__aicore__ inline uint32_t Min(uint32_t a,uint32_t b){return a<b?a:b;}
__aicore__ inline uint32_t Max(uint32_t a,uint32_t b){return a>b?a:b;}
LocalTensor<uint16_t> mrgValueLocal_,scoreOutLocal_,incrementalNew_;
LocalTensor<uint32_t> indicesOutLocal_,incrementalOld_,topkSharedTmpLocal_;
LocalTensor<uint64_t> incrementalMetadata_;bool incrementalMetadataActive_=true;uint32_t incrementalMetadataIndex_=0;
topk::LITopk<uint16_t> topkOp_;
__aicore__ inline __gm__ uint64_t* IncrementalMeta(uint32_t q){return reinterpret_cast<__gm__ uint64_t*>(metadata_)+q*16;}
'''+flush+advance+'''
};
#endif
extern "C" __global__ __aicore__ void incremental_topk_kernel(GM_ADDR scores,GM_ADDR oldIds,GM_ADDR outScores,GM_ADDR outIds,GM_ADDR ticks,GM_ADDR masks,uint32_t rows,uint32_t added,uint32_t profile,uint32_t mode,uint32_t candidates){
#if defined(__CCE_AICORE__)
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
if ASCEND_IS_AIV {
TPipe pipe;Adapter a;TBuf<TPosition::VECCALC> input,ids,values,old,scratch,map,meta;
a.runtimeShape_={16384+added,16384};a.prefixOutput_=ticks;a.metadata_=ticks+131072;a.rotationContext_=ticks+262144;
a.topkOp_.Init(2048,8192);
pipe.InitBuffer(meta,1024);a.incrementalMetadata_=meta.Get<uint64_t>();pipe.InitBuffer(input,(2048+8192)*2);pipe.InitBuffer(ids,8192);pipe.InitBuffer(values,4096);pipe.InitBuffer(old,8192);pipe.InitBuffer(map,16384);pipe.InitBuffer(scratch,a.topkOp_.GetSharedTmpBufferSize());
a.mrgValueLocal_=input.Get<uint16_t>();a.indicesOutLocal_=ids.Get<uint32_t>();a.scoreOutLocal_=values.Get<uint16_t>();a.incrementalOld_=old.Get<uint32_t>();a.incrementalNew_=map.Get<uint16_t>();a.topkSharedTmpLocal_=scratch.Get<uint32_t>();a.topkOp_.InitBuffers(a.topkSharedTmpLocal_,a.indicesOutLocal_);

GlobalTensor<uint16_t> out;out.SetGlobalBuffer((__gm__ uint16_t*)outScores);GlobalTensor<uint32_t> outid;outid.SetGlobalBuffer((__gm__ uint32_t*)outIds);
GlobalTensor<uint64_t> metadata;metadata.SetGlobalBuffer((__gm__ uint64_t*)(ticks+131072));DataCopy(a.incrementalMetadata_,metadata,128);SetFlag<HardEvent::MTE2_S>(EVENT_ID5);WaitFlag<HardEvent::MTE2_S>(EVENT_ID5);
for(uint32_t q=0;q<rows;++q){
a.incrementalMetadataIndex_=q;
a.AdvanceIncremental(q,16384+added,16384+added,8192);

}
SetFlag<HardEvent::S_MTE3>(EVENT_ID5);WaitFlag<HardEvent::S_MTE3>(EVENT_ID5);DataCopy(metadata,a.incrementalMetadata_,128);SetFlag<HardEvent::MTE3_S>(EVENT_ID5);WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
pipe.Destroy();}
#endif
}
'''
if not batched:
    code=code.replace('DataCopy(a.incrementalMetadata_,metadata,128);SetFlag<HardEvent::MTE2_S>(EVENT_ID5);WaitFlag<HardEvent::MTE2_S>(EVENT_ID5);','')
    code=code.replace('SetFlag<HardEvent::S_MTE3>(EVENT_ID5);WaitFlag<HardEvent::S_MTE3>(EVENT_ID5);DataCopy(metadata,a.incrementalMetadata_,128);SetFlag<HardEvent::MTE3_S>(EVENT_ID5);WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);','')
if 'kIncrementalInputBytes' in methods:
    fields=methods[methods.index('// One additional Q:'):methods.index('__aicore__ inline __gm__ uint64_t* IncrementalMeta')]
    code=code.replace('topk::LITopk<uint16_t> topkOp_;', fields+'topk::LITopk<uint16_t> topkOp_;')
    # Reserve the fused LI's other live buffers; this is the same 218 KiB limit.
    code=code.replace('a.topkOp_.Init(2048,8192);', 'TBuf<TPosition::VECCALC> liLive;pipe.InitBuffer(liLive,90624);a.topkOp_.Init(2048,8192);')
    # Match InitBuffers ordering too, including the production ID padding.
    code=code.replace('pipe.InitBuffer(meta,1024);a.incrementalMetadata_=meta.Get<uint64_t>();pipe.InitBuffer(input,(2048+8192)*2);pipe.InitBuffer(ids,8192);pipe.InitBuffer(values,4096);pipe.InitBuffer(old,8192);pipe.InitBuffer(map,16384);pipe.InitBuffer(scratch,a.topkOp_.GetSharedTmpBufferSize());', 'pipe.InitBuffer(input,(2048+8192)*2);pipe.InitBuffer(ids,8448);pipe.InitBuffer(values,4096);pipe.InitBuffer(scratch,a.topkOp_.GetSharedTmpBufferSize());pipe.InitBuffer(map,16384);pipe.InitBuffer(old,8192);pipe.InitBuffer(meta,1024);a.incrementalMetadata_=meta.Get<uint64_t>();')
    code=code.replace('GlobalTensor<uint16_t> out;', 'pipe.InitBuffer(a.incrementalInputBuf_,a.kIncrementalInputBytes);a.incrementalInput_=a.incrementalInputBuf_.Get<uint16_t>();a.incrementalPipeline_='+('false' if args.serial else 'true')+';GlobalTensor<uint16_t> out;')
    code=code.replace('SetFlag<HardEvent::S_MTE3>(EVENT_ID5);', 'if(a.incrementalOutputPending_){WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);a.incrementalOutputPending_=false;}SetFlag<HardEvent::S_MTE3>(EVENT_ID5);')
    code=code.replace('a.incrementalMetadataIndex_=q;', 'a.incrementalMetadataIndex_=q;a.incrementalHasNext_=q+1<rows;a.incrementalNextQuery_=q+1;a.incrementalNextIndex_=q+1;a.incrementalNextEnd0_=16384+added;a.incrementalNextEnd1_=16384+added;')
(dst/'kernel.cpp').write_text(code)
p=dst/'main.cpp';s=p.read_text();s=s.replace('    void* s=upload(deviceScores);', '''    std::vector<uint16_t> expanded(n*(oldHistory+added),0);
    std::vector<uint32_t> fullmask(n*(oldHistory+added)/16,0);
    std::vector<uint64_t> workspace(65536,0);
    for(uint32_t q=0;q<n;++q){
      std::copy(source.begin()+q*width+K,source.begin()+(q+1)*width,expanded.begin()+q*(oldHistory+added)+oldHistory);
      std::copy(masks.begin()+q*added/16,masks.begin()+(q+1)*added/16,fullmask.begin()+(q*(oldHistory+added)+oldHistory)/16);
      auto base=reinterpret_cast<uint8_t*>(workspace.data())+(q/4)*(4*2048*6+1024);
      std::copy(source.begin()+q*width,source.begin()+q*width+K,reinterpret_cast<uint16_t*>(base)+(q%4)*K);
      std::copy(oldIds.begin()+q*K,oldIds.begin()+(q+1)*K,reinterpret_cast<uint32_t*>(base+4*K*2)+(q%4)*K);
    }
    deviceScores=expanded;masks=fullmask;
    void* s=upload(deviceScores);''').replace('upload(std::vector<uint64_t>(n*5))','upload(workspace)').replace('added>16384','added>65536')
s=s.replace('void* ticks=upload(workspace);', 'for(uint32_t q=0;q<n;++q){workspace[131072/8+q*16]=reinterpret_cast<uint64_t>(s)+q*(16384+added)*2;workspace[131072/8+q*16+1]=16384ULL|(16384ULL<<32); }workspace[262144/8]=reinterpret_cast<uint64_t>(mask);void* ticks=upload(workspace);')
s=s.replace('auto actual=read<uint16_t>(values,n*K);auto actualIds=read<uint32_t>(ids,n*K);', 'auto state=read<uint64_t>(ticks,65536);std::vector<uint16_t> actual(n*K);std::vector<uint32_t> actualIds(n*K);for(uint32_t q=0;q<n;++q){auto base=reinterpret_cast<uint8_t*>(state.data())+(q/4)*(4*K*6+1024);std::copy_n(reinterpret_cast<uint16_t*>(base)+(q%4)*K,K,actual.begin()+q*K);std::copy_n(reinterpret_cast<uint32_t*>(base+4*K*2)+(q%4)*K,K,actualIds.begin()+q*K);}')
s=s.replace('    constexpr uint32_t K=2048,oldHistory=16384;', '    if(!sim || blocks!=1 || rows>8 || mode!=1)throw std::runtime_error("local model requires --sim --blocks 1 --rows <=8 --mode bitmap");\n    constexpr uint32_t K=2048,oldHistory=16384;')
a=s.index('auto trace=read<uint64_t>(ticks,n*5);');b=s.index('    for(void* p:',a)
s=s[:a]+s[b:]
p.write_text(s)
print(f'Model harness: {root}; methods={args.revision or "working tree"}; batched_metadata={batched}')
