"""Isolate the four-Q terminal pipeline without modifying production kernels."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

here=Path(__file__).resolve().parent
repo=here.parents[3]
out=Path(sys.argv[1])
# Reuse the production window model and all algorithms verbatim.
# The existing probe is a CLI generator.
subprocess.run([sys.executable,str(here.parent/'pipeline_probe/prepare.py'),str(out)],check=True)
b=out/'csrc/operators/lightning_indexer_with_fetch/op_kernel'
# Restore production AdvanceWindow; expose per-Q orchestration only.
shutil.copy2(repo/'csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/incremental_methods.h',b/'indexer/incremental_methods.h')
# Keep terminal candidates separate from final slot output until every Q consumes them.
methods_file=b/'indexer/incremental_methods.h'
methods=methods_file.read_text()
old='reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+query*2048'
assert methods.count(old)==1
methods=methods.replace(old,'reinterpret_cast<__gm__ uint32_t*>(rankedEpoch_==runtimeShape_.rounds?ctx->sourceBlocks:ctx->locations)+query*2048')
# The terminal pipeline consumes compact candidates on the same core in UB.
# Keep intermediate-window GM publication; final full logical/slot output is unchanged.
export_start=methods.index('__aicore__ inline void ExportWindowCold(')
export_end=methods.index('__aicore__ inline GM_ADDR WindowTopkPrefix()',export_start)
export=methods[export_start:export_end]
assert export.count('if(count) {')==1
export=export.replace('if(count) {','if(count && rankedEpoch_!=runtimeShape_.rounds) {')
methods_file.write_text(methods[:export_start]+export+methods[export_end:])
p=b/'indexer/indexer_vector.h'
s=p.read_text().replace('#include "prefetch.h"','#include "prefetch.h"\n#include "terminal_logical_snapshot.h"')
shutil.copy2(here/'logical_snapshot.h',b/'terminal_logical_snapshot.h')
at=s.index('    __aicore__ inline void RotateFetch(')
methods='''    __aicore__ inline LocalTensor<uint32_t> TerminalCompact() { return incrementalNew_.template ReinterpretCast<uint32_t>(); }
    __aicore__ inline LocalTensor<uint32_t> TerminalScratch() { return topkSharedTmpLocal_; }
    __aicore__ inline LocalTensor<uint32_t> TerminalIds() { return indicesOutLocal_; }
    __aicore__ inline LocalTensor<uint32_t> TerminalWork() { return mrgValueLocal_.template ReinterpretCast<uint32_t>(); }
    __aicore__ inline void TerminalQuery(uint32_t query) {
        exportWindowCold_=true;rankTrace_=false;rankedEpoch_=runtimeShape_.rounds;
        incrementalMetadataActive_=false;incrementalHasNext_=false;
        windowCoveredEmpty_=false;
        uint32_t half=rotating::SequenceHalfWindow(runtimeShape_);
        rankHalf_=half;rankReady_=uint64_t(runtimeShape_.history)|(uint64_t(runtimeShape_.history)<<32);
        AdvanceIncremental(query,runtimeShape_.history,half?runtimeShape_.history:0,half);
        JoinColdOutput();
        if(incrementalOutputPending_){WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);incrementalOutputPending_=false;}
        PipeBarrier<PIPE_ALL>();
        exportWindowCold_=false;
    }
    __aicore__ inline void TerminalLogical(uint32_t query,GM_ADDR logical,GM_ADDR output,GM_ADDR error) {
        Simt::VF_CALL<rotating::TerminalLogicalSnapshot>(Simt::Dim3{512,1,1},reinterpret_cast<__gm__ uint32_t*>(WindowTopkPrefix()+4*2048*2)+query*2048,reinterpret_cast<__gm__ uint64_t*>(logical),reinterpret_cast<__gm__ uint32_t*>(output)+query*2048,reinterpret_cast<__gm__ uint32_t*>(error)+query*32,runtimeShape_.history);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    __aicore__ inline void TerminalPost(uint32_t lane) {
        rotating::FastService<false,true>(runtimeShape_.batch,runtimeShape_.history,runtimeShape_.resident,runtimeShape_.rounds,runtimeShape_.serviceLanes,runtimeShape_.tokenBytes,runtimeShape_.postThreads,runtimeShape_.postBank,runtimeShape_.batchedRows,runtimeShape_.scanTokens,rotationContext_,runtimeShape_.rounds-1,lane/2,lane,topkSharedTmpLocal_,mrgValueLocal_,indicesOutLocal_,rotating::SequenceHalfWindow(runtimeShape_));
    }
'''
p.write_text(s[:at]+methods+s[at:])
# Keep the shared bitmap exclusion algorithm; visit only intersecting tiles.
recent=b/'indexer/recent_seed.h'
r=recent.read_text()
old="""    for(uint32_t i=0;i<n;i+=128) {
        if(base+i>=end || base+i+128<=begin)continue;"""
new="""    if(begin>=end || begin>=base+n || end<=base)return;
    uint32_t first=begin>base?((begin-base)/128)*128:0;
    uint32_t stop=end<base+n?end-base:n;
    for(uint32_t i=first;i<stop;i+=128) {"""
if old not in r:raise ValueError('Bitmap exclusion implementation changed')
recent.write_text(r.replace(old,new))
# Terminal-only admission bitmap is dead: no subsequent window consumes it.
# Keep mapping, journal, change records and slot assignments unchanged.
prefetch=b/'prefetch.h'
p=prefetch.read_text()
old='        asc_atomic_or(admitted+token/32,1U<<(token%32));'
if p.count(old)!=1:raise ValueError('Candidate admission bitmap implementation changed')
p=p.replace(old,'        // Final window: no later window reads the covered bitmap.')
p=p.replace('uint64_t remote=asc_ldcg(addresses+token);', '/* Addresses prefetched with compact IDs. */')
p=p.replace('lo[i]=uint32_t(remote);hi[i]=uint32_t(remote>>32);slots[i]=physical;', 'slots[i]=physical;')
p=p.replace('slots[i]=physical;asc_stcg(map+token,slot);', 'slots[i]=physical;asc_stcg(map+token,physical);')

start=p.index('__aicore__ inline bool SubmitWqesAt(')
end=p.index('__aicore__ inline bool SubmitWqes(',start)
writer=p[start:end]
writer=writer.replace('SubmitWqesAt(', 'WriteReservedWqes(').replace('bool final,__gm__ Row* row)', 'bool final,__gm__ Row* row,uint32_t offset,uint32_t total)')
writer=writer.replace('if(final) {row->finalRequests=count;row->finalHead=head;}', 'head+=offset;')
writer=writer.replace('begin,n,count);','begin,n,total-offset);')
cut=writer.index('    if(!final) {',writer.index('for(uint32_t begin='))
writer=writer[:cut]+'    return true;\n}\n'
# Lease-scoped fields are generated by the harness when binding entities/output.
# SQ producer remains per-call dynamic; preserve the original WQE constructor.
writer=writer.replace('uint32_t offset,uint32_t total) {',
    'uint32_t offset,uint32_t total,__gm__ uint32_t* boundTemplate,uint64_t originalHead) {')
config_start=writer.index('    auto queue=')
config_end=writer.index('    SetFlag<HardEvent::S_V>',config_start)
writer=writer[:config_start]+'''    if(!count)return false;
    auto cfg64=cfg.ReinterpretCast<uint64_t>();uint64_t head=originalHead+offset;
    uint32_t sqDepth=cfg.GetValue(17);uint64_t sqAddress=cfg64.GetValue(12);
    if(final && (head>UINT64_MAX-count || count>=sqDepth))return false;
'''+writer[config_end:]
# Preserve the exact measured single-bank path and generic multi-bank fallback.
loop_start=writer.index('    for(uint32_t begin=0;')
loop_end=writer.index('    SetFlag<HardEvent::MTE3_S>',loop_start)
writer=writer[:loop_start]+'    if(count<=bankSize){\n        FinalBuild(rt_tokenBytes,reinterpret_cast<__ubuf__ uint32_t*>(remoteLow.GetPhyAddr()),\n            reinterpret_cast<__ubuf__ uint32_t*>(remoteHigh.GetPhyAddr()),\n            reinterpret_cast<__ubuf__ uint32_t*>(slots.GetPhyAddr()),\n            reinterpret_cast<__ubuf__ uint32_t*>(words.GetPhyAddr()),\n            reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),0,count,total-offset);\n        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);\n        uint32_t start=uint32_t(head)&(sqDepth-1),first=count<sqDepth-start?count:sqDepth-start;\n        DataCopy(sq[start*16],words,first*16);\n        if(first!=count)DataCopy(sq,words[first*16],(count-first)*16);\n        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);\n    }else{\n'+writer[loop_start:loop_end]+'    }\n'+writer[loop_end:]
prefetch.write_text(p[:end]+writer+p[end:])
for name in ['kernel.cpp','driver.cpp','parallel_post.h']:
    shutil.copy2(here/name,out/name)
stream=(repo/'csrc/operators/lightning_indexer_with_fetch/op_kernel/streaming.cpp').read_text()
start=stream.index('template<bool Restore=false>')
end=stream.index('#if !V9_UBMEM',start)
helpers=stream[start:end].replace('__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS)', '__simt_callee__')
for name in ['SRCSET','DSTSET','PLAN_SRC','PLAN_DST','CONTROL']:
    helpers=helpers.replace(name,'TAIL_'+name)
(out/'helpers.h').write_text(helpers)
# Keep the validated merged-task emitter as source instead of text transforms.
shutil.copy2(here/'emit.h',out/'emit.h')

# Save exact source provenance with every generated experiment.
files=['kernel.cpp','driver.cpp','parallel_post.h','helpers.h','emit.h','CMakeLists.txt']
files += [str((b/name).relative_to(out)) for name in ['indexer/recent_seed.h','indexer/indexer_vector.h','indexer/incremental_methods.h','terminal_logical_snapshot.h','prefetch.h']]
manifest={'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(), 'sha256':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in files}, 'model':'Ascend950PR_9599', 'cq_delay_sys_cnt_ticks':9000}
(out/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
