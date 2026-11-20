#!/usr/bin/env python3
"""Build an isolated experimental source; production source is never patched."""
import argparse
from pathlib import Path
import shutil


def ignore_artifacts(directory, names):
    # build.py and similarly named source files are required by the runner.
    return [name for name in names
            if name == '__pycache__' or name.endswith(('.so', '.o'))
            or (name.startswith('build') and (Path(directory)/name).is_dir())]


def prepare(repo, destination):
    for directory in ('csrc', 'python'):
        shutil.copytree(repo/directory,destination/directory,
            ignore=ignore_artifacts,symlinks=True)
    def patch(relative, old, new):
        path=destination/relative
        text=path.read_text()
        if text.count(old)!=1:
            raise ValueError(f'Expected unique patch site: {relative}: {old[:80]}')
        path.write_text(text.replace(old,new))
    base='csrc/operators/lightning_indexer_with_fetch/'
    # Reuse the existing prefix-event publisher in both the handoff and model.
    stream_path=destination/base/'op_kernel/streaming.cpp'
    stream_text=stream_path.read_text()
    begin=stream_text.index('__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishPrefetch(')
    end=stream_text.index('// ctl[32:64]',begin)
    helper=stream_text[begin:end].replace('i<kPrefetch','i<pipeline_ab::kPrefetch')
    (destination/base/'op_kernel/terminal_prefetch_publish.h').write_text('#pragma once\n'+helper)
    stream_path.write_text(stream_text[:begin]+'#include "terminal_prefetch_publish.h"\n'+stream_text[end:])
    # Export all final misses into the existing 2048-entry per-Q buffer.
    patch(base+'op_kernel/compact_misses.h',
          'if(scan[15]<=kCompactMissLimit){', 'if(scan[15]<=2048){')
    # Keep RunLi available inside the tail translation unit, without duplicate entry points.
    patch(base+'op_kernel/indexer.cpp','extern "C" __global__ __aicore__ void v9_li_kernel(GM_ADDR bindings) {',
          '#ifndef KVIP_FINAL_WINDOW_INCLUDE\nextern "C" __global__ __aicore__ void v9_li_kernel(GM_ADDR bindings) {')
    path=destination/base/'op_kernel/indexer.cpp';path.write_text(path.read_text()+'\n#endif\n')
    path=destination/base/'op_kernel/streaming.cpp'
    text=path.read_text()
    text='#define KVIP_FINAL_WINDOW_INCLUDE\n#include "indexer.cpp"\n'+text
    marker='extern "C" __global__ __aicore__ void v9_streaming_resolve_update(GM_ADDR arg) {'
    text=text.replace(marker,'#include "terminal_window.h"\n'+marker.replace('(GM_ADDR arg)','(GM_ADDR arg, GM_ADDR bindings)'))
    start=text.index('extern "C" __global__ __aicore__ void v9_streaming_resolve_update')
    at=text.index('  uint32_t workers=',start)
    tail=text[start:]
    declaration='  bool profile=(a->prefetchCount & 0x80000000U)!=0;\n'
    if tail.count(declaration)!=1:raise ValueError('Expected tail profiling declaration')
    text=text[:start]+tail.replace(declaration,'',1)
    phases='''  bool profile=(a->prefetchCount & 0x80000000U)!=0;
  if(profile) StampTerminal(a,block,0);
  if(block<2*shape.liCores) RunLi<SequenceLiPhase::Final>(bindings);
  if(profile) StampTerminal(a,block,1);
  AscendC::SyncAll<true>();
  if(profile) StampTerminal(a,block,2);
  if(block<shape.serviceLanes){
    if(profile)RunTerminalService<true>(a,block);
    else RunTerminalService<false>(a,block);
  }
  if(profile) StampTerminal(a,block,3);
  AscendC::SyncAll<true>();
  if(profile) StampTerminal(a,block,4);
'''
    text=text[:at]+phases+text[at:]
    # Extra TopK cores must still join the tail's retirement barrier, but have
    # no slots in the original per-core diagnostic allocation.
    tail_start=text.index('extern "C" __global__ __aicore__ void v9_streaming_resolve_update')
    head,tail=text[:tail_start],text[tail_start:]
    trace_marker='  auto trace=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*256+uint64_t(block)*32;'
    if tail.count(trace_marker)!=1:raise ValueError('Expected tail core trace allocation')
    tail=tail.replace(trace_marker,'  profile=profile && block<shape.batch*kResolveCores+4*workers;\n'+trace_marker,1)
    if tail.count('    Producer(a,b);')!=1:raise ValueError('Expected original resolver call')
    tail=tail.replace('    Producer(a,b);','    PublishTerminalReady(a,b);',1)
    text=head+tail
    # B has no original resolver or resolver fallback, including unused helpers.
    begin=text.index('__aicore__ inline void Producer(')
    end=text.index('// Cold sources and distinct hot destinations',begin)
    text=text[:begin]+text[end:]
    begin=text.index('// Same operations and buffers as the separate VF calls')
    end=text.index('__aicore__ inline void PublishQueryOutputs(',begin)
    text=text[:begin]+text[end:]
    begin=text.index('__simt_callee__\ninline void PublishQ(')
    end=text.index('#include "terminal_prefetch_publish.h"',begin)
    text=text[:begin]+text[end:]
    path.write_text(text)
    shutil.copy2(Path(__file__).with_name('terminal.h'),destination/base/'op_kernel/terminal_window.h')
    patch(base+'op_kernel/prefetch.h','template<bool captureIo, bool predecessorChecked=false>',
          'template<bool captureIo, bool predecessorChecked=false, bool terminal=false>')
    patch(base+'op_kernel/prefetch.h','    uint32_t sequenceHalfWindow) {',
          '    uint32_t sequenceHalfWindow, uint64_t terminalErrors=0) {')
    patch(base+'op_kernel/prefetch.h','        if (rt_batchedRows) postInput.SetValue(64,0);',
          '''        if (rt_batchedRows) postInput.SetValue(64,0);
        if constexpr(terminal){auto e=reinterpret_cast<__gm__ uint32_t*>(terminalErrors)+b*128;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(e),128);
            if(!e[2])continue;}''')
    patch(base+'op_kernel/prefetch.h','        if(available>ctx->windowLimit)available=ctx->windowLimit;',
          '''        if(available>ctx->windowLimit)available=ctx->windowLimit;
        if constexpr(terminal)available=kLruSlots-slots;''')
    patch(base+'op_kernel/prefetch.h','template<bool BuildSimt>', 'template<bool BuildSimt, bool terminal=false>')
    patch(base+'op_kernel/prefetch.h','ctx->locations)+b*8192;','(terminal?ctx->sourceOffsets:ctx->locations))+b*8192;')
    patch(base+'op_kernel/prefetch.h','__simt_vf__ __aicore__ LAUNCH_BOUND(256)\ninline void UnionColdTopk',
          'template<bool packed=false>\n__simt_vf__ __aicore__ LAUNCH_BOUND(256)\ninline void UnionColdTopk')
    patch(base+'op_kernel/prefetch.h','token=input[i];','token=packed?(input[i]>>11):input[i];')
    patch(base+'op_kernel/prefetch.h','Simt::VF_CALL<UnionColdTopk>', 'Simt::VF_CALL<UnionColdTopk<terminal>>')
    patch(base+'op_kernel/prefetch.h','ctx->locations)+b*8192);\n        DataCopy(cfg[128]',
          '(terminal?ctx->sourceBlocks:ctx->locations))+b*8192);\n        DataCopy(cfg[128]')
    patch(base+'op_kernel/prefetch.h','ctx->locations)+b*8192);\n                SetFlag',
          '(terminal?ctx->sourceOffsets:ctx->locations))+b*8192);\n                SetFlag')
    patch(base+'op_kernel/prefetch.h','BatchBuild<V9_BATCH_POST==1>', 'BatchBuild<V9_BATCH_POST==1,terminal>')
    patch('python/kvip/operators/lifetch/workspace.py','((max(1, rounds), lanes, 128), "uint8")',
          '((max(1, rounds)+1, lanes, 128), "uint8")')
    patch('csrc/operators/common/lifetch/li_launch.h',
          '    if (!result) result = LI_LAUNCH_3(shape->liCores, stream, bindings);','    // Final TopK runs inside the experimental tail.')
    patch(base+'op_host/launch.cpp','aclrtlaunch_v9_streaming_resolve_update(uint32_t,void*,void*);',
          'aclrtlaunch_v9_streaming_resolve_update(uint32_t,void*,void*,void*);')
    patch(base+'op_host/launch.cpp',
          'aclrtlaunch_v9_streaming_resolve_update(shape->batch*pipeline_ab::kResolveCores+4*workers,stream,workspace)',
          'aclrtlaunch_v9_streaming_resolve_update(std::max(2*shape->liCores,shape->batch*pipeline_ab::kResolveCores+4*workers),stream,workspace,bindings)')
    patch(base+'op_host/launch.cpp','#include <cstdint>','#include <cstdint>\n#include <algorithm>')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    prepare(Path(__file__).resolve().parents[3],args.destination)
