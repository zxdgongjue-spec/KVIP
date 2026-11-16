"""Create an isolated URMA experiment; leave the checked-out operator unchanged."""
from pathlib import Path
import argparse
import shutil


def replace(path, old, new):
    source = path.read_text()
    if source.count(old) != 1:
        raise RuntimeError(f'Source changed: expected one match in {path}: {old[:100]}')
    path.write_text(source.replace(old, new))



def instrument(dest):
    vector = dest / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer_vector.h'
    replace(vector, '    rotating::RuntimeShape runtimeShape_;',
            '    uint32_t traceCalls_=0; uint64_t traceExtraWait_=0;\n    rotating::RuntimeShape runtimeShape_;')
    replace(vector, '        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);',
            '        bool finalTrace=!prefixActive_ && streamingWorkspace_ && (reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_)->prefetchCount & 0x80000000U);\n'
            '        uint64_t finalBegin=finalTrace?GetSystemCycle():0; traceCalls_=0;traceExtraWait_=0;\n'
            '        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);')
    replace(vector, '        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);\n        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);',
            '        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);\n'
            '        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);\n'
            '        uint64_t finalWait=finalTrace?GetSystemCycle()-finalBegin:0;')
    replace(vector, '        if (validS2Len < topkCount_) {',
            '        if(finalTrace) {\n'
            '            PipeBarrier<PIPE_ALL>(); uint64_t finalEnd=GetSystemCycle();\n'
            '            auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);\n'
            '            uint32_t query=info.indiceOutOffset/topkCount_+curS1Idx+rowIdx;\n'
            '            auto record=reinterpret_cast<__gm__ uint64_t*>(ctx->incremental)+query*16;\n'
            '            record[8]=finalBegin;record[9]=finalEnd;record[10]=finalWait+traceExtraWait_;\n'
            '            record[11]=traceCalls_;record[12]=blockId_;record[13]=1;\n'
            '            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(record),128);\n'
            '        }\n'
            '        if (validS2Len < topkCount_) {')
    vector.write_text(vector.read_text().replace('topkOp_(', '++traceCalls_; topkOp_('))
    methods=vector.with_name('incremental_methods.h')
    if methods.exists():
        text=methods.read_text().replace('    topkOp_(', '    ++traceCalls_; topkOp_(')
        text=text.replace('    if(runtimeShape_.rounds) {',
            '    bool trace=streamingWorkspace_ && (reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_)->prefetchCount & 0x80000000U);\n'
            '    uint64_t waitStart=trace?GetSystemCycle():0;\n    if(runtimeShape_.rounds) {')
        text=text.replace('    uint32_t half=rotating::UseSequenceSplit',
            '    if(trace)traceExtraWait_+=GetSystemCycle()-waitStart;\n    uint32_t half=rotating::UseSequenceSplit')
        methods.write_text(text)
    replace(dest / 'tests/operators/lifetch/support/diagnostics.py',
            '            samples[-1]["algorithm"] = algorithm_after(rotating, algorithm_initial)\n            if diagnostics is not None:',
            '            event_words = rotating.events.cpu().numpy().view(np.uint64).reshape(-1,16)\n'
            '            samples[-1]["incremental_windows"] = [dict(epoch=i//rotating.events.shape[1], lane=i%rotating.events.shape[1], skipped=int(w[3]>>np.uint64(32)), cycles=int(w[14])-int(w[13]), topk_calls=int(w[15]), service_cycles=int(w[1])-int(w[0]), service_wait_cycles=int(w[0])-int(w[4]) if w[0] else 0, service_calls=int(w[1]>0), scan_cycles=int(w[5]), candidate_cycles=int(w[6]), post_cycles=int(w[7])) for i,w in enumerate(event_words)]\n'
            '            samples[-1]["final_topk"] = [dict(query=i, block=int(w[12]), present=int(w[13]), begin=int(w[8]), end=int(w[9]), wait_cycles=int(w[10]), compute_cycles=int(w[9])-int(w[8])-int(w[10]), topk_calls=int(w[11])) for i,w in enumerate(rotating.incremental.cpu().numpy().reshape(-1,16))]\n'
            '            samples[-1]["algorithm"] = algorithm_after(rotating, algorithm_initial)\n            if diagnostics is not None:')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    for directory in ('csrc', 'python', 'tests/operators', 'scripts'):
        shutil.copytree(repo / directory, dest / directory,
                        ignore=shutil.ignore_patterns(
                            '__pycache__', '_custom_opp', 'results',
                            'build', 'build-*', 'build_*', 'CMakeFiles'))
    # Installed LIDU is shared read-only, as in the previous A/B experiments.
    vendor = repo / 'csrc/operators/lidu/_custom_opp'
    if vendor.exists():
        (dest / 'csrc/operators/lidu/_custom_opp').symlink_to(vendor, target_is_directory=True)
    op = dest / 'csrc/operators/lightning_indexer_with_fetch'
    context = op / 'include/context.h'
    replace(context, '    // Optional serving layout:', '    uint64_t incremental; // Experimental per-Q state, 128-byte records.\n    // Optional serving layout:')
    replace(context, 'sizeof(Context) == 320', 'sizeof(Context) == 328')
    replace(op / 'op_kernel/lifecycle.cpp',
            '    FillRoundWords(ctx->rows+b*sizeof(Row),sizeof(Row)/4,local);',
            '    FillRoundWords(ctx->incremental+b*512,128,local);\n'
            '    FillRoundWords(ctx->rows+b*sizeof(Row),sizeof(Row)/4,local);')
    binding = dest / 'python/kvip/operators/lifetch/binding.py'
    replace(binding, '("context_bytes", 320)', '("context_bytes", 328)')
    replace(binding, '        self.context = torch.from_numpy(blob.copy()).to(self.device)',
            '        self.incremental = torch.zeros((batch, 4, 16), dtype=torch.int64, device=self.device)\n'
            '        blob = np.concatenate((blob, np.array([self.incremental.data_ptr()], dtype="<u8").view("u1")))\n'
            '        self.context = torch.from_numpy(blob.copy()).to(self.device)')
    if args.baseline:
        instrument(dest)
        print(f'Instrumented baseline: {dest}')
        return
    vector = op / 'op_kernel/indexer/indexer_vector.h'
    for name in ('methods.h', 'vector_helpers.h'):
        shutil.copyfile(Path(__file__).with_name(name), vector.parent / ('incremental_' + name))
    replace(vector, '#include "topk.h"', '#include "topk.h"\n#include "incremental_vector_helpers.h"')
    replace(vector, '    rotating::RuntimeShape runtimeShape_;',
            '    rotating::RuntimeShape runtimeShape_;\n#include "incremental_methods.h"')
    replace(vector, '    if (!topkOnly) Duplicate(kScaleUB_',
            '    pipe->InitBuffer(incrementalIdsBuf_, trunkLen_*2 + ((topkOnly && streamingWorkspace_) ? 0 : 2048*4));\n'
            '    incrementalNew_ = incrementalIdsBuf_.Get<uint16_t>();\n'
            '    incrementalOld_ = (topkOnly && streamingWorkspace_) ? streamingOutput_ : incrementalNew_[trunkLen_].template ReinterpretCast<uint32_t>();\n'
            '    if (!topkOnly) Duplicate(kScaleUB_')
    replace(vector, '        if (publishProgress) {', '        SeedIncremental(query,scoreRow);\n        if (publishProgress) {')
    replace(vector, '        // Join this producer\'s hit publication. Other producers remain free.',
            '        if(!AdvanceWindow(epoch,sequenceHalfWindow))return;\n        // Join this producer\'s hit publication. Other producers remain free.')
    replace(vector, '        if (validS2Len >= topkCount_) {',
            '        if (!prefixActive_ && validS2Len >= runtimeShape_.resident && validS2Len % 128 == 0) {\n'
            '            FinalIncremental(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx, validS2Len);\n'
            '        } else if (validS2Len >= topkCount_) {')
    # Do not change LI bitmap encoding, slot allocation, WQE or CQ handling.
    # Reject cold prefetch candidates below every currently observed Q cutoff.
    prefetch = op / 'op_kernel/prefetch.h'
    replace(prefetch, '__gm__ uint32_t* map,uint32_t count) {\n    for(uint32_t i=threadIdx.x;i<count;i+=256)slots[i]=asc_ldcg(map+ids[i]);',
            '''__gm__ uint32_t* map,uint32_t count,__gm__ Context* ctx,uint32_t batch,uint32_t resident,uint32_t half) {
    uint64_t sources[4];uint32_t low[4],high[4],cutoff[4];
    for(uint32_t q=0;q<4;++q) {
        auto meta=reinterpret_cast<__gm__ uint64_t*>(ctx->incremental)+(batch*4+q)*16;
        sources[q]=asc_ldcg(meta);uint64_t cursor=asc_ldcg(meta+1);
        low[q]=uint32_t(cursor);high[q]=uint32_t(cursor>>32);cutoff[q]=uint32_t(asc_ldcg(meta+2));
    }
    for(uint32_t i=threadIdx.x;i<count;i+=256) {
        uint32_t token=ids[i],value=asc_ldcg(map+token);
        if(value==UINT32_MAX && token>=resident) {
            bool selected=false;
            for(uint32_t q=0;q<4;++q) {
                uint64_t source=sources[q];uint32_t end=low[q];
                if(half && ((token-resident)/half)%2)end=high[q];
                if(source && token<end) {
                    uint32_t packed=asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(source)+token/2);
                    uint32_t score=(packed>>((token%2)*16))&65535;
                    selected |= score>=cutoff[q];
                }
            }
            if(!selected)value=1; // Already-resolved sentinel for CompactMisses.
        }
        slots[i]=value;
    }''')
    replace(prefetch, 'if(!slots && !ctx->tailBounds) {', 'if(false) { // Incremental candidates always need the current TopK cutoff filter.')
    replace(prefetch, 'reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),map,candidateCount);',
            'reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),map,candidateCount,ctx,b,rt_resident,sequenceHalfWindow);')
    replace(prefetch, '        const uint32_t lower=ctx->lruSlots?0:rt_resident;',
            '        uint32_t previousEpoch=epoch;\n'
            '        while(previousEpoch && events[(previousEpoch-1)*rt_serviceLanes+lane].skipped)--previousEpoch;\n'
            '        const uint32_t lower=ctx->lruSlots?0:rt_resident;')
    replace(dest / 'csrc/operators/common/lifetch/sequence_candidates.h',
            '(epoch?state.GetValue(80+(epoch-1)*4+q):0)',
            '(previousEpoch?state.GetValue(80+(previousEpoch-1)*4+q):0)')
    # Expose only observed score boundaries: producers may advance while the
    # service core computes TopK. Later tiles belong to the following update.
    replace(prefetch, 'uint32_t end=work.GetValue(progress),cap=windowEnd;',
            'auto incremental=reinterpret_cast<__gm__ uint64_t*>(ctx->incremental)+(b*4+q)*16;\n'
            '            uint64_t cursor=incremental[1];\n'
            '            uint32_t end=(work.GetValue(progress)<uint32_t(cursor)?work.GetValue(progress):uint32_t(cursor)),cap=windowEnd;')
    replace(prefetch, 'uint32_t high=work.GetValue(progress+16);',
            'uint32_t high=(work.GetValue(progress+16)<uint32_t(cursor>>32)?work.GetValue(progress+16):uint32_t(cursor>>32));')
    instrument(dest)
    print(f'Experimental URMA source: {dest}')
    print('Bitmap producer, LI tiling, resolver, Index update and KVAF transport are unchanged.')


if __name__ == '__main__':
    main()
