"""A = accepted ABC baseline; B = rank-filtered, per-Q budgeted prefetch."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one patch anchor: {old[:100]!r}')
    return text.replace(old, new, 1)



def integrate_tags(kernel):
    # Admission bitmap starts empty; hot/reserved metadata is not read by init.
    lifecycle=kernel/'lifecycle.cpp';c=lifecycle.read_text()
    begin=c.index('    uint32_t resident=ctx->shape.resident;',c.index('__aicore__ inline void InitializeRound'))
    end=c.index('    pipe.Destroy();',begin);c=c[:begin]+c[end:]
    anchor='    FillRoundWords(ctx->rows+b*sizeof(Row),sizeof(Row)/4,local);'
    c=replace_once(c,anchor,'''    FillRoundWords(kvip_prefetch_rank::CoveredAddress(ctx->incremental,rt_batch,rt_history,b),(rt_history+255)/256*8,local);
'''+anchor);lifecycle.write_text(c)
    prefetch=kernel/'prefetch.h';c=prefetch.read_text()
    begin=c.index('    auto covered=',c.index('inline void InitializeTailMappings'))
    end=c.index('    asc_threadfence();',begin);c=c[:begin]+c[end:]
    # Reuse spare fields in the existing per-request submit metadata; avoid
    # reading Context again inside each WQE lane solely for the bitmap base.
    c=replace_once(c,'kvip_prefetch_rank::CoveredAddressSimt(\n                        ctx->incremental,rt_batch,rt_history,b)', 'input64[37]')
    anchor='                post64.SetValue(35,output);post64.SetValue(36,queueAddress);'
    c=replace_once(c,anchor,anchor+'\n                post64.SetValue(37,kvip_prefetch_rank::CoveredAddress(ctx->incremental,rt_batch,rt_history,b));')
    # Union produces exactly the budgeted new-admission bits below. Its bits
    # are disjoint from prior admissions, so an integer atomic DMA add is OR.
    # Queue validation is unchanged; invalid submissions do not publish bits.
    anchor='                    asc_atomic_or(reinterpret_cast<__gm__ uint32_t*>(input64[37])+token/32,1U<<(token%32));'
    c=replace_once(c,anchor,'')
    anchor='                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);\n                GlobalTensor<uint32_t> jobs;'
    c=replace_once(c,anchor,'''                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                bool bulk=count<postInput.GetValue(19) && pending+1<postInput.GetValue(31);
                if(bulk){
                    GlobalTensor<int32_t> admitted;
                    admitted.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(post64.GetValue(37)));
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID5);WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
                    SetAtomicAdd<int32_t>();DataCopy(admitted,scratch.ReinterpretCast<int32_t>(),rt_history/32);SetAtomicNone();
                    SetFlag<HardEvent::MTE3_S>(EVENT_ID5);WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
                }
                GlobalTensor<uint32_t> jobs;''')
    c=replace_once(c,'uint32_t count,uint32_t slotBase,__gm__ Context* ctx,uint32_t batch) {',
                   'uint32_t count,uint32_t slotBase,__gm__ Context* ctx,uint32_t batch,__gm__ uint32_t* admitted) {')
    c=replace_once(c,'reinterpret_cast<__gm__ uint32_t*>(kvip_prefetch_rank::CoveredAddressSimt(\n            ctx->incremental,ctx->shape.batch,ctx->shape.history,batch))', 'admitted')
    c=replace_once(c,'count,rt_resident+slots,ctx,b);',
                   'count,rt_resident+slots,ctx,b,reinterpret_cast<__gm__ uint32_t*>(kvip_prefetch_rank::CoveredAddress(ctx->incremental,rt_batch,rt_history,b)));')
    c=replace_once(c,'#include "topk_candidate_service.h"','''        uint32_t available=PrefetchAllowance(slots,requests,state.GetValue(208),
            state.GetValue(209),state.GetValue(210),ctx->windowLimit);
#include "topk_candidate_service.h"''')
    begin=c.index('            uint32_t available=PrefetchAllowance',c.index('        uint32_t count=0,missing=0;'))
    end=c.index('            V9_FETCH_PROFILE(candidateTicks',begin)
    c=c[:begin]+'''            // ExportWindowCold already removed local hits and prior admissions.
            // AdvanceWindow joined the predecessor; this request has no other writer.
            missing=candidateCount;count=missing<available?missing:available;
'''+c[end:]
    prefetch.write_text(c)
    candidates=kernel/'topk_candidates.h';c=candidates.read_text()
    c=replace_once(c,'uint32_t n0,uint32_t n1,uint32_t n2,uint32_t n3) {',
        'uint32_t n0,uint32_t n1,uint32_t n2,uint32_t n3,uint32_t limit,uint32_t words,bool last) {')
    c=replace_once(c,'    asc_syncthreads();\n}', '''    asc_syncthreads();
    // Earlier chunks still need all bits for dedup. Prune only after the last.
    if(last && limit && cfg[40]>limit) {
        for(uint32_t i=tid;i<words;i+=256)bits[i]=0;
        asc_syncthreads();
        for(uint32_t i=tid;i<limit;i+=256){uint32_t token=ids[i];asc_atomic_or(bits+token/32,1U<<(token%32));}
        asc_syncthreads();
    }
}''');candidates.write_text(c)
    service=kernel/'topk_candidate_service.h';c=service.read_text()
    c=replace_once(c,'width,n[0],n[1],n[2],n[3]);',
        'width,n[0],n[1],n[2],n[3],available<kCandidateCapacity?available:kCandidateCapacity,rt_history/32,start+256>=maxCount);')
    service.write_text(c)
    vector=kernel/'indexer/indexer_vector.h';c=vector.read_text()
    c=replace_once(c,'        MergeRecentScores(query,scoreRow,availableEnd);','        MergeRecentScores(query,scoreRow,availableEnd);\n        TagResident(query);')
    # Reuse the existing first-half metadata prefetch in the prefix TopK.
    c=replace_once(c,' || prefixActive_ || runtimeShape_.resident!=16384',' || runtimeShape_.resident!=16384')
    begin=c.index('    GlobalTensor<uint32_t> validGm,reservedGm;',c.index('::EmitFinalClassification'))
    end=c.index('    if(profile)liTicks[10]',begin)
    original=c[begin:end].replace('const bool overlapSecond =', 'overlapSecond =')
    c=c[:begin]+'''    bool overlapSecond=false;
    if(taggedLocations_) {
        if(profile)liTicks[3]=GetSystemCycle();
        Simt::VF_CALL<rotating::ClassifyPreparedLocations>(Simt::Dim3{512,1,1},runtimeShape_.history,runtimeShape_.resident,b,ctx,
            reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(result.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(bitmap.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(bitmap[bitmapWords].GetPhyAddr()));
        if(profile){SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);liTicks[4]=liTicks[5]=liTicks[6]=GetSystemCycle();}
        taggedLocations_=false;
    } else {
'''+original+'    }\n'+c[end:]
    vector.write_text(c)
    methods=kernel/'indexer/incremental_methods.h';c=methods.read_text()
    helper='''bool taggedLocations_=false;
__aicore__ inline void TagResident(uint32_t query) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    auto work=reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_);
    bool profile=streamingWorkspace_ && (work->prefetchCount&0x80000000U);
    auto ticks=streamingWorkspace_?reinterpret_cast<__gm__ uint64_t*>(work->times)+uint64_t(runtimeShape_.batch)*64+uint64_t(query)*16:nullptr;
    if(profile)ticks[12]=GetSystemCycle();
    uint32_t b=query/4,owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
    uint32_t tailBegin=0,tailEnd=0;
    if(ctx->tailBounds){auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;tailBegin=bounds[0];tailEnd=bounds[1];
        uint32_t capacity=(ctx->outputStride?ctx->outputStride:runtimeShape_.resident)-runtimeShape_.resident;
        tailEnd=Min(tailEnd,Min(runtimeShape_.history,tailBegin+capacity));}
    LocalTensor<uint32_t> hot,reserved;uint32_t chunk;
    if(streamingTopkOnly_){hot=streamingValid_;reserved=streamingReserved_;chunk=8192;}
    else {PipeBarrier<PIPE_ALL>();hot=resMm1UB_.template ReinterpretCast<uint32_t>();reserved=mrgValueLocal_.template ReinterpretCast<uint32_t>();chunk=4096;}
    GlobalTensor<uint32_t> h,r;
    if(ctx->hotValid)h.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*runtimeShape_.resident);
    if(ctx->reservedHot)r.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot)+uint64_t(b)*runtimeShape_.resident);
    for(uint32_t base=0;base<runtimeShape_.resident;base+=chunk){
        if(base){SetFlag<HardEvent::V_MTE2>(EVENT_ID4);WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);}
        uint32_t n=Min(chunk,runtimeShape_.resident-base);
        if(!base && streamingTopkOnly_ && streamingPrefetched_) {}
        else {if(ctx->hotValid)DataCopy(hot,h[base],n);if(ctx->reservedHot)DataCopy(reserved,r[base],n);}
        if(!ctx->hotValid)Duplicate(hot,uint32_t(1),n);if(!ctx->reservedHot)Duplicate(reserved,UINT32_MAX,n);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        kvip_prefetch_rank::TagResidentHalf(reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(hot.GetPhyAddr()),reinterpret_cast<__ubuf__ int32_t*>(reserved.GetPhyAddr()),base,n,tailBegin,tailEnd);
    }
    PipeBarrier<PIPE_V>();streamingPrefetched_=false;
    if(profile){SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);ticks[13]=GetSystemCycle();kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks),128);}
}
'''
    c=helper+c
    start=c.index('__aicore__ inline void FinalIncremental')
    anchor='    SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);'
    i=c.index(anchor,start)+len(anchor)
    c=c[:i]+'''
    auto initial=streamingTopkOnly_?streamingOutput_:topkSharedTmpLocal_;
    kvip_prefetch_rank::RestorePublicIds(reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(initial.GetPhyAddr()),runtimeShape_.resident);
    PipeBarrier<PIPE_V>();taggedLocations_=true;
'''+c[i:];methods.write_text(c)
    classifier=kernel/'staged_topk_classify.h';c=classifier.read_text()
    helper='''
namespace rotating {
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ClassifyPreparedLocations(uint32_t history,uint32_t resident,uint32_t batch,__gm__ Context* ctx,
    __ubuf__ uint32_t* indices,__ubuf__ uint32_t* locations,__ubuf__ uint32_t* bitmap,__ubuf__ uint32_t* error) {
    const uint32_t words=(resident+31)/32,shards=4096/words<8?4096/words:8;
    auto mapping=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(batch)*history;
    auto lru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+batch*kLruSlots;
    for(uint32_t i=threadIdx.x;i<kTop;i+=512){
        uint32_t token=indices[i],slot=locations[i];
        if(token==UINT32_MAX){locations[i]=UINT32_MAX;continue;}
        if(token>=history){locations[i]=UINT32_MAX;asc_atomic_or(error,1U);continue;}
        if(slot==UINT32_MAX)slot=ResolvePhysicalSlot(resident,lru,asc_ldcg(mapping+token));
        locations[i]=slot;
        if(slot<resident)asc_atomic_or(bitmap+(threadIdx.x%shards)*words+slot/32,1U<<(slot%32));
    }
    asc_syncthreads();
}
}
'''
    classifier.write_text(c+helper)


def instrument_miss(kernel, output):
    methods=kernel/'indexer/incremental_methods.h';s=methods.read_text()
    s=replace_once(s,'bool exportWindowCold_=false;', 'bool exportWindowCold_=false;bool rankTrace_=false;')
    s=replace_once(s,'exportWindowCold_=true;rankedEpoch_=epoch;', 'exportWindowCold_=true;rankedEpoch_=epoch;rankTrace_=trace;')
    begin=s.index('    uint32_t count=kvip_prefetch_rank::Select')
    end=s.index('    mappedExport_=true;',begin)
    selection=s[begin:end].replace('nullptr,','rankTrace_?rankClocks:nullptr,')
    s=s[:begin]+'''    uint64_t rankClocks[5];
'''+selection+'''    if(rankTrace_) {
        uint64_t delta[3]={rankClocks[1]-rankClocks[0],rankClocks[4]-rankClocks[1],1};
        for(uint32_t k=0;k<3;++k) {
            if(incrementalMetadataActive_) {
                uint32_t i=incrementalMetadataIndex_*16+5+k;
                incrementalMetadata_.SetValue(i,incrementalMetadata_.GetValue(i)+delta[k]);
            } else {
                auto meta=IncrementalMeta(query);meta[5+k]+=delta[k];
            }
        }
    }
'''+s[end:];methods.write_text(s)
    path=output/'tests/operators/lifetch/support/diagnostics.py';s=path.read_text()
    anchor=next(line for line in s.splitlines() if 'samples[-1]["final_topk"] =' in line)
    s=replace_once(s,anchor,'''            samples[-1]["rank_selection"] = [dict(query=i, check_pack_cycles=int(w[5]), rank_emit_cycles=int(w[6]), calls=int(w[7])) for i,w in enumerate(rotating.incremental.cpu().numpy().reshape(-1,16))]
'''+anchor);path.write_text(s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--tagged-miss', action='store_true', help='Carry prefix local residency in private IDs; bitmap tracks admissions only')
    parser.add_argument('--bitmap-miss', action='store_true', help='Use maintained covered bitmap instead of per-Q indexed GM checks')
    parser.add_argument('--true-miss', action='store_true', help='Filter hot hits and prior admissions before rank/quota selection')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--minimum-half', action='store_true',
                        help='Lower the score threshold to admit quota/2 eligible candidates')
    parser.add_argument('--abc-tail-tokens', type=int, default=384)
    args = parser.parse_args()
    if args.tagged_miss: args.bitmap_miss=True
    if args.bitmap_miss: args.true_miss=True
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here/'prepare_recent.py'), str(args.output),
                    '--abc-tail-tokens', str(args.abc_tail_tokens)], check=True)
    if args.baseline:
        return
    kernel = args.output/'csrc/operators/lightning_indexer_with_fetch/op_kernel'
    for name in ('rank_threshold.h', 'ranked_candidates.h'):
        shutil.copyfile(here/name, kernel/'indexer'/name)
    vector = kernel/'indexer/window_fast_vector.h'
    v = vector.read_text()
    v = replace_once(v, 'uint32_t resident,uint32_t valid){', 'uint32_t resident,uint32_t valid,bool exportCold=true){')
    begin = v.index('  CompareScalar<uint32_t,CMPMODE::GE>(isCold,mapped')
    end = v.index('\n }', begin)
    v = v[:begin]+'  if(exportCold) {\n'+v[begin:end]+'\n  }'+v[end:]
    v = replace_once(v, ' StoreUnAlignPost(cold,store);ReduceMin', ' if(exportCold)StoreUnAlignPost(cold,store);ReduceMin')
    vector.write_text(v+'\n#include "ranked_candidates.h"\n')
    methods = kernel/'indexer/incremental_methods.h'
    s = methods.read_text()
    start = s.index('    auto compact=mappedExport_?')
    end = s.index('    if(incrementalMetadataActive_)', start)
    s = s[:start]+'''    // The complete Top2048 remains untouched for the next incremental window.
    auto compact=incrementalNew_.template ReinterpretCast<uint32_t>();
    uint32_t rank=runtimeShape_.rounds>1?
        1024+(1024*rankedEpoch_+(runtimeShape_.rounds-1)/2)/(runtimeShape_.rounds-1):2048;
    uint32_t count=kvip_prefetch_rank::Select(indicesOutLocal_,scoreOutLocal_,compact,
        topkSharedTmpLocal_[128],runtimeShape_.resident,valid,rank,ctx->windowLimit/4);
    mappedExport_=true;
'''+s[end:]
    s=replace_once(s,'bool exportWindowCold_=false;', 'bool exportWindowCold_=false;\nuint32_t rankedEpoch_=0;')
    s=replace_once(s,'exportWindowCold_=epoch!=0;', 'exportWindowCold_=true;rankedEpoch_=epoch;')
    s=replace_once(s,'PipeBarrier<PIPE_ALL>();DataCopy(indicesOutLocal_,saved,2048);',
        '''GlobalTensor<uint16_t> savedScores;
            savedScores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(WindowTopkPrefix()+uint64_t(query/4)*prefixBytes)+(query%4)*2048);
            PipeBarrier<PIPE_ALL>();DataCopy(indicesOutLocal_,saved,2048);DataCopy(scoreOutLocal_,savedScores,2048);''')
    s=replace_once(s,'runtimeShape_.resident,mappedValid_);','runtimeShape_.resident,mappedValid_,!exportWindowCold_);')
    if args.minimum_half:
        s=replace_once(s,'valid,rank,ctx->windowLimit/4);',
                       'valid,rank,ctx->windowLimit/4,true,nullptr,(ctx->windowLimit/4)/2);')
    if args.true_miss:
        start=s.index('    uint32_t count=kvip_prefetch_rank::Select(')
        end=s.index('    mappedExport_=true;',start)
        s=s[:start]+'''    uint32_t b=query/4;
    uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
    auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(b)*runtimeShape_.history;
    auto hot=ctx->hotValid?reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*runtimeShape_.resident:nullptr;
    auto reserved=ctx->reservedHot?reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+uint64_t(b)*runtimeShape_.resident:nullptr;
    uint32_t count=kvip_prefetch_rank::SelectMiss(indicesOutLocal_,scoreOutLocal_,compact,
        topkSharedTmpLocal_[128],map,hot,reserved,runtimeShape_.resident,valid,
        rank,ctx->windowLimit/4,nullptr,'''+('(ctx->windowLimit/4)/2' if args.minimum_half else '0')+');\n'+s[end:]
    if args.bitmap_miss:
        start=s.index('    uint32_t b=query/4;')
        end=s.index('    mappedExport_=true;',start)
        s=s[:start]+'''    uint32_t b=query/4;
    auto covered=reinterpret_cast<__gm__ uint32_t*>(kvip_prefetch_rank::CoveredAddress(
        ctx->incremental,runtimeShape_.batch,runtimeShape_.history,b));
    uint32_t count=kvip_prefetch_rank::SelectBitmap(indicesOutLocal_,scoreOutLocal_,compact,
        topkSharedTmpLocal_[128],covered,runtimeShape_.history,valid,rank,ctx->windowLimit/4,nullptr,'''+('(ctx->windowLimit/4)/2' if args.minimum_half else '0')+');\n'+s[end:]
    methods.write_text(s)
    prefetch=kernel/'prefetch.h';s=prefetch.read_text()
    start=s.index('        if(!epoch) {',s.index('uint32_t candidateCount=0;'))
    end=s.index('#include "topk_candidate_service.h"',start)+len('#include "topk_candidate_service.h"\n        }')
    s=s[:start]+'#include "topk_candidate_service.h"'+s[end:]
    if args.bitmap_miss:
        s=s.replace('#include "context.h"', '#include "context.h"\n#include "indexer/ranked_candidates.h"')
        # Map publication and bitmap publication belong to the same producer
        # epoch; AdvanceWindow already waits for its predecessor event.
        anchor='                    asc_stcg(map+token,slot);'
        s=replace_once(s,anchor,anchor+'''
                    asc_atomic_or(reinterpret_cast<__gm__ uint32_t*>(kvip_prefetch_rank::CoveredAddressSimt(
                        ctx->incremental,rt_batch,rt_history,b))+token/32,1U<<(token%32));''')
        anchor='slots[i]=PhysicalSlot(rt_resident,ctx,batch,slot);asc_stcg(map+token,slot);'
        s=replace_once(s,anchor,anchor+'''
        asc_atomic_or(reinterpret_cast<__gm__ uint32_t*>(kvip_prefetch_rank::CoveredAddressSimt(
            ctx->incremental,ctx->shape.batch,ctx->shape.history,batch))+token/32,1U<<(token%32));''')
        anchor='    asc_threadfence();\n}\n\n// Resolver workers'
        if anchor not in s: raise RuntimeError('Tail mapping publication changed')
        s=s.replace(anchor,'''    auto covered=reinterpret_cast<__gm__ uint32_t*>(kvip_prefetch_rank::CoveredAddressSimt(
        ctx->incremental,ctx->shape.batch,rt_history,b));
    if(end>rt_history)end=rt_history;
    if(end>begin && end-begin>tailTokens)end=begin+tailTokens;
    for(uint32_t word=begin/32+threadIdx.x;begin<end && word<(end+31)/32;word+=256){
        uint32_t low=begin>word*32?begin-word*32:0;
        uint32_t high=end-word*32;if(high>32)high=32;
        uint32_t mask=(UINT32_MAX<<low)&(UINT32_MAX>>(32-high));
        asc_atomic_or(covered+word,mask);
    }
'''+anchor)
        lifecycle=kernel/'lifecycle.cpp';c=lifecycle.read_text()
        anchor='    pipe.Destroy();\n}\n#endif\nextern "C" __global__ __aicore__ void v9_init_kernel'
        init='''    uint32_t resident=ctx->shape.resident;
    uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
    auto hot=ctx->hotValid?reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*resident:nullptr;
    auto reserved=ctx->reservedHot?reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+uint64_t(b)*resident:nullptr;
    kvip_prefetch_rank::InitializeCovered(local,reinterpret_cast<__gm__ uint32_t*>(
        kvip_prefetch_rank::CoveredAddress(ctx->incremental,rt_batch,rt_history,b)),hot,reserved,resident,rt_history);
'''
        c=replace_once(c,anchor,init+anchor);lifecycle.write_text(c)
        binding=args.output/'python/kvip/operators/lifetch/binding.py';c=binding.read_text()
        c=replace_once(c,'torch.empty(batch*(512+4*2048*6+1024),', 'torch.empty(batch*(512+4*2048*6+1024+(history+255)//256*32),')
        c=replace_once(c,'        self.window_topk_storage =','''        if history > 131072 or resident % 512:
            raise ValueError("Ranked bitmap experiment requires history<=131072 and resident aligned to 512")
        self.window_topk_storage =''')
        binding.write_text(c)
    prefetch.write_text(s)
    if args.tagged_miss:
        integrate_tags(kernel)
        path=args.output/'tests/operators/lifetch/support/execution.py';c=path.read_text()
        c=replace_once(c,'overlap=bool(t[8]), topk_only=bool(t[9])))',
                       'overlap=bool(t[8]), topk_only=bool(t[9]), prefix_tag_cycles=int(t[13])-int(t[12]) if t[12] else 0))')
        path.write_text(c)
    if args.true_miss: instrument_miss(kernel,args.output)
    # Cold-only window 0 deliberately does not prefetch former resident tokens.
    # Preserve the cold-start/hot-victim KV, Index and ownership checks, but
    # remove the assertion that the old hot-prefetch policy must be exercised.
    execution=args.output/'tests/operators/lifetch/support/execution.py'
    e=execution.read_text()
    begin=e.index('    if not bench_only and not compact and args.prefetch_tokens and args.miss_percent < 100:')
    end=e.index('    if getattr(args, "check_suite", None) == "boundary":',begin)
    e=e[:begin]+e[end:]
    execution.write_text(e)
    print('Ranked prefetch: all windows; rank 1024..2048; per-Q quota=windowLimit/4; '
          f'minimum={"quota/2" if args.minimum_half else "0"}; '
          f'candidates={"unresolved miss before rank/quota" if args.true_miss else "cold before mapping"}; no quota redistribution')


if __name__ == '__main__':
    main()
