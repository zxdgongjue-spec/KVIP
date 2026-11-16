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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--minimum-half', action='store_true',
                        help='Before dedup/mapping, lower the score threshold to admit quota/2 cold candidates')
    parser.add_argument('--abc-tail-tokens', type=int, default=384)
    args = parser.parse_args()
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
    methods.write_text(s)
    prefetch=kernel/'prefetch.h';s=prefetch.read_text()
    start=s.index('        if(!epoch) {',s.index('uint32_t candidateCount=0;'))
    end=s.index('#include "topk_candidate_service.h"',start)+len('#include "topk_candidate_service.h"\n        }')
    s=s[:start]+'#include "topk_candidate_service.h"'+s[end:]
    prefetch.write_text(s)
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
          f'minimum={"quota/2" if args.minimum_half else "0"} before dedup/mapping; no quota redistribution')


if __name__ == '__main__':
    main()
