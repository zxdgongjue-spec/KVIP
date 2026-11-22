"""Isolated schedule probe; preserve TopK, ranking, union and WQE algorithms."""
from pathlib import Path
import shutil
import sys
repo=Path(__file__).resolve().parents[4]
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
shutil.copytree(repo/'csrc',out/'csrc',ignore=shutil.ignore_patterns('build','build_*','*.so','*.o','__pycache__'))
base=out/'csrc/operators/lightning_indexer_with_fetch/op_kernel'
s=(base/'indexer/incremental_methods.h').read_text()
s=s.replace('for(uint32_t b=lane;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes) for(uint32_t q=0;q<4;++q) {','for(uint32_t b=lane;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes) for(uint32_t q=(ctx->windowLimit==8191?epoch:0);q<(ctx->windowLimit==8191?epoch+1:4);++q) {')
s=s.replace('uint32_t end0=Min(uint32_t(ready),cap),end1=half?Min(uint32_t(ready>>32),cap):0;','if(ctx->windowLimit==8191)cap=runtimeShape_.history;\n        uint32_t end0=Min(uint32_t(ready),cap),end1=half?Min(uint32_t(ready>>32),cap):0;')
s=s.replace('incrementalHasNext_=incrementalPipeline_ && incrementalMetadataActive_ && nextB<runtimeShape_.batch;','incrementalHasNext_=ctx->windowLimit!=8191 && incrementalPipeline_ && incrementalMetadataActive_ && nextB<runtimeShape_.batch;')
# Clear other Q candidate counts AFTER metadata commit; earlier Q mappings persist.
marker='    exportWindowCold_=false;\n    // The producer'
s=s.replace(marker,'''    if(ctx->windowLimit==8191){
        auto m=reinterpret_cast<__gm__ uint64_t*>(rotating::TopKHeader(ctx));
        for(uint32_t b=lane;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes)
            for(uint32_t q=0;q<4;++q)if(q!=epoch)m[b*64+q*16+3]=0;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(m),runtimeShape_.batch*512);
    }
    exportWindowCold_=false;
    // The producer''')
(base/'indexer/incremental_methods.h').write_text(s)
decl=(base/'indexer.cpp').read_text().split('namespace {',1)[0]
(out/'window_types.h').write_text(decl+'\n#endif\n')
source=repo/'tests/experiments/final_window/reuse'
for f in ['kernel.cpp','driver.cpp','CMakeLists.txt']:shutil.copy2(source/f,out/f)
k=(out/'kernel.cpp').read_text().replace('op.RotateFetch(s.rounds-1,rotating::SequenceHalfWindow(s));','''auto ts=reinterpret_cast<__gm__ uint64_t*>(a->times);
 ts[0]=GetSystemCycle();
 if(mode==3){for(uint32_t q=0;q<4;++q){ts[1+q*2]=GetSystemCycle();op.RotateFetch(q,rotating::SequenceHalfWindow(s));ts[2+q*2]=GetSystemCycle();}}
 else op.RotateFetch(s.rounds-1,rotating::SequenceHalfWindow(s));
 ts[9]=GetSystemCycle();kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ts),128);''')
(out/'kernel.cpp').write_text(k)
s=(out/'driver.cpp').read_text().replace('c.prefetchLimit=2048;', 'c.prefetchLimit=2048;')
pos=s.index(' std::vector<uint32_t> selected')
s=s[:pos]+' if(mode==3)c.windowLimit=8191;\n'+s[pos:]
s=s.replace(' printf("mode=%u posted=%u bad=%zu', ''' for(unsigned i=0;i<10;++i)printf("stamp%u=%llu\\n",i,(unsigned long long)t[i]);
 printf("mode=%u posted=%u bad=%zu''')
(out/'driver.cpp').write_text(s)
