from pathlib import Path
import shutil
import subprocess
import sys
r=Path(__file__).resolve().parents[4]
o=Path(sys.argv[1])
subprocess.run([sys.executable,str(Path(__file__).with_name('prepare.py')),str(o)],check=True)
b=o/'csrc/operators/lightning_indexer_with_fetch/op_kernel'
s=(r/'csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/incremental_methods.h').read_text()
s=s.replace('incrementalHasNext_=incrementalPipeline_ && incrementalMetadataActive_ && nextB<runtimeShape_.batch;','incrementalHasNext_=ctx->windowLimit!=8191 && incrementalPipeline_ && incrementalMetadataActive_ && nextB<runtimeShape_.batch;')
s=s.replace('        AdvanceIncremental(b*4+q,end0,end1,half);','''        auto probeTimes=reinterpret_cast<__gm__ uint64_t*>(reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_)->times);
        if(ctx->windowLimit==8191)probeTimes[16+q*4]=GetSystemCycle();
        AdvanceIncremental(b*4+q,end0,end1,half);
        if(ctx->windowLimit==8191){
            JoinColdOutput();
            if(incrementalOutputPending_){WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);incrementalOutputPending_=false;}
            probeTimes[17+q*4]=GetSystemCycle();
            auto gmMeta=reinterpret_cast<__gm__ uint64_t*>(rotating::TopKHeader(ctx));
            for(uint32_t x=0;x<4;++x)gmMeta[b*64+x*16+3]=x==q?incrementalMetadata_.GetValue(incrementalMetadataIndex_*16+3):0;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(gmMeta+b*64),512);
            PipeBarrier<PIPE_ALL>();
            probeTimes[18+q*4]=GetSystemCycle();
            rotating::Service(runtimeShape_.batch,runtimeShape_.history,runtimeShape_.resident,runtimeShape_.splitQ,runtimeShape_.rounds,runtimeShape_.serviceAics,runtimeShape_.serviceLanes,runtimeShape_.tokenBytes,runtimeShape_.postThreads,runtimeShape_.postBank,runtimeShape_.batchedRows,runtimeShape_.scanTokens,rotationContext_,q,blockId_/2,blockId_%2,topkSharedTmpLocal_,mrgValueLocal_,indicesOutLocal_,half,false);
            probeTimes[19+q*4]=GetSystemCycle();
        }''')
(b/'indexer/incremental_methods.h').write_text(s)
p=b/'indexer/indexer_vector.h';s=p.read_text().replace('AdvanceWindow(epoch,sequenceHalfWindow);','AdvanceWindow(epoch,sequenceHalfWindow);\n        if(reinterpret_cast<__gm__ rotating::Context*>(rotationContext_)->windowLimit==8191)return;');p.write_text(s)
p=o/'kernel.cpp';s=p.read_text();start=s.index(' if(mode==3)');end=s.index(' ts[9]',start);s=s[:start]+' op.RotateFetch(s.rounds-1,rotating::SequenceHalfWindow(s));\n'+s[end:];s=s.replace('ts),128','ts),256');p.write_text(s)
p=o/'driver.cpp';s=p.read_text().replace('i<10;++i','i<32;++i').replace('get<uint64_t>(a.times,16)','get<uint64_t>(a.times,32)');p.write_text(s)
