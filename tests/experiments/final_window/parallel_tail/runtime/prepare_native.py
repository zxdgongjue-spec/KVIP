"""Generate the private native terminal translation unit for integration.

Not a hardware runner: lifecycle/public-output qualification is still required.
The producer LI translation unit is never rewritten by this generator.
"""
import argparse
from pathlib import Path
import shutil

from index_layout import adapt_index_split
from pack_layout import adapt_pack_layout
from prepare_prefix_case import prepare
from private_topk import adapt, replace_once


HERE = Path(__file__).resolve().parent


def generate(destination):
    prepare(destination)
    adapt(destination / 'csrc/operators/lightning_indexer_with_fetch/op_kernel')
    for source, name in [('abi.h', 'terminal_runtime_abi.h'),
                         ('transport.h', 'terminal_transport.h'),
                         ('output_routes.h', 'terminal_output_routes.h'),
                         ('finish.h', 'terminal_finish.h'),
                         ('host.cpp', 'host.cpp')]:
        shutil.copy2(HERE / source, destination / name)
    path = destination / 'kernel.cpp'
    text = path.read_text()
    text = '#include "terminal_runtime_abi.h"\n' + text
    text = replace_once(text, '#include "parallel_post.h"', '''#include "parallel_post.h"
#include "terminal_output_routes.h"
#include "terminal_finish.h"
#include "staged_fetch_worker.h"
namespace rotating {
#include "terminal_transport.h"
}''')
    text = replace_once(text, 'void test_service(GM_ADDR arg,uint32_t mode)',
                        'void v9_parallel_terminal(GM_ADDR binding)')
    text = replace_once(text,
        ' auto a=reinterpret_cast<__gm__ pipeline_ab::Args*>(arg);',
        ' auto runtime=reinterpret_cast<__gm__ terminal_runtime::Binding*>(binding);\n'
        ' GM_ADDR arg=reinterpret_cast<GM_ADDR>(runtime->pipeline);\n'
        ' auto a=reinterpret_cast<__gm__ pipeline_ab::Args*>(arg);')
    text = replace_once(text,
        ' bool profile=(a->prefetchCount&0x80000000U)!=0;uint64_t postedAt=0;',
        ' bool profile=(runtime->options&1U)!=0;')
    text = replace_once(text,
        ' uint32_t q=GetBlockIdx();auto flags=reinterpret_cast<__gm__ uint32_t*>(a->flags);auto ts=reinterpret_cast<__gm__ uint64_t*>(a->times)+q*32;',
        ''' uint32_t b=GetBlockIdx()/4,q=GetBlockIdx()%4,query=b*4+q;
 auto flags=reinterpret_cast<__gm__ uint32_t*>(runtime->flags)+terminal_runtime::RowFlagOffset(b);
 auto ts=reinterpret_cast<__gm__ uint64_t*>(runtime->times)+terminal_runtime::QueryTraceOffset(b,q);
 const uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
 auto update=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
 const uint64_t addressStride=update[7];
 auto args=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(runtime->queueArgs)+query;
 auto post=reinterpret_cast<__gm__ rotating::FinalPost*>(ctx->owners+s.finalPostOffset)+query;''')
    text = replace_once(text, ' uint32_t generation=flags[q*32+1]+1;',
                        ' kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+q*32),128);\n'
                        ' uint32_t generation=flags[q*32+1]+1;')
    # Every model callback gets the live row index. Never clone/rebase Context:
    # its original batch is required by TopK scratch and incremental layouts.
    text = text.replace('uint32_t count,uint32_t history){', 'uint32_t count,uint32_t history,uint32_t b){')
    text = text.replace('uint32_t q,uint32_t count){', 'uint32_t q,uint32_t count,uint32_t b){')
    text = text.replace('uint32_t priorJournalCount){', 'uint32_t priorJournalCount,uint32_t b){')
    text = text.replace('RetireShared(ctx,first,count,history);', 'RetireShared(ctx,first,count,history,b);')
    text = text.replace('RetirePriorJournal(ctx,history,q,priorJournalCount);', 'RetirePriorJournal(ctx,history,q,priorJournalCount,b);')
    text = text.replace('reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)',
                        '(reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(b)*history)')
    # Main body has s.history rather than callback parameter history.
    start = text.index('extern "C" __global__')
    text = text[:start] + text[start:].replace('uint64_t(b)*history', 'uint64_t(b)*s.history')
    for typ, member, stride in [('uint32_t', 'changes', 'rotating::kChangeCapacity'),
                              ('uint32_t', 'changeSlots', 'rotating::kChangeCapacity'),
                              ('uint32_t', 'lruSlots', 'rotating::kLruSlots'),
                              ('rotating::MappingJournal', 'mappingJournal', '1'),
                              ('rotating::Row', 'rows', '1')]:
        for qualifier in ('', 'volatile '):
            old = f'reinterpret_cast<__gm__ {qualifier}{typ}*>(ctx->{member})'
            text = text.replace(old, f'({old}+uint64_t(b)*{stride})')
    text = text.replace('asc_stcg(&row->requests,total);', '')
    text = text.replace('reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[q*32]',
                        'reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[query*32]')
    text = replace_once(text,
        'if(mode==5 && q==3){uint64_t start=GetSystemCycle();while(GetSystemCycle()-start<36000){} }op.TerminalQuery(q);',
        'uint32_t topkError=op.TerminalQueryRuntime(query);\n'
        ' reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[query*32]=topkError;')
    text = text.replace('(ctx->locations)+q*2048', '(ctx->locations)+query*2048')
    text = text.replace('(ctx->selected)+q*2048', '(ctx->selected)+query*2048')
    text = text.replace('(rotating::TopKHeader(ctx))[q*32+6]', '(rotating::TopKHeader(ctx))[query*32+6]')
    text = text.replace('(rotating::TopKHeader(ctx))+q*32', '(rotating::TopKHeader(ctx))+query*32')
    text = text.replace('(ctx->progress)+q*32', '(ctx->progress)+query*32')
    text = replace_once(text, 'uint32_t tailBegin=tailBounds?tailBounds[0]:0,tailEnd=tailBounds?tailBounds[1]:0;',
                        'uint32_t tailBegin=tailBounds?tailBounds[b*2]:0,tailEnd=tailBounds?tailBounds[b*2+1]:0;')
    text = replace_once(text,
        'CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->rows),128);',
        'CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->rows)+uint64_t(b)*sizeof(rotating::Row),128);')
    text = text.replace('reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+q*32',
                        'reinterpret_cast<__gm__ uint32_t*>(runtime->templates)+query*32')
    text = text.replace('reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(ctx->args)+q', 'args')
    text = text.replace(' auto args=args;\n', '')
    text = text.replace('reinterpret_cast<__gm__ uint64_t*>(a->logical)',
                        '(reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*s.history)')
    text = text.replace('reinterpret_cast<__gm__ uint64_t*>(a->keys)',
                        '(reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*s.history*16)')
    text = text.replace('reinterpret_cast<__gm__ uint32_t*>(a->scales)',
                        '(reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*s.history)')
    text = text.replace('reinterpret_cast<__gm__ uint64_t*>(a->addresses)',
                        '(reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*addressStride)')
    # Keep the original temporary declaration; routed publication below writes
    # actual public outputs and avoids a duplicate private logical DMA.
    text = text.replace('reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+q*2048',
                        'reinterpret_cast<__gm__ uint32_t*>(runtime->logical)+query*2048')
    text = text.replace('priorJournalCount);', 'priorJournalCount,b);')
    text = text.replace('reinterpret_cast<__gm__ uint64_t*>((reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+uint64_t(b)*1)->addresses)',
                        'reinterpret_cast<__gm__ uint64_t*>(rotating::RequestAddressTable(ctx,b,owner,s.history))')
    begin = text.index(' queue->sq_producer_index=originalHead+own;')
    end = text.index(' if(profile)ts[4]=GetSystemCycle();', begin)
    text = text[:begin] + ' rotating::CommitTerminalPost(args,originalHead,own,post);\n' + text[end:]
    text = replace_once(text, ' while(GetSystemCycle()-postedAt<kSimCqDelayTicks){}',
                        ' uint64_t cqError=rotating::CompleteTerminalPost(args,post);\n'
                        ' if(q==1)rotating::CompleteTerminalPrefetch(ctx,b);\n'
                        ' reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[query*32]|=uint32_t(cqError);')
    old = ' flags[q*32+1]=generation;\n kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+q*32),128);'
    new = ''' if(q==1)cqError|=(reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b)->error;
 uint32_t ownError=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[query*32];
 flags[(28+q)*32]=uint32_t(cqError)|ownError;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+(28+q)*32),4);
 Simt::VF_CALL<FinishTerminalRow>(Simt::Dim3{512,1,1},a,ctx,runtime,b,q,owner,total,first,n,flags,generation,work);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);'''
    text = replace_once(text, old, new)
    text = replace_once(text,
        'uint32_t first=total*q/4,last=total*(q+1)/4,n=last-first;',
        'uint32_t first=total*q/4,last=total*(q+1)/4,n=total<=rotating::kLruSlots?last-first:0;')
    text = replace_once(text, 'uint32_t indexError=n?local.GetValue(TAIL_CONTROL):0;',
                        'uint32_t indexError=total>rotating::kLruSlots?1U:(n?local.GetValue(TAIL_CONTROL):0);')
    # TopK returns the first error value before any consumer reads it.
    text = replace_once(text, ' reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[query*32]=0;\n', '')
    text = replace_once(text, ' uint32_t initialCount=(reinterpret_cast<__gm__ volatile rotating::Row*>(ctx->rows)+uint64_t(b)*1)->requests;\n', '')
    text = replace_once(text, 'uint32_t first,uint32_t initialCount,__gm__ uint32_t* header', 'uint32_t first,__gm__ uint32_t* header')
    text = replace_once(text, 'work,ctx,total,q,first,initialCount,reinterpret_cast', 'work,ctx,total,q,first,reinterpret_cast')
    path.write_text(text)
    # Native args address the whole acquired output. Retain local Index slots,
    # but add the live owner stride to WQE destinations only.
    post_path = destination / 'parallel_post.h'
    post_text = post_path.read_text()
    post_text = post_text.replace('__gm__ uint32_t* selected){',
                                  '__gm__ uint32_t* selected,uint32_t outputBase,__gm__ uint64_t* update,uint32_t batch,uint32_t query){')
    post_text = post_text.replace('slots[i]=physical;lo[i]', 'slots[i]=outputBase+physical;lo[i]')
    post_text = replace_once(post_text,
        '  if(!lane){counter[0]=count;counter[1]=slotBase;}',
        '  if(slotBase>rotating::kLruSlots || count>rotating::kLruSlots-slotBase){\n'
        '   if(!lane)asc_atomic_or(error,8U);count=0;\n'
        '  }\n'
        '  if(!lane){counter[0]=count;counter[1]=slotBase;}')
    post_text = replace_once(post_text, '   uint32_t f=tid-96;auto cfg=lo+6144;', '''   uint32_t f=tid-96;auto cfg=lo+6144;
   auto unpack=reinterpret_cast<__gm__ int64_t*>(asc_ldcg(update+20));
   auto valid=reinterpret_cast<__gm__ int64_t*>(asc_ldcg(update+21));
   for(uint32_t w=0;w<2;++w){
    uint32_t row=w*32+f;
    bool active=row<batch*4 && asc_ldcg(valid+row)!=0;
    uint64_t source=active?uint64_t(asc_ldcg(unpack+row)):UINT64_MAX;
    uint32_t mask=asc_ballot(active && source==query);
    uint32_t pad=asc_ballot(row<batch*4 && !active && query%4==0 && row/4==query/4);
    if(!f){counter[9+w]=mask;counter[11+w]=pad;}
   }''')
    post_path.write_text(post_text)
    text = path.read_text().replace('(ctx->selected)+query*2048);',
                                  '(ctx->selected)+query*2048,owner*uint32_t(ctx->outputStride),update,s.batch,query);')
    text = replace_once(text,
        ' DataCopy(slots,postScratch,2048);DataCopy(output,ids,2048);\n'
        ' SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);',
        ' DataCopy(slots,postScratch,2048);\n'
        ' PublishTerminalOutputs(update,s.batch,query,ids,postScratch,postScratch[bitmapWords+9],ctx->locations+uint64_t(query)*2048*4);')
    # AIV-only task metadata does not exclude the body from the AIC compiler.
    # Match the production vector entry's compile-time core guard.
    text = replace_once(text,
        ' KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);',
        ' KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);\n if ASCEND_IS_AIV {')
    text = replace_once(text, '\n#endif\n}\n', '\n } // ASCEND_IS_AIV\n#endif\n}\n')
    path.write_text(text)
    cmake = destination / 'CMakeLists.txt'
    text = cmake.read_text().replace('-DV9_SIMULATION=1', '-DV9_SIMULATION=0')
    # Build only the native object, never link/execute the model's fake driver.
    text = text[:text.index('add_executable(real_topk_sim')]
    text = text.replace('set(RUN_MODE sim)', 'set(RUN_MODE npu)\nset(CMAKE_POSITION_INDEPENDENT_CODE ON)')
    text = text.replace('set(SOC_VERSION Ascend950PR_9599)', 'set(SOC_VERSION Ascend950PR_9599 CACHE STRING \"Target SoC\")')
    text = text.replace('set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann)', 'set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann CACHE PATH \"CANN root\")')
    text += '''\nadd_library(parallel_terminal SHARED host.cpp)
target_include_directories(parallel_terminal PRIVATE ${OP_ROOT}/include)
target_link_libraries(parallel_terminal PRIVATE real_topk_kernel host_intf_pub ascendcl)
'''
    cmake.write_text(text)
    adapt_pack_layout(destination)
    adapt_index_split(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    generate(parser.parse_args().output.resolve())
