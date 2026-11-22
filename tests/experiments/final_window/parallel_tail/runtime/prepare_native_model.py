"""Exercise the generated native tail with a software CQ, not a real NIC.

Only transport.h is replaced. Output routing, Index changes, retirement,
per-request ABI and kernel dispatch come from the native generator.
"""
import argparse
from pathlib import Path
import shutil

from prepare_native import generate
from private_topk import replace_once


HERE = Path(__file__).resolve().parent


def prepare_model(destination):
    generate(destination)
    shutil.copy2(HERE / 'model_transport.h', destination / 'terminal_transport.h')
    cmake = destination / 'CMakeLists.txt'
    text = cmake.read_text().replace('set(RUN_MODE npu)', 'set(RUN_MODE sim)').replace('-DV9_SIMULATION=0', '-DV9_SIMULATION=1')
    text += '''
add_executable(native_tail_sim driver.cpp)
target_include_directories(native_tail_sim PRIVATE ${OP_ROOT}/include ${OP_ROOT}/op_kernel ${KVAF_INCLUDE})
target_link_libraries(native_tail_sim PRIVATE real_topk_kernel host_intf_pub ascendcl)
target_link_options(native_tail_sim PRIVATE -Wl,--allow-shlib-undefined)
target_compile_definitions(native_tail_sim PRIVATE KVIP_OUTPUT_WORKER=1)
'''
    cmake.write_text(text)
    driver = destination / 'driver.cpp'
    text = driver.read_text()
    text = '#include "terminal_runtime_abi.h"\n' + text
    text = text.replace('aclrtlaunch_test_service(uint32_t,void*,void*,uint32_t)',
                        'aclrtlaunch_v9_parallel_terminal(uint32_t,void*,void*)')
    insertion = '''
 c.tokenAddresses=row.addresses;c.addressStride=H;
 c.shape.finalWorkers=4;c.shape.finalPostOffset=0;
 c.owners=put(std::vector<rotating::FinalPost>(4));
 c.finalStatus=put(std::vector<KvafBatchGetCoreStatus>(2));
 c.changeCount=put(std::vector<uint32_t>(1));
 auto outputSlots=put(std::vector<uint32_t>(8192,0xdeadbeef));
 auto updateStatus=put(std::vector<uint32_t>(1,0xdeadbeef));
 std::vector<uint64_t> update(28);update[6]=updateStatus;update[7]=H;
 // Exercise non-identity output routing: Q4,Q2,Q1,Q3.
 update[20]=put(std::vector<int64_t>{3,1,0,2});update[21]=put(std::vector<int64_t>(4,1));
 update[22]=a.eventsSrc;update[23]=outputSlots;
 a.updateBindings=put(update);a.completedRows=put(std::vector<rotating::Row>(1));
 a.completedEvents=put(std::vector<rotating::Event>(5));
 a.context=put(std::vector<rotating::Context>{c});
'''
    text = replace_once(text, ' auto deviceArg=put(std::vector<pipeline_ab::Args>{a});',
                        insertion + ''' auto deviceArg=put(std::vector<pipeline_ab::Args>{a});
 terminal_runtime::Binding binding{};binding.pipeline=deviceArg;binding.queueArgs=c.args;
 binding.templates=a.eventsDst;binding.flags=a.flags;binding.times=a.times;binding.logical=a.eventsSrc;
 binding.version=1;binding.options=mode==4?1:0;
 auto terminal=put(std::vector<terminal_runtime::Binding>{binding});''')
    text = text.replace('auto rows=get<rotating::Row>(c.rows,1);',
                        'auto rows=get<rotating::Row>(a.completedRows,1);')
    text = text.replace('bad+=rows[0].requests!=230 || totalPosted!=146;',
                        'bad+=rows[0].requests!=84 || rows[0].slots!=230 || rows[0].finalRequests!=146 || totalPosted!=146;\n'
                        ' for(auto x:get<uint32_t>(c.rows,sizeof(rotating::Row)/4))bad+=x!=0;\n'
                        ' bad+=get<uint32_t>(updateStatus,1)[0]!=0;')
    # Preserve the existing per-Q output oracle, while reading routed public
    # rows and verifying BOTH the public and internal slot outputs.
    text = text.replace('auto logicalOut=get<uint32_t>(a.eventsSrc,8192);', '''auto routed=get<uint32_t>(a.eventsSrc,8192);
 auto routedSlots=get<uint32_t>(outputSlots,8192);
 std::vector<uint32_t> logicalOut(8192);const uint32_t routes[4]={3,1,0,2};
 for(uint32_t outputRow=0;outputRow<4;++outputRow)for(uint32_t i=0;i<K;++i){
  uint32_t query=routes[outputRow];logicalOut[query*K+i]=routed[outputRow*K+i];
  bad+=routedSlots[outputRow*K+i]!=slots[query*K+i];
 }
''')
    text = text.replace('mode,rows[0].requests-84,bad', 'mode,rows[0].finalRequests,bad')
    text = text.replace('aclrtlaunch_test_service(4,stream,(void*)deviceArg,mode)',
                        'aclrtlaunch_v9_parallel_terminal(4,stream,(void*)terminal)')
    text = replace_once(text, ' auto generations=get<uint32_t>(a.flags,16*32);',
        ' auto savedEvents=get<rotating::Event>(a.completedEvents,4);\n'
        ' auto liveEvents=get<rotating::Event>(c.events,4);\n'
        ' for(uint32_t i=0;i<4;++i){\n'
        '  if(memcmp(&savedEvents[i],&events[i],sizeof(rotating::Event)))throw std::runtime_error("event snapshot mismatch");\n'
        '  rotating::Event empty{};if(memcmp(&liveEvents[i],&empty,sizeof(empty)))throw std::runtime_error("live event not retired");\n'
        ' }\n'
        ' auto generations=get<uint32_t>(a.flags,16*32);')
    # The second simulated prefix must republish its events, as real LI does.
    text = replace_once(text, '   restore(c.mapping,initialMapping);',
                        '   restore(c.events,events);\n   restore(c.mapping,initialMapping);')
    driver.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    prepare_model(parser.parse_args().output.resolve())
