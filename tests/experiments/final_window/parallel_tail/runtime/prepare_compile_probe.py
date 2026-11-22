"""Generate a compile-only probe of the real post/CQ adapter with v537.

This is deliberately not an end-to-end runner. It has no real lease and must
not be used to report transport performance or acceptance.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from private_topk import adapt


PROBE = r'''
#if defined(__CCE_AICORE__)
using namespace rotating;
#include "staged_fetch_worker.h"
namespace rotating {
#include "terminal_transport.h"
}
extern "C" __global__ __aicore__ void terminal_transport_compile_probe(
    GM_ADDR arg,GM_ADDR state,GM_ADDR context) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 auto args=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(arg);
 auto post=reinterpret_cast<__gm__ rotating::FinalPost*>(state);
 rotating::CommitTerminalPost(args,0,1,post);
 rotating::CompleteTerminalPost(args,post);
 rotating::CompleteTerminalPrefetch(reinterpret_cast<__gm__ rotating::Context*>(context),0);
}
#endif
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    destination = args.output.resolve()
    subprocess.run([sys.executable, str(here.parent / 'prepare_candidate.py'),
                    'v537', str(destination)], check=True)
    shutil.copy2(here / 'transport.h', destination / 'terminal_transport.h')
    shutil.copy2(here / 'abi.h', destination / 'terminal_runtime_abi.h')
    adapt(destination / 'csrc/operators/lightning_indexer_with_fetch/op_kernel')
    kernel = destination / 'kernel.cpp'
    text = kernel.read_text()
    if text.count('op.TerminalQuery(q);') != 1:
        raise ValueError('Expected one terminal query call')
    text = text.replace('op.TerminalQuery(q);', 'op.TerminalQueryRuntime(q);')
    kernel.write_text('#include "terminal_runtime_abi.h"\n' + text + PROBE)
    cmake = destination / 'CMakeLists.txt'
    text = cmake.read_text()
    if text.count('-DV9_SIMULATION=1') != 1:
        raise ValueError('Unexpected simulation build definition')
    cmake.write_text(text.replace('-DV9_SIMULATION=1', '-DV9_SIMULATION=0'))
    print('Compile-only real transport probe prepared; do not execute the fake driver.')


if __name__ == '__main__':
    main()
