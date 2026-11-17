"""Run the production Context snapshot copy including the appended URMA threshold pointer."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("backend,size,tail", [("urma", 328, 320), ("incremental", 336, 320)])
@pytest.mark.parametrize("caller", ["streaming.cpp", "output_worker.h"])
def test_exact_context_dma(tmp_path, backend, size, tail, caller):
    root = Path(__file__).resolve().parents[4]
    operators = root / "csrc/operators"
    kernel = operators / "lightning_indexer_with_fetch/op_kernel"
    context = operators / (
        "lightning_indexer_with_fetch_ubmem" if backend == "ubmem" else "lightning_indexer_with_fetch"
    ) / "include/context.h"
    # Execute the actual call at each site, not a separate model of its length.
    source = (kernel / caller).read_text()
    begin = source.index("CopyContextSnapshot(ctl[contextOffset],ctx);")
    call = source[begin:source.index(";", begin) + 1]
    (tmp_path / "batch_get_abi.h").write_text("#pragma once\n")
    program = r'''
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>
#define KVIP_INCREMENTAL_TOPK OPTION_VALUE
#define __gm__
#define __aicore__
#define V9_UBPIPE_DEPTH 8
#define V9_UBPIPE_ROWS 4
#define TOKEN_BYTES 656
#include "CONTEXT"
using rotating::Context;
template<class T> struct LocalTensor {
 T* data;
 LocalTensor operator[](size_t i) const {return {data+i};}
 template<class U> LocalTensor<U> ReinterpretCast() const {return {reinterpret_cast<U*>(data)};}
};
template<class T> struct GlobalTensor {
 T* data;
 void SetGlobalBuffer(T* p){data=p;}
};
struct DataCopyExtParams {uint16_t blockCount;uint32_t blockLen,srcStride,dstStride,rsv;};
template<class T> struct DataCopyPadExtParams {bool isPad;uint8_t left,right;T value;};
std::vector<std::function<void()>> pending;
unsigned copied=0;
void DataCopyPad(LocalTensor<uint8_t> dst, GlobalTensor<uint8_t> src,
                 DataCopyExtParams p, DataCopyPadExtParams<uint8_t> pad){
 assert(p.blockCount==1 && p.blockLen==sizeof(Context));
 assert(p.srcStride==0 && p.dstStride==0 && !pad.isPad);
 assert(reinterpret_cast<uintptr_t>(dst.data)%32==0);
 copied=p.blockLen;
 pending.push_back([=]{std::memcpy(dst.data,src.data,p.blockLen);});
}
#include "SNAPSHOT"
int main(){
 static_assert(sizeof(Context)==SIZE);
 static_assert(offsetof(Context,thresholdBounds)==TAIL);
 Context original{};
 auto bytes=reinterpret_cast<uint8_t*>(&original);
 for(unsigned i=0;i<sizeof(original);++i)bytes[i]=uint8_t(i*13+7);
 original.thresholdBounds=0x123456789abcdef0ULL;
 alignas(32) uint32_t memory[640];
 std::memset(memory,0xa5,sizeof(memory));
 LocalTensor<uint32_t> ctl{memory};
 constexpr unsigned contextOffset=128;
 auto ctx=&original;
 CALL
 // MTE2 is asynchronous; the real callers join MTE2_V before consumption.
 assert(reinterpret_cast<uint8_t*>(memory)[contextOffset*4]==0xa5);
 for(auto& copy:pending)copy();
 assert(copied==sizeof(Context));
 assert(std::memcmp(memory+contextOffset,&original,sizeof(Context))==0);
 Context restored{};
 std::memcpy(&restored,memory+contextOffset,sizeof(restored));
 assert(restored.thresholdBounds==original.thresholdBounds);
 auto out=reinterpret_cast<uint8_t*>(memory);
 for(unsigned i=0;i<contextOffset*4;++i)assert(out[i]==0xa5);
 for(unsigned i=contextOffset*4+sizeof(Context);i<sizeof(memory);++i)assert(out[i]==0xa5);
}
'''
    for name, value in {
        "OPTION_VALUE": str(int(backend == "incremental")),
        "CONTEXT": str(context),
        "SNAPSHOT": str(kernel / "context_snapshot.h"),
        "SIZE": str(size),
        "TAIL": str(tail),
        "CALL": call,
    }.items():
        program = program.replace(name, value)
    cpp = tmp_path / "snapshot.cpp"
    cpp.write_text(program)
    executable = tmp_path / "snapshot"
    subprocess.run(
        [shutil.which("clang++") or "c++", "-std=c++17", "-fsanitize=address,undefined",
         "-fno-omit-frame-pointer", "-I", str(tmp_path), str(cpp), "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True, timeout=15)
