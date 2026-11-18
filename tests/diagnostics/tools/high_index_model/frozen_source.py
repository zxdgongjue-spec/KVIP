# SPDX-License-Identifier: Apache-2.0
"""Build a diagnostic entry from the pinned production source, with the production queue ABI."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from prepare import split_once


def isolate_frozen(kernel):
    path = kernel / "high_post_index.h"
    src = path.read_text()
    prefix, rest = split_once(src, "    uint32_t useful=0;")
    _, rest = split_once(rest, "    if (!lane) CompletePrefetch(ctx,b,true);")
    # In the original run the per-row planner completed and cached these fields.
    # Load them before the same production row admission logic reuses workerStatus.
    capture = """
    const bool getOnly=(a->prefetchCount & 0x00800000U)!=0;
    if(lane==2){
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ws),128);
        plan.error=ws[0]; plan.chunks=ws[2]; plan.pairs=ws[3];
        plan.singles=ws[4]; plan.invalidations=ws[5];
    }
"""
    before_wait, after_wait = split_once(
        rest, "    if (waitFirst) {\n        error|=post_index_transport::Wait(transport);"
    )
    get = """
    if(getOnly){
        error|=post_index_transport::Wait(transport);
        if(profile)ticks[7]=r->cqDone=GetSystemCycle();
        r->error=error; r->completed=transport.completed; r->active=2;
        post_index_tail::Flush(r);
        SyncAll();
        if(profile)ticks[13]=GetSystemCycle();
        if(profile)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks),256);
        return;
    }
"""
    before_publish, _ = split_once(after_wait, "    if (profile) ticks[13]=GetSystemCycle();")
    path.write_text(
        prefix
        + capture
        + before_wait
        + get
        + "    if (waitFirst) {\n        error|=post_index_transport::Wait(transport);"
        + before_publish
        + "    if (profile) {ticks[13]=GetSystemCycle();"
        " kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks),256);}\n}\n} }\n"
    )
    streaming = kernel / "streaming.cpp"
    prefix, entry = split_once(
        streaming.read_text(), 'extern "C" __global__ __aicore__ void v9_streaming_resolve_update(GM_ADDR arg) {'
    )
    entry = entry[: entry.index(" if ASCEND_IS_AIV {")]
    streaming.write_text(
        prefix
        + 'extern "C" __global__ __aicore__ void kvip_frozen_high_tail(GM_ADDR arg) {'
        + entry
        + " if ASCEND_IS_AIV {high_post_index::Run(reinterpret_cast<__gm__ Args*>(arg),GetBlockIdx());}\n#endif\n}\n"
    )
    hook = kernel.parent / "cmake/set_tail_schedule.cmake"
    text = hook.read_text()
    if "v9_streaming_resolve_update" not in text:
        raise ValueError("Unexpected target in schedule hook")
    hook.write_text(text.replace("v9_streaming_resolve_update", "kvip_frozen_high_tail"))


def main(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    here = Path(__file__).resolve().parent
    rev = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryFile() as tar:
        subprocess.run(["git", "-C", repo, "archive", rev, "csrc"], stdout=tar, check=True)
        tar.seek(0)
        subprocess.run(["tar", "-x", "-C", output, "-f", "-"], stdin=tar, check=True)
    isolate_frozen(output / "csrc/operators/lightning_indexer_with_fetch/op_kernel")
    cmake = (here / "CMakeLists.txt").read_text().split("add_executable(high_index_case main.cpp)")[0]
    (output / "CMakeLists.txt").write_text(
        cmake
        + """
add_library(kvip_frozen_tail SHARED frozen_bridge.cpp)
target_link_libraries(kvip_frozen_tail PRIVATE tail_kernel)
target_link_options(kvip_frozen_tail PRIVATE "-Wl,-Bsymbolic" "-Wl,-rpath-link,${ASCEND_CANN_PACKAGE_PATH}/lib64")
"""
    )
    shutil.copyfile(here / "frozen_bridge.cpp", output / "frozen_bridge.cpp")
    (output / "revision.json").write_text(json.dumps(dict(revision=rev)))


if __name__ == "__main__":
    main(*sys.argv[1:])
