# SPDX-License-Identifier: Apache-2.0
"""Build an opt-in reproduction from a clean KVIP worktree, preserving its Index implementation."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def split_once(text, anchor):
    if text.count(anchor) != 1:
        raise ValueError(f"unsupported source layout (expected one {anchor!r})")
    return text.split(anchor)


def isolate(kernel):
    header = kernel / "high_post_index.h"
    text = header.read_text()
    start = "    if (profile) ticks[4]=GetSystemCycle();"
    pre, body = split_once(text, start)
    before_cq, _ = split_once(body, "    if (!waitFirst) {")
    before_wait, after_wait = split_once(before_cq, "    // Wait-first")
    after_publish = after_wait[after_wait.index("    SyncAll();"):]
    prefix, _ = split_once(pre, "    auto info=")
    # All blocks begin together. Transport records are preinitialised to success.
    # Keep RowErrors and all the original production Index joins inside the timed region.
    prefix += """
    auto update=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
    uint32_t total=0;
    SyncAll();
"""
    # Copy the private planning body directly from the candidate. Its wait on
    # already-published flags is measured. A common join prevents hiding work
    # behind any replay preparation in this isolated comparison.
    if '#include "high_index_balance.h"' in pre:
        pointers = pre[pre.index("    auto edges="):pre.index("    r->error=0;")]
        early, _ = split_once(pre.split("    else if (lane==2) {")[1],
                              "    if (!lane) CompletePrefetch(ctx,b,true);")
        prefix += pointers + "    if (lane==2) {" + early + "    SyncAll();\n"
    else:
        prefix += "    auto moves=reinterpret_cast<__gm__ Move*>(a->highIndex)+uint64_t(b)*kCapacity;\n"
    header.write_text(prefix + start + before_wait + after_publish +
                      "    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks),256);\n}\n} }\n")
    streaming = kernel / "streaming.cpp"
    prefix, entry = split_once(streaming.read_text(),
                               'extern "C" __global__ __aicore__ void v9_streaming_resolve_update(GM_ADDR arg) {')
    entry = entry[:entry.index(" if ASCEND_IS_AIV {")]
    streaming.write_text(prefix + 'extern "C" __global__ __aicore__ void v9_streaming_resolve_update(GM_ADDR arg) {' +
                         entry + " if ASCEND_IS_AIV {high_post_index::Run(reinterpret_cast<__gm__ Args*>(arg),GetBlockIdx());}\n#endif\n}\n")


def main():
    repo, dest = (Path(x).resolve() for x in sys.argv[1:3])
    mode, soc, include = sys.argv[3:]
    here = Path(__file__).resolve().parent
    op = "csrc/operators/lightning_indexer_with_fetch"
    sha = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    files = ("main.cpp", "CMakeLists.txt", "prepare.py")
    code = hashlib.sha256(b"".join((here / f).read_bytes() for f in files)).hexdigest()
    provenance = dict(revision=sha, mode=mode, soc=soc, include=include, harness_sha256=code)
    marker = dest / "model-source.json"
    if marker.exists():
        if json.loads(marker.read_text()) != provenance:
            raise SystemExit(f"Case directory contains a different build: {dest}")
        return
    if (dest / "csrc").exists() or (dest / "build").exists():
        raise SystemExit(f"Unrecognised build directory: {dest}")
    with tempfile.TemporaryFile() as archive:
        subprocess.run(["git", "-C", repo, "archive", sha, "csrc"], stdout=archive, check=True)
        archive.seek(0)
        subprocess.run(["tar", "-xf", "-", "-C", dest], stdin=archive, check=True)
    isolate(dest / op / "op_kernel")
    for filename in files[:2]:
        shutil.copyfile(here / filename, dest / filename)
    marker.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
