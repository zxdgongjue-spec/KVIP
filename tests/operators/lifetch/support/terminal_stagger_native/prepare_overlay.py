# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import shutil
import sys

native, build = map(Path, sys.argv[1:])
def prepare(name, changes):
    code = (native / name).read_text()
    for old, new in changes:
        if code.count(old) != 1:
            raise ValueError(f"Unexpected original {name} at {old!r}")
        code = code.replace(old, new)
    (build / name).write_text(code)

prepare("terminal_transport.h", [
    ("struct TerminalBatches {", '#include "stagger.h"\n\nstruct TerminalBatches {'),
    ("__gm__ FinalPost* post) {", "__gm__ FinalPost* post, uint32_t group) {"),
    ("PublishTerminalPrefix(args, entity, head + first);", "WaitDoorbellQuarter(group);\n        PublishTerminalPrefix(args, entity, head + first);"),
    ("const TerminalBatches& batches, uint64_t head) {", "const TerminalBatches& batches, uint64_t head, uint32_t group) {"),
    ("PublishTerminalPrefix(args, entity, head + end);", "WaitDoorbellQuarter(group);\n        PublishTerminalPrefix(args, entity, head + end);")
])
prepare("parallel_post.h", [
    ("template<bool LargeBatch>", '#include "ordered.h"\n\ntemplate<bool LargeBatch>'),
    ("uint32_t batch,uint32_t query){", "uint32_t batch,uint32_t query,__gm__ uint32_t* plan){"),
    ("uint32_t slotBase=0;if(!lane && count)slotBase=asc_atomic_add(&row->slots,count);slotBase=asc_shfl(slotBase,0);",
     "const bool sameSet=MatchFinalOrder(plan,ids,count,lane,hi);\n  const uint32_t slotBase=ReserveCapturedOrder(plan,&row->slots,count,flags,generation,q,lane,sameSet);"),
])
prepare("row_kernel.h", [
    ("update,s.batch,query);", "update,s.batch,query,reinterpret_cast<__gm__ uint32_t*>(runtime->reserved[7])+query*kPrivateFinalWords);"),
    ("rotating::CommitTerminalPost(args,originalHead,batches.ends[0],post);",
     "const uint32_t group = batches.count == 4 && (runtime->reserved[8] != 0) ? (b + q) & 3U : rotating::kGateDisabled;\n rotating::CommitTerminalPost(args,originalHead,batches.ends[0],post,group);"),
    ("rotating::CompleteTerminalPost(args,post,batches,originalHead);",
     "rotating::CompleteTerminalPost(args,post,batches,originalHead,group);")
])
for name in ("stagger.h", "ordered.h", "kernel.cpp"):
    shutil.copyfile(Path(__file__).with_name(name), build / name)
