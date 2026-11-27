# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import shutil
import sys

native, build = map(Path, sys.argv[1:3])
natural = len(sys.argv) > 3 and sys.argv[3] == "ON"
def prepare(name, changes):
    code = (native / name).read_text()
    for old, new in changes:
        if code.count(old) != 1:
            raise ValueError(f"Unexpected original {name} at {old!r}")
        code = code.replace(old, new)
    (build / name).write_text(code)

prepare("terminal_schedule.h", [
    ("rotating::Context* ctx, uint32_t batch) {", "rotating::Context* ctx, uint32_t batch, __gm__ uint32_t* permits) {"),
    ("if (row < batch) asc_stcg(schedule->order + rank, row);",
     """if (row < batch) {
        asc_stcg(schedule->order + rank, row);
        if (permits) {
            auto rows = reinterpret_cast<__gm__ rotating::Row*>(ctx->rows);
            asc_stcg(permits + (row * 2) * 32, 0U);
            asc_stcg(permits + (row * 2) * 32 + 1, asc_ldcg(&rows[row].slots));
            asc_stcg(permits + (row * 2 + 1) * 32, 0U);
        }
    }"""),
    ("schedule, ctx, batch);", "schedule, ctx, batch, reinterpret_cast<__gm__ uint32_t*>(runtime->reserved[6]));"),
])
prepare("terminal_transport.h", [
    ("struct TerminalBatches {", '#include "first_receipt.h"\nstruct TerminalBatches {'),
    ("const TerminalBatches& batches, uint64_t head) {",
     "const TerminalBatches& batches, uint64_t head, __gm__ uint32_t* pilot, __gm__ uint32_t* waitFor, __gm__ uint64_t* firstClock) {"),
    ("auto result = DrainQueue(entity, queue, post->pending);",
     """uint64_t peerError = 0;
    if (waitFor) {
        peerError = WaitFirstReceipt(waitFor);
        if (!peerError) {
            CommitTerminalPost(args, head, batches.ends[0], post);
            if (firstClock) firstClock[0] = GetSystemCycle();
        }
    }
    if (firstClock && post->pending) firstClock[1] = GetSystemCycle();
    auto result = peerError ? CqDrainResult{peerError, 0, 0} : DrainQueue(entity, queue, post->pending);
    if (firstClock && post->pending && !peerError) firstClock[2] = GetSystemCycle();
    if (pilot) PublishFirstReceipt(pilot, result.error == 0);"""),
])
row_changes = []
if not natural:
    prepare("parallel_post.h", [
        ("template<bool LargeBatch>", '#include "ordered.h"\n\ntemplate<bool LargeBatch>'),
        ("uint32_t batch,uint32_t query){", "uint32_t batch,uint32_t query,__gm__ uint32_t* plan){"),
        ("uint32_t slotBase=0;if(!lane && count)slotBase=asc_atomic_add(&row->slots,count);slotBase=asc_shfl(slotBase,0);",
         "const bool sameSet=MatchFinalOrder(plan,ids,count,lane,hi);\n  const uint32_t slotBase=ReserveCapturedOrder(plan,&row->slots,count,flags,generation,q,lane,sameSet);"),
    ])
    row_changes.append(("update,s.batch,query);", "update,s.batch,query,reinterpret_cast<__gm__ uint32_t*>(runtime->reserved[7])+query*kPrivateFinalWords);"))
row_changes += [
    ("rotating::CommitTerminalPost(args,originalHead,batches.ends[0],post);",
     """auto permits = reinterpret_cast<__gm__ uint32_t*>(runtime->reserved[6]);
 const bool enabled = LargeBatch && runtime->reserved[8] && permits;
 const uint32_t prefetchSlots = enabled ? rotating::LoadPermit(permits + b * 64 + 1) : total;
 const bool longRow = enabled && total >= prefetchSlots && total - prefetchSlots >= 256;
 const bool isPilot = ((q ^ b) & 1U) == 0;
 auto permit = longRow ? permits + (b * 2 + q / 2) * 32 : nullptr;
 // Both members of a pair always clear the ordinary Index rendezvous before waiting.
 // A pilot with no final GET can release its partner as soon as both lists are known.
 if (isPilot && permit && !own) rotating::PublishFirstReceipt(permit, true);
 auto waitFor = enabled && longRow && !isPilot && own ? permit : nullptr;
 rotating::CommitTerminalPost(args, originalHead, waitFor ? 0 : batches.ends[0], post);
 if (profile && own && !waitFor) ts[27] = GetSystemCycle();"""),
    ("rotating::CompleteTerminalPost(args,post,batches,originalHead);",
     "rotating::CompleteTerminalPost(args,post,batches,originalHead, isPilot && own ? permit : nullptr, waitFor, profile ? ts + 27 : nullptr);"),
]
prepare("row_kernel.h", row_changes)
for name in ("ordered.h", "first_receipt.h", "kernel.cpp"):
    shutil.copyfile(Path(__file__).with_name(name), build / name)
