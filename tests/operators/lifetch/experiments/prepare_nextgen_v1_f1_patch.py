#!/usr/bin/env python3
"""Generate an isolated candidate patch from exact source blobs."""
import difflib,hashlib,subprocess,sys
from pathlib import Path
FROZEN="79c6a22557ed63a53da687e58f1f5bc1401fd8ed"
EDITS=[
  {
    "path": "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer.cpp",
    "sha": "6c8c85818c1fd78d683889c2f8fdbc8b93bdc028",
    "edits": [
      {
        "line": 47,
        "old": [],
        "next": [
          "#if !V9_UBMEM",
          "    // Experimental F1: only the proved Q4, two-Q-group/request geometry.",
          "    bool closure = !splitSequence && shape.splitQ && shape.indexHeads == 32 &&",
          "        shape.liCores == 2 * shape.batch && shape.batch * 4 <= 2 * shape.aicCount &&",
          "        shape.rounds && shape.serviceAics && P(5) && P(6);",
          "    if (closure) {",
          "        auto q = reinterpret_cast<__gm__ const uint32_t*>(P(5));",
          "        auto k = reinterpret_cast<__gm__ const uint32_t*>(P(6));",
          "        for (uint32_t b = 0; b < shape.batch; ++b) {",
          "            if (q[b] != 4 || k[b] < shape.resident || k[b] > shape.history || k[b] % 128)",
          "                closure = false;",
          "        }",
          "    }",
          "    op.genericEpochClosure_ = closure;",
          "#endif"
        ]
      },
      {
        "line": 56,
        "old": [],
        "next": [
          "#if !V9_UBMEM",
          "        op.vectorService.SetActualSequenceSplit(splitSequence);",
          "#endif"
        ]
      }
    ]
  },
  {
    "path": "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer.h",
    "sha": "c497d1c2111bf94917df4eb3eb450e37ccf33900",
    "edits": [
      {
        "line": 90,
        "old": [],
        "next": [
          "#if !V9_UBMEM",
          "    bool genericEpochClosure_ = false;",
          "#endif"
        ]
      },
      {
        "line": 927,
        "old": [],
        "next": [
          "#if !V9_UBMEM",
          "    const int32_t normalEpoch = rotating::EpochAt(runtimeShape_.batch,runtimeShape_.history,runtimeShape_.resident,runtimeShape_.splitQ,runtimeShape_.rounds,runtimeShape_.serviceAics,s2LoopIdx + 1, aiCoreIdx);",
          "    int32_t closureEpoch = -1;",
          "    // Before final TopK: after that call a short producer may wait for all",
          "    // epochs, including the one it is itself responsible for closing.",
          "    if (genericEpochClosure_ && !rotationServed_ && normalEpoch < 0 &&",
          "        runInfo.isAllLoopEnd && runInfo.isLastS2InnerLoop &&",
          "        constInfo.s1BaseSize == 2 && !constInfo.isLDOpen && !constInfo.attenMaskFlag &&",
          "        runInfo.actS1Size == 4 && aiCoreIdx == 2 * runInfo.bIdx + runInfo.gS1Idx &&",
          "        (s2LoopIdx + 1) * 128 == runInfo.actS2Size) {",
          "        uint32_t epoch = rotating::ServiceRank(runtimeShape_.batch,runtimeShape_.splitQ,aiCoreIdx) / runtimeShape_.serviceAics;",
          "        if (epoch < runtimeShape_.rounds &&",
          "            rotating::Trigger(runtimeShape_.history,runtimeShape_.resident,runtimeShape_.rounds,epoch) > s2LoopIdx + 1)",
          "            closureEpoch = int32_t(epoch);",
          "    }",
          "#endif"
        ]
      },
      {
        "line": 933,
        "old": [],
        "next": [
          "#if !V9_UBMEM",
          "        else if (closureEpoch >= 0) {",
          "            CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(8);",
          "            rotationServed_ = true;",
          "        }",
          "#endif"
        ]
      },
      {
        "line": 947,
        "old": [],
        "next": [
          "#if !V9_UBMEM",
          "        else if (closureEpoch >= 0) {",
          "            vectorService.CloseSkippedRotation(uint32_t(closureEpoch));",
          "            CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(8);",
          "            rotationServed_ = true;",
          "        }",
          "#endif"
        ]
      }
    ]
  },
  {
    "path": "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer_vector.h",
    "sha": "7cc20aeda020264d1edc301ee974c0f9602aff89",
    "edits": [
      {
        "line": 50,
        "old": [],
        "next": [
          "#if !V9_UBMEM",
          "    __aicore__ inline void SetActualSequenceSplit(bool value) { actualSequenceSplit_ = value; }",
          "    __aicore__ inline void CloseSkippedRotation(uint32_t epoch) {",
          "        auto ctx = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);",
          "        uint32_t lane = rotating::ServiceLane(runtimeShape_.batch,runtimeShape_.splitQ,runtimeShape_.serviceAics,blockId_/2,blockId_%2);",
          "        auto events = reinterpret_cast<__gm__ rotating::Event*>(ctx->events);",
          "        PipeBarrier<PIPE_ALL>();",
          "        if (epoch) {",
          "            auto prev = reinterpret_cast<__gm__ volatile rotating::Event*>(events) + (epoch-1)*runtimeShape_.serviceLanes+lane;",
          "            do {",
          "                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ rotating::Event*>(prev)),64);",
          "            } while (!prev->complete);",
          "        }",
          "        auto event = events + epoch*runtimeShape_.serviceLanes+lane;",
          "        // No row, mapping, queue, WQE, CQ or incremental-state mutation.",
          "        // Round preparation zeroes these records; publish only the lane closure.",
          "        event->core = blockId_/2;",
          "        event->epoch = epoch;",
          "        event->skipped = 1;",
          "        event->complete = 1;",
          "        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(event),64);",
          "    }",
          "#endif"
        ]
      },
      {
        "line": 203,
        "old": [],
        "next": [
          "#if !V9_UBMEM",
          "    bool actualSequenceSplit_ = false;",
          "#endif"
        ]
      }
    ]
  },
  {
    "path": "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/incremental_methods.h",
    "sha": "84280ee9feea8308ee1acd17f5e2400dd66cab23",
    "edits": [
      {
        "line": 682,
        "old": [
          "    uint32_t half=rotating::UseSequenceSplit(runtimeShape_)?rotating::SequenceHalfWindow(runtimeShape_):0;"
        ],
        "next": [
          "    // Shape eligibility is not the actual execution mode for variable K.",
          "    uint32_t half=actualSequenceSplit_?rotating::SequenceHalfWindow(runtimeShape_):0;"
        ]
      }
    ]
  }
]
def main():
    source,out=map(Path,sys.argv[1:])
    def git(*args):
        return subprocess.check_output(["git","-C",str(source),*args],text=True).strip()
    assert git("rev-parse","HEAD")==FROZEN
    subprocess.run(["git","-C",str(source),"diff","--quiet","HEAD"],check=True)
    patch=[]
    for item in EDITS:
        path=item["path"];original=(source/path).read_text()
        assert git("hash-object",path)==item["sha"],path
        lines=original.split("\n")
        for e in reversed(item["edits"]):
            i=e["line"];n=len(e["old"])
            assert lines[i:i+n]==e["old"],(path,i)
            lines[i:i+n]=e["next"]
        new="\n".join(lines)
        patch.extend(difflib.unified_diff(original.splitlines(keepends=True),new.splitlines(keepends=True),fromfile="a/"+path,tofile="b/"+path))
    target=out/"v1-f1-candidate.patch"
    target.write_text("".join(patch))
    subprocess.run(["git","-C",str(source),"apply","--check",str(target)],check=True)
    print("V1_F1_PATCH_APPLY_CHECK_PASS",hashlib.sha256(target.read_bytes()).hexdigest())
if __name__=="__main__":main()
