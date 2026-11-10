# SPDX-License-Identifier: Apache-2.0
"""Execute the production move helpers with both metadata worker policies."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
PROGRAM = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define LAUNCH_BOUND(x)
#define __gm__
#define __ubuf__
struct { unsigned x = 0; } threadIdx;
template<class T> T asc_ldcg(T* p) { return *p; }
template<class T> void asc_stcg(T* p, T value) { *p = value; }
uint32_t asc_atomic_cas(uint32_t* p, uint32_t old, uint32_t value) {
  auto result = *p; if (result == old) *p = value; return result;
}
void asc_atomic_or(uint32_t* p, uint32_t value) { *p |= value; }
void asc_syncthreads() {}
void asc_threadfence() {}
uint32_t asc_shfl_up(uint32_t value, uint32_t) { return value; }
#include "index_cache_update.h"

constexpr unsigned history = 32768, resident = 16384, owners = 3, stride = history + 8;
struct Data {
  std::vector<uint64_t> keys, addresses, logical;
  std::vector<uint32_t> scales, valid;
  Data() : keys(owners * history * 16), addresses(owners * stride),
           logical(owners * history), scales(owners * history), valid(owners * resident) {
    std::iota(keys.begin(), keys.end(), 100);
    std::iota(addresses.begin(), addresses.end(), UINT64_C(0x8000000000100000));
    std::iota(logical.begin(), logical.end(), 3000000);
    std::iota(scales.begin(), scales.end(), 4000000);
    for (unsigned i = 0; i < valid.size(); ++i) valid[i] = i % 3 != 0;
  }
  bool operator==(const Data& other) const {
    return keys == other.keys && addresses == other.addresses && logical == other.logical &&
           scales == other.scales && valid == other.valid;
  }
};

template<bool Balanced>
void apply(Data& data, std::vector<uint32_t>& plan, unsigned owner, bool owned, std::mt19937& rng) {
  uint64_t args[26]{};
  args[0] = reinterpret_cast<uint64_t>(data.keys.data());
  args[1] = reinterpret_cast<uint64_t>(data.scales.data());
  args[2] = reinterpret_cast<uint64_t>(data.addresses.data());
  args[3] = reinterpret_cast<uint64_t>(data.logical.data());
  args[7] = stride;
  args[9] = args[16] = reinterpret_cast<uint64_t>(data.valid.data());
  args[24] = history; args[25] = resident;
  std::vector<uint64_t> staged(kMoveScratchBytes / sizeof(uint64_t), UINT64_MAX);
  unsigned groups[] = {0, 1, 2, 3};
  std::shuffle(groups, groups + 4, rng);
  for (unsigned group : groups) {
    if (plan[0] <= kPlanRecords)
      Move<1, Balanced>(args, owner, group, owned, plan.data(), staged.data(), plan[0]);
    else
      MoveExtended<1, Balanced>(args, owner, group, owned, plan.data(), staged.data(), plan[0]);
  }
}

int main() {
  std::mt19937 rng(117);
  const unsigned cases[] = {0, 2, 2048, 2049, 4096, 8192, 10240, 10240};
  for (unsigned trial = 0; trial < 32; ++trial) {
    unsigned n = trial < 8 ? cases[trial] : rng() % (kExtendedCandidates + 1);
    std::vector<uint32_t> slots(kExtendedCandidates), changes(kExtendedCandidates, UINT32_MAX);
    std::vector<uint32_t> destinations(resident), sources(history);
    std::iota(destinations.begin(), destinations.end(), 0);
    std::iota(sources.begin(), sources.end(), 0);
    std::shuffle(destinations.begin(), destinations.end(), rng);
    std::shuffle(sources.begin(), sources.end(), rng);
    for (unsigned i = 0; i < n; ++i) {
      slots[i] = trial < 8 ? i : destinations[i];
      changes[i] = trial < 8 ? resident + i : sources[i];
    }
    if (trial == 1) { changes[0] = 1; changes[1] = 0; }
    if (trial == 7) // Sparse extended inputs retain holes, rather than compacting keys.
      for (unsigned i = 0; i < n; ++i) if (i % 7) changes[i] = UINT32_MAX;
    std::vector<uint32_t> status(128), plan(kExtendedTransferWords);
    if (n <= kPlanCandidates) {
      std::vector<uint32_t> work(kPlanWords);
      BuildUpdatePlan<1>(slots.data(), changes.data(), history, resident, work.data(), status.data());
      assert(!status[0]);
      PublishUpdatePlan<1>(work.data(), status.data(), plan.data(), resident);
    } else {
      std::vector<uint32_t> hashes(2 * kExtendedHash), records(kExtendedWorkWords);
      BuildExtendedPlan<1>(slots.data(), changes.data(), n, history, resident,
                           hashes.data(), status.data(), records.data());
      assert(!status[0]);
      PublishExtendedPlan<1>(records.data(), status.data(), plan.data(), resident);
    }
    if (trial == 2) assert(plan[0] == kPlanRecords);
    if (trial == 3) assert(plan[0] == kPlanRecords + 2);
    if (trial == 6) assert(plan[0] == kExtendedRecords);
    unsigned owner = trial % owners;
    Data original, expected = original, baseline = original, balanced = original;
    for (unsigned i = 0; i < plan[0]; ++i) {
      unsigned tagged = plan[kTransferHeaderWords + 2 * i], dst = tagged & ~kInvalidateHot;
      unsigned src = plan[kTransferHeaderWords + 2 * i + 1];
      for (unsigned field = 0; field < 16; ++field)
        expected.keys[(owner * history + dst) * 16 + field] = original.keys[(owner * history + src) * 16 + field];
      expected.scales[owner * history + dst] = original.scales[owner * history + src];
      expected.addresses[owner * stride + dst] = original.addresses[owner * stride + src];
      expected.logical[owner * history + dst] = original.logical[owner * history + src];
      if (tagged & kInvalidateHot) expected.valid[owner * resident + dst] = 0;
    }
    apply<false>(baseline, plan, owner, trial % 2, rng);
    apply<true>(balanced, plan, owner, trial % 2, rng);
    assert(baseline == expected && balanced == expected);
  }
}
"""


class BalancedIndexMoveTest(unittest.TestCase):
    def test_partial_permutations_cycles_thresholds_and_owner_isolation(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        with tempfile.TemporaryDirectory() as directory:
            source, binary = Path(directory) / "move.cpp", Path(directory) / "move"
            source.write_text(PROGRAM)
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    str(source),
                    "-I" + str(ROOT / "csrc/operators/lightning_indexer_with_fetch_ubmem/op_kernel/resolver"),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
