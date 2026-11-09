import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from kvip.operators.lifetch.index_update import DirectIndexUpdate

ROOT = Path(__file__).resolve().parents[4]


class UpdatePlanTest(unittest.TestCase):
    def test_paths_cycles_and_random_injective_changes(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ unavailable")
        source = '\n#include <cstdint>\n#include <vector>\n#include <iostream>\n#define __simt_callee__\n#define __simt_vf__\n#define __aicore__\n#define LAUNCH_BOUND(x)\n#define SIM_HISTORY 8192\n#define SIM_RESIDENT 4096\n#define __gm__\n#define __ubuf__\nstruct { unsigned x=0; } threadIdx;\ntemplate<class T> T asc_ldcg(T* p) { return *p; }\nuint32_t asc_atomic_cas(uint32_t* p,uint32_t old,uint32_t v) { auto x=*p;if(x==old)*p=v;return x; }\nvoid asc_atomic_or(uint32_t* p,uint32_t v) { *p|=v; }\nvoid asc_syncthreads() {}\nvoid asc_threadfence() {}\ntemplate<class T> void asc_stcg(T* p,T v){*p=v;}\nuint32_t asc_shfl_up(uint32_t v,uint32_t) { return v; }\n#include "index_cache_update.h"\nint main() {\n unsigned n;\n while(std::cin>>n) {\n  std::vector<uint32_t> lru(2048),change(2048,UINT32_MAX),work(kPlanWords),status(128);\n  for(unsigned i=0;i<n;++i)std::cin>>lru[i]>>change[i];\n  BuildUpdatePlan<1>(lru.data(),change.data(),8192,4096,work.data(),status.data());\n  auto dst=work.data()+2*kPlanHash,src=dst+kPlanRecords;\n  std::cout<<status[0];\n  if(!status[0])for(unsigned i=0;i<kPlanRecords;++i)\n   if(src[i]!=UINT32_MAX)std::cout<<" "<<dst[i]<<":"<<src[i];\n  std::cout<<"\\n";\n }\n}\n'
        rng = random.Random(17)
        cases = [[(0, 1), (1, 0)], [(0, 1), (1, 5000)], [(3, 3), (4, 2), (2, 5)], []]
        for _ in range(100):
            n = rng.randrange(1, 2049)
            cases.append(list(zip(rng.sample(range(4096), n), rng.sample(range(8192), n))))
        cases += [[(0, 4), (1, 4)], [(0, 4), (0, 5)], [(4096, 4)], [(0, 8192)]]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "test.cpp").write_text(source)
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    str(folder / "test.cpp"),
                    "-I" + str(ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel"),
                    "-o",
                    str(folder / "test"),
                ],
                check=True,
                capture_output=True,
            )
            data = "".join(str(len(c)) + " " + " ".join((f"{d} {s}" for d, s in c)) + "\n" for c in cases)
            result = subprocess.run([str(folder / "test")], input=data, text=True, capture_output=True, check=True)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), len(cases))
        for case, line in zip(cases[:-4], lines[:-4]):
            fields = line.split()
            self.assertEqual(fields[0], "0")
            records = [tuple(map(int, item.split(":"))) for item in fields[1:]]
            old = list(range(8192))
            new = old.copy()
            for dst, src in records:
                new[dst] = old[src]
            self.assertEqual(sorted(new), old)
            kv = old.copy()
            written = {dst for dst, _ in case}
            invalidated = {dst for dst, src in records[len(case) :] if dst < 4096 and dst != src}
            self.assertTrue(written.isdisjoint(invalidated))
            # Apply's invalidations and Fetch's residency publication commute.
            before = [slot % 3 == 0 for slot in range(4096)]
            fetch_first, update_first = before.copy(), before.copy()
            for slots, value in ((written, True), (invalidated, False)):
                for slot in slots:
                    fetch_first[slot] = value
            for slots, value in ((invalidated, False), (written, True)):
                for slot in slots:
                    update_first[slot] = value
            self.assertEqual(fetch_first, update_first)
            for dst, src in case:
                kv[dst] = old[src]
            valid = [True] * 4096
            for dst, src in records:
                if dst < 4096 and dst != src and (dst not in written):
                    valid[dst] = False
            for slot in range(4096):
                if valid[slot]:
                    self.assertEqual(kv[slot], new[slot])
            for dst, src in case:
                self.assertEqual(new[dst], src)
            self.assertEqual(len({d for d, _ in records}), len(records))
        for line in lines[-4:]:
            self.assertEqual(line, "1")


ROOT = Path(__file__).resolve().parents[4]


class UpdateMoveTest(unittest.TestCase):
    def test_cycles_sparse_records_fields_and_owner_isolation(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        program = '\n#include <cstdint>\n#include <vector>\n#include <numeric>\n#include <algorithm>\n#include <random>\n#include <cassert>\n#define __simt_callee__\n#define __simt_vf__\n#define __aicore__\n#define LAUNCH_BOUND(x)\n#define SIM_HISTORY 8192\n#define SIM_RESIDENT 4096\n#define __simt_vf__\n#define __aicore__\n#define __gm__\n#define __ubuf__\n#define LAUNCH_BOUND(x)\n#define SIM_HISTORY 8192\n#define SIM_RESIDENT 4096\nstruct { unsigned x=0; } threadIdx;\ntemplate<class T> T asc_ldcg(T* p) { return *p; }\ntemplate<class T> void asc_stcg(T* p,T v) { *p=v; }\nuint32_t asc_atomic_cas(uint32_t* p,uint32_t old,uint32_t v) { auto x=*p;if(x==old)*p=v;return x; }\nvoid asc_atomic_or(uint32_t* p,uint32_t v) { *p|=v; }\nvoid asc_syncthreads() {}\nvoid asc_threadfence() {}\nuint32_t asc_shfl_up(uint32_t v,uint32_t) { return v; }\n#include "index_cache_update.h"\n\nint main() {\n std::mt19937 rng(17);\n for(unsigned trial=0;trial<70;++trial) {\n  std::vector<uint32_t> lru(2048),changes(2048,UINT32_MAX),work(kPlanWords),status(128),plan(kTransferPlanWords);\n  std::vector<uint32_t> dsts(SIM_RESIDENT),srcs(SIM_HISTORY);\n  std::iota(dsts.begin(),dsts.end(),0);std::iota(srcs.begin(),srcs.end(),0);\n  std::shuffle(dsts.begin(),dsts.end(),rng);std::shuffle(srcs.begin(),srcs.end(),rng);\n  unsigned n=trial==0?0:(trial==1?2:(trial==2?2048:rng()%2049));\n  for(unsigned i=0;i<n;++i){lru[i]=dsts[i];changes[i]=srcs[i];}\n  if(trial==1){lru[0]=0;lru[1]=1;changes[0]=1;changes[1]=0;}\n  BuildUpdatePlan<1>(lru.data(),changes.data(),SIM_HISTORY,SIM_RESIDENT,work.data(),status.data());\n  assert(status[0]==0);\n  PublishUpdatePlan<1>(work.data(),status.data(),plan.data(),SIM_RESIDENT);\n  auto dst=work.data()+2*kPlanHash,src=dst+kPlanRecords;\n  unsigned count=0;\n  for(unsigned i=0;i<kPlanRecords;++i)if(src[i]!=UINT32_MAX && src[i]!=dst[i]) {\n    auto tagged=dst[i]|((i>=kPlanCandidates && dst[i]<SIM_RESIDENT)?kInvalidateHot:0);\n    assert(plan[8+2*count]==tagged && plan[9+2*count]==src[i]);++count;\n  }\n  assert(plan[0]==count);\n  constexpr unsigned owners=2,stride=SIM_HISTORY+8;\n  std::vector<uint64_t> keys(owners*SIM_HISTORY*16),addresses(owners*stride),ids(owners*SIM_HISTORY);\n  std::vector<uint32_t> scales(owners*SIM_HISTORY),valid(owners*SIM_RESIDENT,1);\n  std::iota(keys.begin(),keys.end(),100);std::iota(addresses.begin(),addresses.end(),1000000);\n  std::iota(ids.begin(),ids.end(),2000000);std::iota(scales.begin(),scales.end(),3000000);\n  auto ek=keys,ea=addresses,ei=ids;auto es=scales,ev=valid;\n  unsigned owner=trial%2;\n  for(unsigned i=0;i<kPlanRecords;++i)if(src[i]!=UINT32_MAX && src[i]!=dst[i]) {\n    for(unsigned f=0;f<16;++f)ek[(owner*SIM_HISTORY+dst[i])*16+f]=keys[(owner*SIM_HISTORY+src[i])*16+f];\n    ea[owner*stride+dst[i]]=addresses[owner*stride+src[i]];\n    ei[owner*SIM_HISTORY+dst[i]]=ids[owner*SIM_HISTORY+src[i]];\n    es[owner*SIM_HISTORY+dst[i]]=scales[owner*SIM_HISTORY+src[i]];\n    if(i>=kPlanCandidates && dst[i]<SIM_RESIDENT)ev[owner*SIM_RESIDENT+dst[i]]=0;\n  }\n  uint64_t a[26]{};a[24]=SIM_HISTORY;a[25]=SIM_RESIDENT;a[0]=(uint64_t)keys.data();a[1]=(uint64_t)scales.data();\n  a[2]=(uint64_t)addresses.data();a[3]=(uint64_t)ids.data();a[7]=stride;\n  a[9]=a[16]=(uint64_t)valid.data();\n  std::vector<uint64_t> staged(kPlanRecords*kKeyFieldsPerWorker);\n  unsigned groups[]={0,1,2,3};std::shuffle(groups,groups+4,rng);\n  for(auto g:groups)Move<1>(a,owner,g,trial%2,plan.data(),staged.data(),count);\n  assert(keys==ek && addresses==ea && ids==ei && scales==es && valid==ev);\n }\n}\n'
        extended = program.replace("SIM_HISTORY 8192", "SIM_HISTORY 32768").replace(
            "SIM_RESIDENT 4096", "SIM_RESIDENT 16384"
        )
        extended = extended.replace("trial<70", "trial<20").replace("2048", "10240").replace("2049", "10241")
        extended = extended.replace("work(kPlanWords)", "work(2*kExtendedRecords), hashes(2*kExtendedHash)")
        extended = extended.replace("plan(kTransferPlanWords)", "plan(kExtendedTransferWords)")
        extended = extended.replace(
            "BuildUpdatePlan<1>(lru.data(),changes.data(),SIM_HISTORY,SIM_RESIDENT,work.data(),status.data())",
            "BuildExtendedPlan<1>(lru.data(),changes.data(),n,SIM_HISTORY,SIM_RESIDENT,hashes.data(),status.data(),work.data())",
        )
        extended = extended.replace("PublishUpdatePlan<1>", "PublishExtendedPlan<1>")
        extended = extended.replace("work.data()+2*kPlanHash", "work.data()")
        extended = extended.replace("kPlanRecords", "kExtendedRecords").replace(
            "kPlanCandidates", "kExtendedCandidates"
        )
        extended = extended.replace("Move<1>", "MoveExtended<1>")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, text in (("small", program), ("extended", extended)):
                with self.subTest(planner=name):
                    cpp, exe = root / (name + ".cpp"), root / name
                    cpp.write_text(text)
                    subprocess.run(
                        [
                            compiler,
                            "-std=c++17",
                            "-O2",
                            str(cpp),
                            "-I" + str(ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel"),
                            "-o",
                            str(exe),
                        ],
                        check=True,
                    )
                    subprocess.run([str(exe)], check=True)


class EarlyUpdateTest(unittest.TestCase):
    def test_pending_dma_and_late_error_with_actual_device_functions(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        kernel = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel"
        text = (kernel / "index_cache_update.cpp").read_text()
        update = text[
            text.index("template <bool WaitForFetch, bool Extended>") : text.index(
                "// Select the original small planner"
            )
        ]
        source = r"""
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert>
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(x)
struct { unsigned x=0; } threadIdx;
template<class T> T asc_ldcg(T* p) { return *p; }
template<class T> void asc_stcg(T* p,T v) { *p=v; }
uint32_t asc_atomic_cas(uint32_t* p,uint32_t old,uint32_t v) {
    auto x=*p; if(x==old)*p=v; return x;
}
void asc_atomic_or(uint32_t* p,uint32_t v) { *p|=v; }
void asc_syncthreads() {}
void asc_threadfence() {}
uint32_t asc_shfl_up(uint32_t v,uint32_t) { return v; }
#include "index_cache_update.h"
constexpr uint32_t kCandidates=2048, kThreads=1;
UPDATE_FUNCTION
int main() {
    constexpr uint32_t resident=4096, history=8192, statusBytes=1216;
    std::vector<uint32_t> lru(kCandidates), changes(kCandidates,UINT32_MAX);
    std::iota(lru.begin(),lru.end(),0);
    changes[7]=5000;
    std::vector<uint32_t> work(kPlanWords+128), plan(kTransferPlanWords);
    std::vector<uint32_t> rows(256), rowStatus(32), posts(2*32), valid(2*resident,1);
    std::vector<uint64_t> prefetch(2*statusBytes/8), final(2*statusBytes/8);
    uint32_t error=0, owner=1;
    uint64_t a[28]{}; uint32_t count=2048; a[26]=(uint64_t)&count;
    a[4]=(uint64_t)lru.data(); a[5]=(uint64_t)changes.data(); a[6]=(uint64_t)&error;
    a[8]=(uint64_t)rows.data(); a[9]=(uint64_t)&owner; a[10]=(uint64_t)rowStatus.data();
    a[11]=(uint64_t)posts.data(); a[12]=(uint64_t)prefetch.data(); a[13]=(uint64_t)final.data();
    a[14]=2; a[15]=statusBytes; a[16]=(uint64_t)valid.data(); a[17]=(uint64_t)plan.data();
    a[24]=history; a[25]=resident;
    // Resolver has stable records, but prefetch/final CQs are still pending.
    rows[211]=rows[248]=posts[1]=posts[35]=1;
    Update<false, false>(a,0,owner,true,work.data());
    assert(error==0 && plan[0]>0);
    Update<true, false>(a,0,owner,true,work.data());
    assert(error!=0 && plan[0]==0);
    // Standalone API keeps its completed-transport validation.
    rows[211]=rows[248]=posts[1]=posts[35]=0;
    for(auto states:{prefetch.data(),final.data()})
        for(unsigned lane=0;lane<2;++lane)states[lane*statusBytes/8]=0x4B5641464247444FULL;
    Update<true, false>(a,0,owner,true,work.data());
    assert(error==0 && plan[0]>0);
    for(auto fault:{&rows[11],&rowStatus[0],&posts[34]}) {
        *fault=1;
        Update<false, false>(a,0,owner,true,work.data());
        assert(error!=0 && plan[0]==0);
        *fault=0;
    }
    error=0;
    CompleteEarlyUpdate<1>(a,0,0);
    assert(error==0 && std::all_of(valid.begin(),valid.end(),[](auto v){return v==1;}));
    CompleteEarlyUpdate<1>(a,0,1);
    assert(error==4);
    assert(std::all_of(valid.begin(),valid.begin()+resident,[](auto v){return v==1;}));
    assert(std::all_of(valid.begin()+resident,valid.end(),[](auto v){return v==0;}));
    std::fill(valid.begin(),valid.end(),1);
    error=2;
    CompleteEarlyUpdate<1>(a,0,0);
    assert(error==2 && valid[0]==1 && valid[resident]==0);
}
""".replace("UPDATE_FUNCTION", update)
        with tempfile.TemporaryDirectory() as directory:
            cpp, exe = Path(directory) / "test.cpp", Path(directory) / "test"
            cpp.write_text(source)
            subprocess.run(
                [compiler, "-std=c++17", "-O2", str(cpp), "-I" + str(kernel), "-o", str(exe)],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([str(exe)], check=True, capture_output=True, text=True)


class IndexUpdateTest(unittest.TestCase):
    def setUp(self):
        self.library = SimpleNamespace(
            kvop_direct_update_abi_version=Mock(return_value=3),
            kvop_direct_update_work_words=Mock(return_value=40960),
            kvop_direct_update_plan_words=Mock(return_value=8200),
            kvop_direct_update_resident=Mock(return_value=128),
            kvop_direct_update_owned=Mock(return_value=0),
            kvop_direct_update_outputs=Mock(return_value=0),
        )
        self.rotating = SimpleNamespace(
            lru_slots=torch.zeros((2, 2048), dtype=torch.int32),
            change_slots=torch.zeros((2, 10240), dtype=torch.int32),
            change_count=torch.full((2,), 2048, dtype=torch.int32),
            changes=torch.full((2, 10240), -1, dtype=torch.int32),
            rows=torch.zeros((2, 256), dtype=torch.int32),
            hot_valid=torch.zeros((5, 128), dtype=torch.int32),
            row_status=torch.zeros((2, 32), dtype=torch.int32),
            final_posts=torch.zeros((2, 2, 32), dtype=torch.int32),
            prefetch_status=torch.zeros((4, 1216), dtype=torch.uint8),
            final_status=torch.zeros((4, 1216), dtype=torch.uint8),
            final_workers=2,
            owner=SimpleNamespace(abi_sizes={"status": 1216}),
        )
        self.inputs = dict(
            keys=torch.zeros((5, 256, 128), dtype=torch.int8),
            scales=torch.ones((5, 256)),
            addresses=torch.zeros((5, 256), dtype=torch.int64),
            logical_ids=torch.zeros((5, 256), dtype=torch.int64),
            row_owners=torch.tensor([4, 1], dtype=torch.int32),
            rotating=self.rotating,
        )
        self.torch = SimpleNamespace(
            zeros=torch.zeros,
            empty=torch.empty,
            tensor=torch.tensor,
            int32=torch.int32,
            int64=torch.int64,
            float32=torch.float32,
            npu=SimpleNamespace(current_stream=lambda device: SimpleNamespace(npu_stream=123)),
        )

    def make(self):
        with patch("kvip.operators.lifetch.index_update.ctypes.CDLL", return_value=self.library):
            return DirectIndexUpdate(self.torch, "test", **self.inputs)

    def test_stable_owner_rows_and_replay_without_allocation(self):
        update = self.make()
        self.assertEqual(
            update.bindings.tolist(),
            [self.inputs[name].data_ptr() for name in ("keys", "scales", "addresses", "logical_ids")]
            + [
                self.rotating.change_slots.data_ptr(),
                self.rotating.changes.data_ptr(),
                update.status.data_ptr(),
                256,
                self.rotating.rows.data_ptr(),
                self.inputs["row_owners"].data_ptr(),
                self.rotating.row_status.data_ptr(),
                self.rotating.final_posts.data_ptr(),
                self.rotating.prefetch_status.data_ptr(),
                self.rotating.final_status.data_ptr(),
                2,
                1216,
                self.rotating.hot_valid.data_ptr(),
                update.plan_buffer.data_ptr(),
            ]
            + [0] * 6
            + [256, 128, self.rotating.change_count.data_ptr(), update.work_buffer.data_ptr()],
        )
        with patch.object(self.torch, "tensor", side_effect=AssertionError("replay allocation")):
            update()
            self.inputs["row_owners"].copy_(torch.tensor([1, 3], dtype=torch.int32))
            update()
        self.assertEqual(self.library.kvop_direct_update_owned.call_count, 2)
        self.library.kvop_direct_update_owned.assert_called_with(2, update.bindings.data_ptr(), 123)

    def test_service_output_binding_carries_runtime_geometry(self):
        query_inputs = SimpleNamespace(unpack_indices=torch.arange(8), output_valid=torch.ones(8, dtype=torch.int64))
        self.rotating.arguments = [torch.zeros(1)] * 9 + [torch.zeros(2, 4, 2048, dtype=torch.int32)]
        self.rotating.locations = torch.zeros(2, 4, 2048, dtype=torch.int32)
        self.inputs["query_inputs"] = query_inputs
        outputs = torch.zeros(2, 8, 1, 2048, dtype=torch.int32)
        self.inputs["outputs"] = outputs
        update = self.make()
        self.assertEqual(update.bindings.numel(), 28)
        self.assertEqual(update.bindings[22], outputs[0].data_ptr())
        self.assertEqual(update.bindings[23], outputs[1].data_ptr())
        update()
        self.library.kvop_direct_update_outputs.assert_called_once()
        self.library.kvop_direct_update_owned.assert_not_called()

    def test_reject_old_update_abi_before_launch(self):
        self.library.kvop_direct_update_abi_version.return_value = 1
        with self.assertRaisesRegex(ValueError, "ABI 3"):
            self.make()
        self.library.kvop_direct_update_owned.assert_not_called()

    def test_reject_wrong_producer_batch(self):
        self.rotating.changes = self.rotating.changes[:1]
        with self.assertRaisesRegex(ValueError, "producer changes"):
            self.make()

    def test_actual_output_function_handles_owners_padding_and_invalid_topk(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ unavailable")
        root = Path(__file__).resolve().parents[4]
        text = (root / "csrc/operators/lightning_indexer_with_fetch/op_kernel/index_cache_update.cpp").read_text()
        function = text[
            text.index("__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishOutputs") : text.index(
                "template <bool Owned>"
            )
        ]
        source = "\n#include <cstdint>\n#include <vector>\n#include <cassert>\n#define __simt_vf__\n#define __aicore__\n#define __gm__\n#define LAUNCH_BOUND(x)\n#define SIM_HISTORY 64\nstruct { uint32_t x; } threadIdx;\ntemplate<class T> T asc_ldcg(T* p) { return *p; }\ntemplate<class T> void asc_stcg(T* p,T v) { *p=v; }\nFUNCTION\nint main() {\n std::vector<int32_t> topk(8*2048,-1),slots(8*2048,-1),out(8*2048,999),dest(8*2048,999);\n std::vector<int64_t> ids(3*64),unpack={0,4,5,0,0,0,0,0},valid={1,1,1,0,0,0,0,0};\n uint32_t owners[]={2,1}; uint64_t a[26]={};a[24]=SIM_HISTORY;\n for(int i=0;i<3*64;++i)ids[i]=i+100;\n for(int r=0;r<8;++r) {topk[r*2048]=r+1;slots[r*2048]=r+20;}\n a[3]=(uint64_t)ids.data();a[9]=(uint64_t)owners;\n a[18]=(uint64_t)topk.data();a[19]=(uint64_t)slots.data();\n a[20]=(uint64_t)unpack.data();a[21]=(uint64_t)valid.data();\n a[22]=(uint64_t)out.data();a[23]=(uint64_t)dest.data();\n for(uint32_t b=0;b<2;++b)for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)PublishOutputs(a,b);\n for(int r=0;r<8;++r)for(int k=0;k<2048;++k) {\n  int q=unpack[r];bool active=valid[r] && k==0;\n  assert(out[r*2048+k]==(active?100+owners[q/4]*64+q+1:-1));\n  assert(dest[r*2048+k]==(active?q+20:-1));\n }\n}\n".replace(
            "FUNCTION", function
        )
        with tempfile.TemporaryDirectory() as folder:
            cpp, exe = (Path(folder) / "test.cpp", Path(folder) / "test")
            cpp.write_text(source)
            subprocess.run([compiler, str(cpp), "-o", str(exe)], check=True, capture_output=True)
            subprocess.run([str(exe)], check=True, capture_output=True)

    def test_fixture_reuses_production_binding_with_padded_address_rows(self):
        from index_update import DirectCacheUpdate

        self.rotating.hot_valid = self.rotating.hot_valid[:2]
        self.rotating.arguments = [torch.zeros(1)] * 9 + [torch.zeros(2, 4, 2048, dtype=torch.int32)]
        self.rotating.locations = torch.zeros(2, 4, 2048, dtype=torch.int32)
        addresses = torch.zeros(2, 300, dtype=torch.int64)[:, :256]
        with patch("kvip.operators.lifetch.index_update.ctypes.CDLL", return_value=self.library):
            update = DirectCacheUpdate(
                torch,
                self.inputs["keys"][:2],
                self.inputs["scales"][:2],
                addresses,
                None,
                2,
                256,
                128,
                library="test",
                topk=self.rotating.arguments[9],
                locations=self.rotating.locations,
                rotating=self.rotating,
            )
        self.assertIsInstance(update, DirectIndexUpdate)
        self.assertEqual(update.bindings[2], addresses.data_ptr())
        self.assertEqual(update.bindings[7], 300)
        self.assertEqual(update.bindings[22], update.logical_output.data_ptr())
        self.assertEqual(update.bindings[23], update.slot_output.data_ptr())
        self.assertEqual(update.bindings[24:26].tolist(), [256, 128])

    def test_report_launch_error(self):
        update = self.make()
        self.library.kvop_direct_update_owned.return_value = 7
        with self.assertRaisesRegex(RuntimeError, "launch failed: 7"):
            update()
