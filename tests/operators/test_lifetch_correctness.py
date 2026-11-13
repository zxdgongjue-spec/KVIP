"""CPU regressions for the unified LIFetch correctness module."""
import inspect
import random
import shutil
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from correctness import boundary_evidence, require_boundary
from diagnostics import account_ubmem, final_transfer_scope, summarize
from kvip.operators.lifetch.binding import LIFetchBinding
from kvip.operators.lifetch.geometry import prefetch_budget
from kvip.operators.lifetch.index_update import DirectIndexUpdate
from kvip.operators.lifetch_ubmem import build as specialization
from kvip.operators.lifetch_ubmem.binding import UbmemBinding
from reporting import atomic_json

ROOT = Path(__file__).resolve().parents[2]


LANES = 128
VICTIMS = 2048
RESIDENT = 16384

SOURCE_ROOT = Path(__file__).resolve().parent / 'lifetch'


def evidence(*, selected=(0, 1), valid=(1, 1, 1, 1), posted=0, final=(0, 0, 0, 0)):
    ids = np.tile(selected, (1, 4, 1))
    return boundary_evidence(ids, [valid], [range(4)], np.array([posted]), [final])


def test_hit_case_cannot_pass_with_reads_or_cold_inputs():
    require_boundary('all_hit_budget0', {'regular': evidence()})
    for rows in (evidence(posted=1), evidence(selected=(0, 5)), evidence(final=(1, 0, 0, 0))):
        with pytest.raises(AssertionError, match='zero reads'):
            require_boundary('all_hit_budget0', {'regular': rows})


def test_all_miss_distinguishes_full_unrelated_hot_from_cold_start():
    full = evidence(selected=(4, 5), final=(2, 0, 0, 0))
    require_boundary('all_miss', {'regular': full})
    cold = evidence(selected=(4, 5), valid=(0, 0, 0, 0), final=(2, 0, 0, 0))
    with pytest.raises(AssertionError, match='full unrelated'):
        require_boundary('all_miss', {'regular': cold})
    assert cold[0]['initial_cold'] == full[0]['initial_cold'] == 2


def test_zero_budget_still_requires_actual_final_repair():
    require_boundary('prefetch_zero', {'regular': evidence(selected=(0, 4), final=(1, 0, 0, 0))})
    with pytest.raises(AssertionError, match='final repair'):
        require_boundary('prefetch_zero', {'regular': evidence(selected=(0, 4))})


def test_cross_query_empty_repair_and_cold_start_have_separate_evidence():
    regular = evidence(selected=(0, 4), final=(1, 0, 0, 0))
    cold = evidence(selected=(0, 4), valid=(0, 0, 0, 0), final=(2, 0, 0, 0))
    assert regular[0]['cross_query_duplicates'] == [0, 2, 2, 2]
    extra = {'transfer': [dict(fallback_slots=1, conflicts=1)], 'padding': 'PASS'}
    require_boundary('boundary_regular', {'regular': regular, 'cold-start': cold, **extra})
    with pytest.raises(AssertionError, match='clear validity'):
        require_boundary('boundary_regular', {'regular': regular, 'cold-start': regular, **extra})
    with pytest.raises(AssertionError, match='no new repair'):
        require_boundary('boundary_regular', {'regular': evidence(final=(0, 0, 1, 0)), 'cold-start': cold, **extra})


def test_invalid_counter_data_is_not_a_correctness_assertion():
    with pytest.raises(ValueError, match='invalid counters'):
        evidence(posted=-1)


def test_unknown_boundary_and_empty_evidence_cannot_pass():
    import pytest
    from correctness import require_boundary
    with pytest.raises(ValueError, match='Unknown boundary'):
        require_boundary('typo', {'regular': []})
    with pytest.raises(AssertionError, match='no observed'):
        require_boundary('all_hit_budget0', {'regular': []})


def test_extended_contracts_use_production_validation():
    from correctness import check_contracts
    for transport in ('urma', 'ubmem'):
        checks = check_contracts(transport)
        assert all(row['correctness'] == 'PASS' for row in checks), checks
        names = {row['case'] for row in checks}
        assert {'reject_unaligned_history', 'reject_unaligned_resident',
                'reject_no_cold_history', 'reject_window_budget481'} <= names
        if transport == 'ubmem':
            assert {'reject_heads', 'reject_token_bytes', 'reject_soc'} <= names


def test_transfer_coverage_requires_real_outside_candidate_slots_and_conflicts():
    from correctness import require_transfer_coverage, transfer_coverage
    posts = np.zeros((1, 4, 6), dtype=np.int32)
    posts[0, 0] = [3, 0, 0, 0, 0, 2]
    result = transfer_coverage(posts, [0, 3, 1], [[0, 1]], 8, 4)
    assert result == [dict(request=0, final=3, conflicts=1, fallback_slots=1)]
    require_transfer_coverage(result)
    for key in ('conflicts', 'fallback_slots'):
        with pytest.raises(AssertionError):
            require_transfer_coverage([dict(result[0], **{key: 0})])
    posts[0, 0, 4] = 9
    with pytest.raises(RuntimeError, match='range'):
        transfer_coverage(posts, [0, 3, 1], [[0, 1]], 8, 4)


class IndexUpdateTest(unittest.TestCase):
    def setUp(self):
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
        return DirectIndexUpdate(self.torch, **self.inputs)

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
                0,
            ]
            + [0] * 6
            + [256, 128, self.rotating.change_count.data_ptr(), 0],
        )
        before = update.bindings.clone()
        self.inputs["row_owners"].copy_(torch.tensor([1, 3], dtype=torch.int32))
        self.assertTrue(torch.equal(before, update.bindings))
        self.assertFalse(callable(update))

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
        self.assertFalse(callable(update))

    def test_reject_wrong_producer_batch(self):
        self.rotating.changes = self.rotating.changes[:1]
        with self.assertRaisesRegex(ValueError, "producer changes"):
            self.make()

    def test_fixture_reuses_production_binding_with_padded_address_rows(self):
        from execution import DirectCacheUpdate

        self.rotating.hot_valid = self.rotating.hot_valid[:2]
        self.rotating.arguments = [torch.zeros(1)] * 9 + [torch.zeros(2, 4, 2048, dtype=torch.int32)]
        self.rotating.locations = torch.zeros(2, 4, 2048, dtype=torch.int32)
        addresses = torch.zeros(2, 300, dtype=torch.int64)[:, :256]
        update = DirectCacheUpdate(
            torch,
            self.inputs["keys"][:2],
            self.inputs["scales"][:2],
            addresses,
            None,
            2,
            256,
            128,
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


def pack_predicate(predicate):
    """Model the consumer's low bit in each two-bit token field."""
    words = [0] * (len(predicate) // 16)
    for lane, selected in enumerate(predicate):
        if selected:
            words[lane // 16] |= 1 << (2 * (lane % 16))
    return b"".join(word.to_bytes(4, "little") for word in words)


def reload_predicate(stored):
    words = [int.from_bytes(stored[i : i + 4], "little") for i in range(0, len(stored), 4)]
    return [bool(word & (1 << (2 * lane))) for word in words for lane in range(16)]


def base_bitmap(scores, victims, threshold, end):
    """Model ClassifyScore -> UB store/reload -> visibility AND."""
    result = bytearray()
    for begin in range(0, VICTIMS, LANES):
        threshold_mask = [score > threshold for score in scores[begin : begin + LANES]]
        reloaded = reload_predicate(pack_predicate(threshold_mask))
        result.extend(
            pack_predicate(
                [selected and token < end for selected, token in zip(reloaded, victims[begin : begin + LANES])]
            )
        )
    return bytes(result)


def register_bitmap(scores, victims, threshold, end):
    """Model register threshold mask AND register visibility mask, one store."""
    result = bytearray()
    for begin in range(0, VICTIMS, LANES):
        score_words = scores[begin : begin + LANES]
        token_words = victims[begin : begin + LANES]
        threshold_mask = sum(1 << i for i, score in enumerate(score_words) if score > threshold)
        visible_mask = sum(1 << i for i, token in enumerate(token_words) if token < end)
        selected = threshold_mask & visible_mask
        result.extend(pack_predicate([bool(selected & (1 << i)) for i in range(LANES)]))
    return bytes(result)


class ResidentPredicateContractTest(unittest.TestCase):
    def check_case(self, scores, victims, threshold, end):
        expected_flags = [score > threshold and token < end for score, token in zip(scores, victims)]
        expected = pack_predicate(expected_flags)
        self.assertEqual(len(expected), 512)
        self.assertEqual(base_bitmap(scores, victims, threshold, end), expected)
        self.assertEqual(register_bitmap(scores, victims, threshold, end), expected)
        self.assertEqual(reload_predicate(expected), expected_flags)
        # Odd bits are not candidates in FastService's 0x55555555 mask.
        for offset in range(0, len(expected), 4):
            self.assertEqual(int.from_bytes(expected[offset : offset + 4], "little") & 0xAAAAAAAA, 0)

    def test_equal_scores_keep_strict_gt(self):
        victims = list(range(VICTIMS))
        for threshold in (0, 1, 0x3F80, 0x7FFF, 0xFFFF):
            with self.subTest(threshold=threshold):
                self.check_case([threshold] * VICTIMS, victims, threshold, RESIDENT)
                self.assertFalse(any(register_bitmap([threshold] * VICTIMS, victims, threshold, RESIDENT)))

    def test_causal_boundary_and_all_mask_extremes(self):
        # Deliberately include token=end-1, end, end+1 across vector boundaries.
        victims = [(i * 97) % RESIDENT for i in range(VICTIMS)]
        for end in (0, 1, 127, 128, 129, 2048, RESIDENT - 1, RESIDENT):
            for threshold in (0, 0x7FFF, 0xFFFF):
                with self.subTest(end=end, threshold=threshold):
                    scores = [0xFFFF] * VICTIMS
                    victims[127:130] = [max(0, end - 1), min(RESIDENT - 1, end), min(RESIDENT - 1, end + 1)]
                    self.check_case(scores, victims, threshold, end)

    def test_random_unsigned_scores_and_sparse_victims(self):
        rng = random.Random(20261105)
        for case in range(48):
            threshold = rng.randrange(0x10000)
            scores = [rng.randrange(0x10000) for _ in range(VICTIMS)]
            # BF16 ties must remain excluded, including each chunk boundary.
            for lane in range(0, VICTIMS, LANES):
                scores[lane] = threshold
                scores[lane + LANES - 1] = threshold
            victims = rng.sample(range(RESIDENT), VICTIMS)
            end = rng.randrange(RESIDENT + 1)
            with self.subTest(case=case, threshold=threshold, end=end):
                self.check_case(scores, victims, threshold, end)


def test_host_pipeline_order_and_first_error(tmp_path):
    import subprocess

    source = (ROOT / "csrc/operators/lightning_indexer_with_fetch/op_host/launch.cpp").read_text()
    begin = source.index('extern "C" uint32_t kvip_lifetch_streaming_forward(')
    entry = source[begin:source.index('\nextern "C" uint32_t aclrtlaunch_kvip_transport_snapshot', begin)]
    stub = r"""
#include <cstdint>
#include <cassert>
#include "runtime_shape.h"
namespace pipeline_ab { constexpr uint32_t kResolveCores=3; }
int fail=0, count=0;
void* stream_expected=(void*)3;
uint32_t step(int n,void* a,void* stream) { assert(a==(void*)1);assert(stream==stream_expected);assert(++count==n);return fail==n?99:0; }
uint32_t v9_init(void*a,void*s,const rotating::RuntimeShape*){return step(1,a,s);}
uint32_t v9_li(void*a,void*s,const rotating::RuntimeShape*){return step(2,a,s);}
extern "C" uint32_t aclrtlaunch_v9_streaming_resolve_update(uint32_t cores,void*s,void*a){assert(cores==56);return step(3,a,s);}
"""
    main = r"""
int main(){rotating::RuntimeShape shape{};shape.batch=16;shape.finalWorkers=4;shape.aicCount=32;
for(fail=0;fail<=3;++fail){count=0;assert(kvip_lifetch_streaming_forward((void*)1,(void*)2,(void*)1,stream_expected,&shape,2)==(fail?99:0));assert(count==(fail?fail:3));}
count=0;shape.batch=17;assert(kvip_lifetch_streaming_forward((void*)1,(void*)2,(void*)1,stream_expected,&shape,2)!=0);assert(count==0);return 0;}
"""
    path = tmp_path / "launch.cpp"
    path.write_text(stub + entry + main)
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-I" + str(ROOT / "csrc/operators/lightning_indexer_with_fetch/include"),
            str(path),
            "-o",
            str(tmp_path / "test"),
        ],
        check=True,
    )
    subprocess.run([str(tmp_path / "test")], check=True)


def test_default_fixture_preserves_requested_misses(tmp_path):
    import numpy as np
    from workloads import prepare_full_chain_fixture

    root = tmp_path / 'workloads'
    prepare_full_chain_fixture(root, 131072, 16384, 10, index_heads=32)
    order = np.load(root / "physical_to_logical.npy")
    scores = np.load(root / "logical_scores.npy")[:, order]
    selected = []
    for row in scores:
        threshold = np.partition(row, -2048)[-2048]
        tokens = np.flatnonzero(row >= threshold)
        assert len(tokens) == 2048
        assert np.count_nonzero(tokens >= 16384) == 205
        selected.extend(tokens[tokens >= 16384])
    assert len(set(selected)) == 410


def test_runtime_tiling_matches_original_geometry(tmp_path):
    import ctypes
    import subprocess

    import pytest
    from kvip.operators.lifetch.geometry import final_aivs, li_geometry, trigger_tokens
    from kvip.operators.lifetch.tiling import RuntimeShape

    source = tmp_path / "shape.cpp"
    # Compile the actual host tiling and device-shared geometry helpers.
    (tmp_path / "batch_get_abi.h").write_text("")
    source.write_text("""
#include "context.h"
static_assert(sizeof(rotating::Context)==320,"context ABI");
extern "C" uint32_t kvip_prepare_shape(rotating::RuntimeShape* s){return rotating::PrepareShape(s);}
extern "C" uint32_t trigger(rotating::RuntimeShape* s,uint32_t e){return rotating::Trigger(s->history,s->resident,s->rounds,e);}
extern "C" uint32_t core(rotating::RuntimeShape* s,uint32_t e,uint32_t g){return rotating::ServiceCore(s->batch,s->splitQ,s->serviceAics,e,g);}
extern "C" uint32_t queue(rotating::RuntimeShape* s,uint32_t b,uint32_t w){return rotating::FinalQueue(s->batch,s->finalWorkers,b,w);}
""")
    library = tmp_path / "shape.so"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(source),
            "-I" + str(ROOT / "csrc/operators/lightning_indexer_with_fetch/include"),
            "-I" + str(tmp_path),
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(library))
    for aics in (16, 24, 32):
        for batch in range(1, aics + 1):
            for history, resident in ((65536, 8192), (131072, 16384), (196608, 32768)):
                for windows in (1, 4, 8, 16, 32):
                    for heads in (32, 64):
                        shape = RuntimeShape.prepare(
                            lib,
                            batch=batch,
                            history=history,
                            resident=resident,
                            index_heads=heads,
                            aic_count=aics,
                            windows=windows,
                        )
                        geo = li_geometry(batch, "auto", windows, aics)
                        assert shape.liCores == geo["rotating_aic"]
                        assert shape.finalWorkers == final_aivs(batch, "auto", aics)
                        assert shape.serviceLanes == geo["service_aivs_per_window"]
                        assert shape.rounds == geo["intermediate_fetches"]
                        pointer = ctypes.byref(shape)
                        assert [128 * lib.trigger(pointer, e) for e in range(shape.rounds)] == trigger_tokens(
                            batch, history, resident, "auto", windows, aics
                        )
                        assert [
                            [lib.core(pointer, e, g) for g in range(shape.serviceAics)] for e in range(shape.rounds)
                        ] == geo["service_groups"]
                        queues = [lib.queue(pointer, b, w) for b in range(batch) for w in range(shape.finalWorkers)]
                        assert len(set(queues)) == len(queues)
                        assert set(queues).isdisjoint(range(0, 2 * batch, 2))
    for changes in ({"history": 1048576}, {"resident": 12345}, {"batch": 33}, {"index_heads": 16}, {"windows": 0}):
        inputs = dict(batch=16, history=131072, resident=16384, index_heads=32, aic_count=24, windows=8)
        with pytest.raises(ValueError):
            RuntimeShape.prepare(lib, **(inputs | changes))


def test_build_has_no_workload_specialization(tmp_path):
    import json
    from types import SimpleNamespace
    from unittest.mock import patch

    from kvip.runtime.build import build_libraries

    def build(command, **kwargs):
        if command[:2] == ["cmake", "--build"]:
            folder = Path(command[2])
            for name in ("libv9_bridge.so",):
                (folder / name).write_bytes(b"universal")
        return SimpleNamespace(returncode=0)

    with patch("kvip.runtime.build.subprocess.run", side_effect=build) as commands:
        args = dict(
            cache_dir=tmp_path / "cache", cann=tmp_path / "cann", headers=tmp_path / "headers", soc="Ascend950PR_9599"
        )
        first = build_libraries(**args)
        second = build_libraries(**args)
    assert first == second and commands.call_count == 2
    options = json.loads((first[0].parent / "complete.json").read_text())
    assert not any(
        x in options
        for x in ("SIM_BATCH", "SIM_HISTORY", "SIM_RESIDENT", "INDEX_HEADS", "INTERMEDIATE_FETCHES", "AIC_COUNT")
    )


def test_build_failure_reports_first_diagnostic(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch

    import pytest
    from kvip.runtime.build import build_libraries

    def fail(command, **kwargs):
        kwargs["stdout"].write("CMake Error: unsupported target\n" + "detail\n" * 80)
        return SimpleNamespace(returncode=1)

    with patch("kvip.runtime.build.subprocess.run", side_effect=fail):
        with pytest.raises(RuntimeError, match="configure failed") as error:
            build_libraries(cache_dir=tmp_path, cann=tmp_path, headers=tmp_path, soc="Ascend950PR_9599")
    assert "CMake Error: unsupported target" in str(error.value)
    assert len(str(error.value).splitlines()) <= 40
    assert not list(tmp_path.glob("*/complete.json"))


def test_install_and_cmake_default_to_concrete_a5_target():
    import ast

    tree = ast.parse((ROOT / "scripts/build.py").read_text())
    soc_option = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--soc"
    )
    default = next(ast.literal_eval(k.value) for k in soc_option.keywords if k.arg == "default")
    assert default == "Ascend950PR_9599"
    cmake = (ROOT / "csrc/operators/lightning_indexer_with_fetch/CMakeLists.txt").read_text()
    assert f"set(SOC_VERSION {default} CACHE" in cmake


def test_completion_diagnostics_distinguish_transport_states():
    import numpy as np
    from diagnostics import completion_failures

    rows = np.zeros((2, 256), dtype=np.uint32)
    posts = np.zeros((2, 4, 32), dtype=np.uint32)
    statuses = np.zeros((2, 32), dtype=np.uint32)
    cq = np.zeros((4, 8), dtype=np.uint64)
    cq[:, 0] = 0x4B5641464247444F
    assert not completion_failures(rows, posts, statuses, cq, cq)
    posts[1, 2, 3] = 1
    result = completion_failures(rows, posts, statuses, cq, cq)
    assert len(result) == 1 and result[0]["request"] == 1
    assert result[0]["posts_count_pending_error_deferred_descriptor"][2][3] == 1
    posts.fill(0)
    failed = cq.copy()
    failed[0, 2] = 7
    result = completion_failures(rows, posts, statuses, cq, failed)
    assert len(result) == 1 and result[0]["final_cq"][0][2] == 7


def test_load_bridge_without_legacy_update_library(tmp_path):
    import subprocess

    from kvip.runtime.loader import load_lifetch_library
    source = tmp_path / "bridge.cpp"
    source.write_text('extern "C" int value(){return 42;}')
    library = tmp_path / "libv9_bridge.so"
    subprocess.run(["c++", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    assert load_lifetch_library(library).value() == 42


def test_context_snapshot_survives_delayed_vector_clear(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = (root / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/output_worker.h').read_text()
    # Compile the actual scalar initialization between scratch aliases and bindings.
    body = source.split('auto slots=local[kFinalSlotsOffset]', 1)[1].split(';', 1)[1].split(' auto u=', 1)[0]
    program = r'''
#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>
std::vector<std::function<void()>> pending;
enum class HardEvent {V_MTE2};
constexpr int EVENT_ID0=0, PIPE_V=0;
void Duplicate(uint32_t* p,uint32_t value,uint32_t count){
 pending.emplace_back([=]{for(uint32_t i=0;i<count;++i)p[i]=value;});
}
template<int> void PipeBarrier() {} // Does not join the independent MTE2 engine.
template<HardEvent> void SetFlag(int) {}
template<HardEvent> void WaitFlag(int) {for(auto& op:pending)op();pending.clear();}
int main(){
 uint32_t memory[2560];for(auto& x:memory)x=0xfeed1234U;
 auto ctl=memory;
 INITIALIZE
 // MTE2 is allowed to complete before an unjoined Vector clear.
 for(unsigned i=128;i<208;++i)memory[i]=0x12340000U+i;
 for(auto& op:pending)op();
 for(unsigned i=128;i<208;++i)assert(memory[i]==0x12340000U+i);
 for(unsigned i=0;i<64;++i)assert(memory[i]==0);
 assert(memory[64]==0xfeed1234U);
}
'''.replace('INITIALIZE', body)
    cpp = tmp_path / 'publisher.cpp'
    cpp.write_text(program)
    exe = tmp_path / 'publisher'
    subprocess.run([shutil.which('clang++') or 'c++', '-std=c++17', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=10)


def test_binding_defaults_and_budget_boundaries():
    for binding in (LIFetchBinding, UbmemBinding):
        params = inspect.signature(binding).parameters
        assert prefetch_budget(params["prefetch_limit"].default, params["window_limit"].default, 8) == (512, 64)
    assert prefetch_budget(0, None, 8) == (0, 1)
    assert prefetch_budget(2048, None, 8) == (2048, 256)
    for total, window, rounds in ((2049, 64, 8), (-1, 64, 8), (512, 481, 8), (512, 0, 8), (512, None, 0)):
        with pytest.raises(ValueError):
            prefetch_budget(total, window, rounds)


def test_device_budget_saturates_without_unsigned_wrap(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ unavailable")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "budget.cpp"
    source.write_text(
        '#include <iostream>\n#define __aicore__\n#include "prefetch_budget.h"\n'
        "int main() { unsigned s,r,sl,pl,tl,w; while(std::cin>>s>>r>>sl>>pl>>tl>>w) "
        'std::cout<<rotating::PrefetchAllowance(s,r,sl,pl,tl,w)<<"\\n"; }\n'
    )
    binary = tmp_path / "budget"
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-I" + str(root / "csrc/operators/common/lifetch"), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    cases = [(0, 0, 8192, 0, 2048, 1), (8192, 0, 8192, 512, 2048, 64), (0, 512, 8192, 512, 2048, 64)]
    rng = random.Random(17)
    cases.extend(tuple(rng.randrange(10000) for _ in range(6)) for _ in range(1000))
    data = "".join(" ".join(map(str, case)) + "\n" for case in cases)
    result = subprocess.run([str(binary)], input=data, text=True, capture_output=True, check=True)
    expected = [min(w, max(0, sl - s), max(0, pl - r), max(0, tl - r)) for s, r, sl, pl, tl, w in cases]
    assert list(map(int, result.stdout.split())) == expected


def test_shared_hot_selector_invalidates_specialization_cache(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    module = repo / "python/kvip/operators/lifetch_ubmem/build.py"
    module.parent.mkdir(parents=True)
    module.touch()
    source = repo / "csrc/operators/lightning_indexer_with_fetch_ubmem"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("# fixture\n")
    shared = source.parent / "lightning_indexer_with_fetch"
    for name in ("op_kernel/indexer/indexer.h", "op_kernel/index_cache_update.h", "include/runtime_shape.h"):
        path = shared / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    selector = source.parent / "lidu/op_kernel/hot_slot_selection.h"
    selector.parent.mkdir(parents=True)
    selector.write_text("// first algorithm\n")
    monkeypatch.setattr(specialization, "__file__", str(module))
    kvaf_root = tmp_path / "kvaf"
    include = kvaf_root / "access_operator/include/ubmem"
    include.mkdir(parents=True)
    for name in ("copy.h", "batch_abi.h", "fused_batch_get_device.h"):
        (include / name).write_text(name)
    monkeypatch.setattr(specialization, "kvaf_source", lambda _: kvaf_root)
    calls = []

    def compile_fixture(command, **kwargs):
        calls.append(command)
        if command[1] == "--build":
            (Path(command[2]) / "libkvip_lifetch_ubmem.so").write_bytes(b"fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(specialization.subprocess, "run", compile_fixture)
    options = dict(
        cache_dir=tmp_path / "cache",
        cann=tmp_path / "cann",
        shape=specialization.UbmemGeometry(2, 33024, 8192, 64, 656, 32, 8, "Ascend950DT_9582"),
    )
    old = specialization.build_ubmem_library(**options)
    assert specialization.build_ubmem_library(**options) == old
    assert len(calls) == 2
    alternate = specialization.UbmemGeometry(16, 131072, 16384, 32, 2624, 32, 8, "Ascend950DT_9582")
    assert specialization.build_ubmem_library(**(options | {"shape": alternate})) == old
    assert len(calls) == 2  # no recompilation for a new runtime geometry or row width
    selector.write_text("// changed hot-slot algorithm\n")
    new = specialization.build_ubmem_library(**options)
    assert new != old and len(calls) == 4
    (include / "fused_batch_get_device.h").write_text("// updated device entry\n")
    updated = specialization.build_ubmem_library(**options)
    assert updated != new and len(calls) == 6
    (shared / "op_kernel/indexer/indexer.h").write_text("// changed shared indexer\n")
    rebuilt = specialization.build_ubmem_library(**options)
    assert rebuilt != updated and len(calls) == 8


def test_service_chunk_assignment_covers_every_request_once(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ unavailable")
    root = Path(__file__).resolve().parents[2]
    header = root / "csrc/operators/lightning_indexer_with_fetch_ubmem/op_kernel/controller"
    source = tmp_path / "chunks.cpp"
    source.write_text(
        "#include <cstdint>\n#include <cassert>\n#include <algorithm>\n"
        "#define SIM_BATCH 32\nnamespace rotating { constexpr uint32_t kServiceLanes=2; }\n"
        '#include "post_geometry.h"\n'
        "int main() { for(uint32_t b=1;b<=32;++b) for(uint32_t lanes=2;lanes<=64;lanes+=2) {"
        "uint32_t groups=(b+lanes-1)/lanes,chunk=std::min(groups,8U),seen[32]={};"
        "auto geometry=rotating::MakePostGeometry(b,lanes);assert(geometry.groups==chunk);"
        "assert(geometry.threads*geometry.groups==512 || geometry.groups==3 || geometry.groups>=5);"
        "assert(geometry.groups*geometry.bank<=480 && geometry.bank%8==0);"
        "assert(geometry.groups*1024<=8448);"
        "for(uint32_t base=0;base<groups;base+=chunk) for(uint32_t lane=0;lane<lanes;++lane){"
        "auto first=rotating::ServiceChunkFirst(lane,base,lanes);"
        "auto count=rotating::ServiceChunkCount(b,first,lanes,chunk);"
        "assert(count<=8);for(uint32_t i=0;i<count;++i){auto row=first+i*lanes;assert(row<b);++seen[row];}}"
        "for(uint32_t i=0;i<b;++i)assert(seen[i]==1); } }\n"
    )
    binary = tmp_path / "chunks"
    subprocess.run([compiler, "-std=c++17", str(source), "-I" + str(header), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("batch,windows,history,resident", [
    (1, 1, 33024, 8192),
    (1, 8, 33024, 8192),
    (8, 8, 131072, 16384),
    (16, 8, 131072, 16384),
    (32, 32, 131072, 16384),
])
def test_ubmem_window_triggers_match_urma(tmp_path, batch, windows, history, resident):
    from kvip.operators.lifetch.geometry import trigger_tokens

    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ unavailable")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "triggers.cpp"
    source.write_text(
        '#include "context.h"\n#include <cstdio>\n'
        'int main() { for(unsigned e=0;e<ROUNDS;++e) '
        'std::printf("%u ",128*rotating::RuntimeTrigger(SIM_HISTORY,SIM_RESIDENT,ROUNDS,e)); '
        'for(unsigned t=0;t<=SIM_HISTORY/128+1;++t) { bool found=false; '
        'for(unsigned e=0;e<ROUNDS;++e)found |= rotating::RuntimeTrigger(SIM_HISTORY,SIM_RESIDENT,ROUNDS,e)==t; '
        'if(found != rotating::IsTrigger(SIM_HISTORY,SIM_RESIDENT,ROUNDS,t))return 1; } }\n'
    )
    definitions = dict(
        SIM_BATCH=batch, SIM_HISTORY=history, SIM_RESIDENT=resident, TOKEN_BYTES=656,
        ROUNDS=min(windows,batch*(2 if 2*batch<=32 else 1)),
        V9_UBPIPE_DEPTH=8, V9_UBPIPE_ROWS=4, V9_DEDICATED_SERVICE=0,
        V9_AIC_COUNT=32, V9_SPLIT_Q=int(2 * batch <= 32),
        V9_INTERMEDIATE_FETCHES=windows, V9_FINAL_REPLICATED=1,
    )
    binary = tmp_path / "triggers"
    subprocess.run(
        [compiler, "-std=c++17", str(source), *[f"-D{k}={v}" for k, v in definitions.items()],
         "-I" + str(root / "csrc/operators/lightning_indexer_with_fetch_ubmem/include"), "-o", str(binary)],
        check=True, capture_output=True,
    )
    actual = subprocess.check_output([str(binary)], text=True)
    assert list(map(int, actual.split())) == trigger_tokens(batch, history, resident, intermediate_fetches=windows)


def test_runtime_trigger_and_epoch_assignment_use_urma_schedule(tmp_path):
    from kvip.operators.lifetch.geometry import li_geometry, trigger_tokens

    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ unavailable")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "runtime_schedule.cpp"
    source.write_text(
        '#include "context.h"\n#include <cstdio>\n'
        'int main(){ unsigned b,h,r,s,n,lanes; '
        'while(std::scanf("%u%u%u%u%u%u",&b,&h,&r,&s,&n,&lanes)==6){'
        'for(unsigned e=0;e<n;++e)std::printf("%u ",128*rotating::RuntimeTrigger(h,r,n,e));'
        'std::printf("\\n"); for(unsigned t=0;t<=h/128;++t){bool found=false;'
        'for(unsigned e=0;e<n;++e)found |= rotating::RuntimeTrigger(h,r,n,e)==t;'
        'if(found!=rotating::IsTrigger(h,r,n,t))return 2;}'
        'for(unsigned c=0;c<b*(1+s);++c){unsigned rank=s?(c%2)*b+c/2:c,e=rank/lanes;'
        'if(e<n && rotating::EpochAt(b,h,r,s,n,lanes,rotating::RuntimeTrigger(h,r,n,e),c)!=int(e))return 3;}'
        '} }\n'
    )
    defs = dict(SIM_BATCH=1, SIM_HISTORY=33024, SIM_RESIDENT=8192, TOKEN_BYTES=656,
                V9_UBPIPE_DEPTH=8, V9_UBPIPE_ROWS=4, V9_DEDICATED_SERVICE=0,
                V9_AIC_COUNT=32, V9_SPLIT_Q=1, V9_INTERMEDIATE_FETCHES=2, V9_FINAL_REPLICATED=1)
    binary = tmp_path / "runtime_schedule"
    subprocess.run([compiler, "-std=c++17", str(source), *[f"-D{k}={v}" for k, v in defs.items()],
                    "-I" + str(root / "csrc/operators/lightning_indexer_with_fetch_ubmem/include"),
                    "-o", str(binary)], check=True, capture_output=True)
    cases = [(1, 33024, 8192, 1), (8, 131072, 16384, 8), (16, 131072, 16384, 8),
             (32, 131072, 16384, 32)]
    rows, expected = [], []
    for batch, history, resident, windows in cases:
        geo = li_geometry(batch, intermediate_fetches=windows)
        rows.append(f"{batch} {history} {resident} {geo['cores_per_batch']-1} "
                    f"{geo['intermediate_fetches']} {geo['service_aics_per_window']}\n")
        expected.append(trigger_tokens(batch, history, resident, intermediate_fetches=windows))
    result = subprocess.run([str(binary)], input="".join(rows), text=True, capture_output=True, check=True)
    assert [list(map(int, row.split())) for row in result.stdout.splitlines()] == expected


def test_query_worker_batches_kvaf_calls_and_preserves_conflicts(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = (root / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/streaming.cpp').read_text()
    begin = source.index('__aicore__ inline void FetchQueryGroup(')
    end = source.index('__aicore__ inline void FinalizeStreaming', begin)
    body = source[begin:end].removesuffix("#endif\n")
    code = r'''
#include <cstdint>
#include <cassert>
#include <vector>
#define __aicore__
#define __gm__
#define V9_SIMULATION 0
constexpr uint32_t kResolveCores=3,kFlagCount=16,kTop=2048,kSlots=8192,kBatchGetInflight=2048;
constexpr uint64_t KVAF_AIV_BATCH_GET_STATUS_MAGIC=123,KVAF_AIV_BATCH_GET_PROFILE_MAGIC=456;
struct BatchFlag {uint32_t ready=0,count=0,error=0,reserved[29]={};};
struct FinalPost {uint32_t count=0,pending=0,error=0,deferred=0,descriptor=0,nonconflicting=0;};
struct KvafBatchGetQueueState {uint32_t x=0;};
struct Profile {uint64_t magic=0,post_active_cycles=0,drain_wait_cycles=0,submitted_group_count=0;};
struct KvafBatchGetCoreStatus {uint64_t magic=123,processed_request_count=0,result=0,completed_batch_count=0; Profile profile;};
struct DeviceArgs {uint64_t indices=0,offsets=0,output_indices=0,miss_count=0,address_table=0,valid_table=0,queue_state=0,core_status=0;
 uint32_t request_capacity=0,table_capacity=0,batch_get_max_inflight_total=0,jettys_per_core=1;};
struct Context {uint64_t sourceBlocks,sourceOffsets,destinations,count,fallbackAddresses=0,fallbackValid=0;};
struct Shape {uint32_t batch=4,finalWorkers=4,finalPostOffset=0,resident=16384;};
struct Args {uint64_t context=0,flags=0,times=0,packedFetch=0;uint32_t prefetchCount=0;};
Shape LoadRuntimeShape(Context*){return {};}
FinalPost posts[16];DeviceArgs deviceArgs[20];std::vector<std::vector<int32_t>> calls,offsetCalls,destCalls;
FinalPost* FinalPostAt(uint32_t,Context*,uint32_t i){return posts+i;}
unsigned FinalQueue(unsigned batch,unsigned workers,unsigned row,unsigned q){return q==0?row*2+1:batch*2+row*(workers-1)+q-1;}
DeviceArgs GetArgs(Context*,unsigned i){return deviceArgs[i];}
void SyncFetchDescriptors(Context*,unsigned,unsigned){}
uint64_t GetSystemCycle(){return 100;}
enum class HardEvent {V_S};template<HardEvent>void SetFlag(int){} template<HardEvent>void WaitFlag(int){} constexpr int EVENT_ID0=0;
void PackFetchRanges(Context* c,int32_t* p,uint32_t stride,uint32_t* ranges,uint32_t rows){
 for(unsigned r=0;r<rows;r++){auto begin=ranges[3*r],count=ranges[3*r+1],out=ranges[3*r+2];
 for(unsigned i=0;i<count;i++){p[out+i]=((int32_t*)c->sourceBlocks)[begin+i];p[stride+out+i]=((int32_t*)c->sourceOffsets)[begin+i];p[2*stride+out+i]=((int32_t*)c->destinations)[begin+i];}}}
void PublishQueryCompletion(Context*,BatchFlag*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t,uint32_t){}
namespace Simt {struct Dim3{int x,y,z;};template<auto f,class... T>void VF_CALL(Dim3,T... a){f(a...);}}
namespace kvaf_device {
constexpr uint64_t kCqTimeout=6,kInvalidArgs=1;
void CleanCacheLines(uint8_t*,uint64_t){}
void WriteStatus(KvafBatchGetCoreStatus* s,uint64_t result,uint64_t processed,uint64_t completed,uint64_t,uint64_t,uint64_t,uint64_t){s->magic=123;s->result=result;s->processed_request_count=processed;s->completed_batch_count=completed;}
void KvafBatchGetDevice(DeviceArgs a,unsigned,unsigned begin,unsigned n){
 assert(a.batch_get_max_inflight_total==2048 && a.jettys_per_core==2);auto status=(KvafBatchGetCoreStatus*)a.core_status;
 calls.emplace_back((int32_t*)a.indices+begin,(int32_t*)a.indices+begin+n);
 offsetCalls.emplace_back((int32_t*)a.offsets+begin,(int32_t*)a.offsets+begin+n);
 destCalls.emplace_back((int32_t*)a.output_indices+begin,(int32_t*)a.output_indices+begin+n);
 status->magic=123;status->result=0;status->processed_request_count=n;status->completed_batch_count=1;
}
}
''' + body + r'''
int main(){
 std::vector<int32_t> src(4*kSlots),off(4*kSlots),dst(4*kSlots),packed(4*2*(9*4*kTop+32+64));
 for(unsigned i=0;i<src.size();i++){src[i]=i;off[i]=i+1;dst[i]=i+2;}
 Context c{uint64_t(src.data()),uint64_t(off.data()),uint64_t(dst.data()),0};
 BatchFlag flags[4*kFlagCount];Args a{uint64_t(&c),uint64_t(flags),0,uint64_t(packed.data()),0};
 KvafBatchGetQueueState queues[20];KvafBatchGetCoreStatus statuses[20];
 for(int i=0;i<20;i++){deviceArgs[i].queue_state=uint64_t(&queues[i]);deviceArgs[i].core_status=uint64_t(&statuses[i]);}
 for(unsigned b=0;b<4;b++){flags[b*kFlagCount+1].ready=1;flags[b*kFlagCount+7].ready=1;
  auto &p=posts[b*4];p.count=2;p.nonconflicting=1;p.descriptor=b*kSlots;}
 FetchQueryGroup(&a,0,0,2,false);
 assert(calls.size()==2);assert((calls[0]==std::vector<int32_t>{0,16384}));
 assert((calls[1]==std::vector<int32_t>{1,16385}));
 assert((offsetCalls[0]==std::vector<int32_t>{1,16385}));
 assert((destCalls[0]==std::vector<int32_t>{2,16386}));
 for(unsigned b: {0U,2U})assert(flags[b*kFlagCount+8].ready && !flags[b*kFlagCount+8].error && !posts[b*4].error);
 assert(!flags[1*kFlagCount+8].ready && !flags[3*kFlagCount+8].ready);
 FetchQueryGroup(&a,0,1,2,false);assert(calls.size()==4);
 assert((calls[2]==std::vector<int32_t>{8192,24576}));
 assert((calls[3]==std::vector<int32_t>{8193,24577}));
 // Direct producer output consumes the same descriptors without invoking packing.
 calls.clear();offsetCalls.clear();destCalls.clear();
 auto direct=packed.data()+8*(3*4*kTop+32);unsigned stride=4*kTop;
 for(unsigned phase=0;phase<2;phase++){
  auto out=direct+phase*3*stride;
  for(unsigned i=0;i<2;i++){unsigned id=i*16384+phase;out[i]=id;out[stride+i]=id+1;out[2*stride+i]=id+2;}
  direct[6*stride+phase*8]=2;
 }
 FetchQueryGroup(&a,0,0,2,true);
 assert(calls.size()==2 && (calls[0]==std::vector<int32_t>{0,16384}) && (calls[1]==std::vector<int32_t>{1,16385}));
 assert((offsetCalls[1]==std::vector<int32_t>{2,16386}));
 assert((destCalls[1]==std::vector<int32_t>{3,16387}));
 // Mismatched publication counters fail closed.
 direct[6*stride]=3;calls.clear();FetchQueryGroup(&a,0,0,2,true);assert(calls.empty());
 // Invalid producer descriptors must fail closed without calling KVAF.
 posts[0].nonconflicting=3;calls.clear();FetchQueryGroup(&a,0,0,2,false);
 assert(calls.empty() && flags[8].error==1 && flags[2*kFlagCount+8].error==1);
 // More workers than requests do not touch a queue or flag.
 calls.clear();FetchQueryGroup(&a,0,4,8);assert(calls.empty());
}
'''
    cpp, exe = tmp_path / 'query.cpp', tmp_path / 'query'
    cpp.write_text(code)
    subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_prefetch_does_not_reset_owned_final_status():
    root = Path(__file__).resolve().parents[2]
    kernel = (root / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/streaming.cpp').read_text()
    helper = (root / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/staged_fetch_worker.h').read_text()
    assert 'CompletePrefetch(ctx,b,true,profile?' in kernel
    assert 'if(initializeSpare)' in helper
    assert 'pair=2*shape.batch+2*(q*workers+lane)' in kernel
    assert 'args.jettys_per_core=2' in kernel
    worker = kernel.split('__aicore__ inline void FetchQueryGroup(', 1)[1].split(
        '__aicore__ inline void FinalizeStreaming', 1)[0]
    assert worker.count('reinterpret_cast<__gm__ uint8_t*>(queue),2*sizeof(*queue)') == 2


def test_two_jetty_pairs_do_not_overlap_prefetch_or_each_other():
    for batch in (1, 2, 4, 8, 16):
        for workers in (1, 2, 4):
            prefetch = {2 * b for b in range(batch)}
            pairs = [2 * batch + 2 * (q * workers + lane)
                     for q in range(4) for lane in range(min(workers, batch))]
            views = [v for pair in pairs for v in (pair, pair + 1)]
            capacity = max(batch * 5, 2 * batch + 8 * workers)
            assert len(views) == len(set(views))
            assert not prefetch.intersection(views)
            assert max(views) < capacity
            assert (capacity + 1) // 2 <= 48


def test_standalone_dispatch_reuses_fused_worker():
    root = Path(__file__).resolve().parents[2]
    kernel = (root / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/streaming.cpp').read_text()
    entry = kernel.split('void v9_query_fetch_only(', 1)[1]
    assert 'FetchQueryGroup(reinterpret_cast<__gm__ Args*>(arg),q,lane,workers,skipEmptyConflict)' in entry
    assert 'uint32_t workers=2' in entry
    assert 'query==4?8U:2U' in entry


def test_native_reference_uses_kvaf_default_partition():
    root = Path(__file__).resolve().parents[2]
    kernel = (root / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/streaming.cpp').read_text()
    entry = kernel.split('void v9_native_fetch_only(', 1)[1]
    assert 'args.block_count=cores;args.jettys_per_core=2' in entry
    assert 'kvaf_device::KvafBatchGetDevice(args,core)' in entry
    assert 'query==4?8U:2U' in entry
    assert entry.count('reinterpret_cast<__gm__ uint8_t*>(queues),2*sizeof(*queues)') == 2
    assert entry.count('reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status)') == 2


def test_classifier_partial_metadata_chunks_and_full_capacity(tmp_path):
    header = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel/staged_topk_classify.h"
    code = (
        r"""
#include <cstdint>
#include <vector>
#include <cassert>
#include <algorithm>
#define __simt_vf__
#define __aicore__
#define __ubuf__
#define __gm__
#define LAUNCH_BOUND(x)
struct { uint32_t x; } threadIdx;
void asc_syncthreads() {}
void asc_atomic_or(uint32_t* p,uint32_t v){*p|=v;}
template<class T> T asc_ldcg(T* p){return *p;}
namespace rotating {
constexpr uint32_t kTop=2048,kLruSlots=2048;
struct Context { uint64_t mapping,lruSlots; };
uint32_t ResolvePhysicalSlot(uint32_t resident,uint32_t* lru,uint32_t slot){
 return uint32_t(slot-resident)<kLruSlots?lru[slot-resident]:slot;
}
}
"""
        + f'#include "{header}"\n'
        + r"""
int main(){using namespace rotating;
 for(uint32_t resident:{2048U,8192U,16384U,33024U,131072U}){
  uint32_t history=resident+32768,words=resident/32;
  std::vector<uint32_t> valid(resident,1),map(history,UINT32_MAX),lru(2048),indices(2048),loc(2048,0),bits(4104),expected(words);
  std::vector<int32_t> reserved(resident,-1);
  for(uint32_t i=0;i<2048;++i)lru[i]=i;
  for(uint32_t i=0;i<resident;i++){valid[i]=(i%7!=0);if(i%11==0)reserved[i]=i%2048;}
  for(uint32_t i=0;i<history;++i)if(i%3==0)map[i]=i%2?i%resident:resident+i%2048;
  for(uint32_t i=0;i<2048;++i){
   indices[i]=(i*997)%history;
   if(i%97==0)indices[i]=UINT32_MAX;
   if(i==2046)indices[i]=resident-1; // final partial metadata chunk
  }
  Context ctx{uint64_t(map.data()),uint64_t(lru.data())};
  uint32_t tailBegin=resident-128,tailEnd=resident+128;
  for(uint32_t base=0;base<resident;base+=8192){
   uint32_t size=std::min(8192U,resident-base);
   for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)
    ClassifyTopkHalf(history,resident,0,512,base,size,tailBegin,tailEnd,true,true,true,&ctx,
      indices.data(),loc.data(),valid.data()+base,reserved.data()+base,bits.data(),bits.data()+4096);
  }
  for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)MergeClassifyBitmap(bits.data(),resident);
  assert(bits[4096]==0);
  for(uint32_t i=0;i<2048;++i){
   uint32_t t=indices[i],want=UINT32_MAX;
   if(t!=UINT32_MAX){
    if(t<resident && !(t>=tailBegin&&t<tailEnd) && valid[t] && reserved[t]<0)want=t;
    else want=ResolvePhysicalSlot(resident,lru.data(),map[t]);
   }
   assert(loc[i]==want);
   if(want<resident)expected[want/32]|=1U<<(want%32);
  }
  for(uint32_t i=0;i<words;++i)assert(bits[i]==expected[i]);
 }
}
"""
    )
    source = tmp_path / "classify.cpp"
    source.write_text(code)
    binary = tmp_path / "classify"
    subprocess.run(["c++", "-std=c++17", "-fsanitize=address,undefined", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_shared_li_dispatch_and_shard_boundaries(tmp_path):
    common = ROOT / 'csrc/operators/common/lifetch'
    source = r'''
#include <cstdint>
#include <cassert>
#include <vector>
namespace rotating {
struct RuntimeShape {uint32_t splitQ=1,indexHeads=32,rounds=8,resident=16384,history=131072,liCores=8;};
#include "sequence_schedule.h"
}
std::vector<int> calls;
int failure=-1;
uint32_t run(int phase,uint32_t cores,void*,void*) {calls.push_back(phase);return phase==failure?99:0;}
uint32_t launch(void* bindings,void* stream,const rotating::RuntimeShape* shape) {
#define LI_LAUNCH_0(c,s,b) run(0,c,s,b)
#define LI_LAUNCH_1(c,s,b) run(1,c,s,b)
#define LI_LAUNCH_2(c,s,b) run(2,c,s,b)
#define LI_LAUNCH_3(c,s,b) run(3,c,s,b)
#include "li_launch.h"
}
int main() {
 rotating::RuntimeShape shape;auto ptr=reinterpret_cast<void*>(1);
 for(auto rounds:{2U,4U,8U}) {
  shape.rounds=rounds;assert(rotating::UseSequenceSplit(shape));
  calls.clear();assert(!launch(ptr,ptr,&shape));assert((calls==std::vector<int>{0,1,2,3}));
  auto half=rotating::SequenceHalfWindow(shape);
  for(uint32_t token=0;token<shape.history;++token) {
   auto part=rotating::SequenceSliceAt(token,shape.resident,half);
   assert(part.end>token && part.end<=shape.history+2*half);
   if(token<shape.resident)assert(part.shard==0 && part.end==shape.resident);
   else assert(part.shard==((token-shape.resident)/half)%2);
  }
 }
 // BS2 uses the same split path, with an aligned truncated final shard.
 shape.rounds=4;assert(rotating::SequenceHalfWindow(shape)==19200);
 std::vector<unsigned> visits(shape.history,0);
 for(unsigned epoch=0;epoch<3;++epoch)for(unsigned shard=0;shard<2;++shard){
  unsigned begin=shape.resident+(2*epoch+shard)*19200;
  unsigned end=begin+19200;if(end>shape.history)end=shape.history;
  assert(begin%256==0 && end%256==0);
  for(unsigned t=begin;t<end;++t){++visits[t];assert(rotating::SequenceSliceAt(t,shape.resident,19200).shard==shard);}
 }
 for(unsigned t=shape.resident;t<shape.history;++t)assert(visits[t]==1);
 shape.rounds=8;
 for(int phase=0;phase<4;++phase) {failure=phase;calls.clear();assert(launch(ptr,ptr,&shape)==99);assert(calls.size()==phase+1);}
 assert(launch(nullptr,ptr,&shape)==100000);
}
'''
    path = tmp_path / 'shared.cpp'
    path.write_text(source)
    binary = tmp_path / 'shared'
    subprocess.run(['c++', '-std=c++17', '-I', str(common), str(path), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_saved_transport_comparison_uses_full_trace(tmp_path, capsys):
    import json

    from diagnostics import summarize_transport_tail
    for transport, filename, stage in (
        ('urma', 'full_attribution.json', 'resolve_update'),
        ('ubmem', 'attribution.json', 'resolve_update_fetch'),
    ):
        root = tmp_path / transport
        case = root / 'case0-b1-prototype-graph'
        case.mkdir(parents=True)
        samples = [dict(round=i, accounting=dict(stages_us={stage: value}))
                   for i, value in enumerate((60, 80, 70))]
        (case / filename).write_text(json.dumps(dict(samples=samples) if transport == 'urma' else samples))
        assert summarize_transport_tail(root, transport) == {1: 70}
    assert 'missing per-core diagnostics' in capsys.readouterr().out


def test_scale8_releases_partial_sequence_shard(tmp_path):
    """Exercise the actual bank-release condition with BS2's 150-tile shard."""
    import re
    header = ROOT / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer_vector.h'
    source = header.read_text()
    condition = re.search(r'if \((\(info.s2Idx - info.s2Start \+ 1\).*?)\) \{', source, re.S)[1]
    cpp = tmp_path / 'scale_bank.cpp'
    cpp.write_text('''
#include <cassert>
namespace QLICommon {constexpr unsigned KScaleTiles=8;}
struct Info {unsigned s2Idx,s2Start,validS2Len;bool isLastS2InnerLoop;};
bool release(Info info) {unsigned s2BaseSize_=128;return ''' + condition + ''';}
int main() {
 // A non-final 19200-token shard has 150 tiles. Its tail must return the bank.
 assert(release({277,128,278*128,false}));
 assert(!release({276,128,278*128,false}));
 assert(release({135,128,278*128,false}));
 // The next shard has a different absolute start and the same partial tail.
 assert(release({427,278,428*128,false}));
}
''')
    exe = tmp_path / 'scale_bank'
    subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_query_copy_completion_and_failure_publication(tmp_path):
    root = Path(__file__).resolve().parents[2]
    header = root / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/staged_ubmem_fetch.h'
    sdk = tmp_path / 'ubmem'
    sdk.mkdir()
    (sdk / 'fused_batch_get_device.h').write_text('')
    source = r'''
#include <cstdint>
#include <cassert>
#include <vector>
#define __aicore__
#define __simt_vf__
#define LAUNCH_BOUND(x)
#define __ubuf__
struct {unsigned x=0;} threadIdx;
struct {unsigned x=1;} blockDim;
void asc_syncthreads(){}
template<class T>T asc_ldcg(T* p){return *p;}
void asc_atomic_or(unsigned* p,unsigned v){*p|=v;}
void asc_atomic_add(unsigned* p,unsigned v){*p+=v;}
#define __gm__
constexpr uint32_t kFlagCount=16,kResolveCores=3,kSlots=8192,kTop=2048;
constexpr uint32_t kUbmemInvalidRegion=1,kUbmemIncomplete=2,kUbmemInvalidAddress=4;
struct BatchFlag {uint32_t ready=0,count=0,error=0,pad[29]={};};
struct FinalPost {uint32_t count=0,pending=0,error=0,deferred=0,descriptor=0,nonconflicting=0;uint64_t head=0;};
struct Shape {uint32_t batch=2,resident=16384,finalPostOffset=0,tokenBytes=656;};
struct Context {uint64_t owners=0,fallbackAddresses=1,destinations=2,rowOwners=0,outputStride=0;};
struct Args {uint64_t context,flags,times=0;uint32_t prefetchCount=0;};
struct Region {uint64_t peerBase=1,mappedBase=2,bytes=3,outputBase=4,outputBytes=5;};
Region region;FinalPost posts[8];BatchFlag* flags;FinalPost* awaited=nullptr;
bool fail=false,invalid=false;uint32_t calls=0,waits=0;uint64_t cycle=0;
Shape LoadRuntimeShape(Context*){return {};}
Region* UbmemRegionAt(Context*){return &region;}
bool UbmemRegionValid(Region*,uint32_t){return !invalid;}
FinalPost* FinalPostAt(uint32_t,Context*,uint32_t i){return posts+i;}
uint64_t GetSystemCycle(){return ++cycle;}
void SyncFetchDescriptors(Context*,uint32_t,uint32_t){}
template<class T>struct LocalTensor {T* p=nullptr;T GetValue(unsigned i){return p[i];} T* GetPhyAddr(){return p;} LocalTensor operator[](unsigned i){return {p+i};}};
struct TPipe {template<class T>void InitBuffer(T& b,unsigned n){b.data.resize(n/4);} void Destroy(){}};
enum class TPosition {VECCALC};
template<TPosition>struct TBuf {std::vector<uint32_t> data;template<class T>LocalTensor<T> Get(){return {reinterpret_cast<T*>(data.data())};}};
enum class HardEvent {V_S};constexpr int EVENT_ID0=0;
template<HardEvent>void SetFlag(int){} template<HardEvent>void WaitFlag(int){}
void PublishQueryCompletion(Context*,BatchFlag* f,uint32_t batch,uint32_t,uint32_t,uint32_t q,
 uint32_t lane,uint32_t workers,uint64_t error,uint32_t){
 for(unsigned b=lane;b<batch;b+=workers){f[b*kFlagCount+8+q].ready=1;f[b*kFlagCount+8+q].error=error;}}
namespace Simt {struct Dim3 {int x,y,z;};template<auto f,class... A>void VF_CALL(Dim3,A... a){f(a...);}}
namespace kvaf_device {
void CleanCacheLines(uint8_t* ptr,uint64_t){
 if(ptr==(uint8_t*)awaited && !awaited->head){++waits;if(waits==3)awaited->head=1;}}
struct UbmemBatchGetDeviceResult{uint32_t scheduled=0,completed=0,error=0;};
struct UbmemBatchGetDeviceArgs{uint64_t* addresses;int32_t* output_indices;
 uint64_t peer_base,mapped_base,region_bytes,output_base,output_bytes;
 uint32_t begin,count,owner_base,resident,worker,workers,token_bytes;};
UbmemBatchGetDeviceResult KvafBatchGetDevice(UbmemBatchGetDeviceArgs a,LocalTensor<uint8_t>){
 if(awaited && !awaited->head)assert(a.begin==kTop);assert(a.workers==1 && a.worker==0);
 assert(a.token_bytes==656 && a.count>0);++calls;
 return {a.count,fail?0:a.count,fail?4U:0U};}
}
''' + f'\n#include "{header}"\n' + r'''
void reset(){calls=waits=0;fail=invalid=false;awaited=nullptr;for(auto &p:posts)p={};}
int main(){
 Context c;c.owners=uint64_t(posts);int32_t dest[kSlots*2]{};c.destinations=uint64_t(dest);BatchFlag f[32];flags=f;Args a{uint64_t(&c),uint64_t(f)};
 for(auto &x:f)x.ready=1;
 reset();posts[1].count=2;posts[1].descriptor=kTop;awaited=posts;
 FetchQueryGroup(&a,1,0,2);
 assert(waits==3 && calls==1 && posts[1].head && posts[1].pending==2 && !f[9].error);
 // Empty Q still releases successors and acknowledges completion.
 reset();posts[2].descriptor=2*kTop;posts[1].head=1;
 FetchQueryGroup(&a,2,0,2);assert(!calls && posts[2].head && f[10].ready);
 // Transport failure is propagated; incomplete data is not marked successful.
 reset();posts[0].count=2;fail=true;FetchQueryGroup(&a,0,0,2);
 assert(calls==1 && posts[0].head && posts[0].pending==0 && f[8].error);
 // Invalid descriptors never reach transport, but wake the next Q.
 reset();posts[0].descriptor=1;FetchQueryGroup(&a,0,0,2);
 assert(!calls && posts[0].head && f[8].error);
 // Disjoint Q2 starts without waiting for Q1.
 reset();a.prefetchCount=kUbmemOverlapFetch;posts[0].count=1;dest[0]=10;
 posts[1].count=2;posts[1].descriptor=kTop;dest[kTop]=20;dest[kTop+1]=22;
 awaited=posts;FetchQueryGroup(&a,1,0,2);assert(calls==1 && waits==0 && posts[1].pending==2);
 // Safe prefix is copied first; only the adjacent-sector suffix waits.
 reset();posts[0].count=1;dest[0]=10;posts[1].count=2;posts[1].descriptor=kTop;
 dest[kTop]=20;dest[kTop+1]=11;awaited=posts;
 FetchQueryGroup(&a,1,0,2);assert(calls==2 && waits==3 && posts[1].pending==2);
 // Empty Q2 must not wait on Q1.
 reset();posts[1].descriptor=kTop;awaited=posts;
 FetchQueryGroup(&a,1,0,2);assert(!calls && !waits && posts[1].head);
 // Q3 must also account for Q1 when Q2 is empty.
 reset();posts[0].count=1;dest[0]=10;posts[1].descriptor=kTop;
 posts[2].count=1;posts[2].descriptor=2*kTop;dest[2*kTop]=11;awaited=posts;
 FetchQueryGroup(&a,2,0,2);assert(calls==1 && waits==3 && posts[2].pending==1);
 // A spare worker must not access a request/region/queue.
 reset();FetchQueryGroup(&a,0,2,4);assert(!calls);
}
'''
    cpp = tmp_path / 'worker.cpp'
    cpp.write_text(source)
    exe = tmp_path / 'worker'
    subprocess.run(['c++', '-std=c++17', '-I', str(tmp_path), str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=5)


def capture(width=656):
    cores = [[10, 1000] + [0] * 30 for _ in range(14)]
    requests = [dict(final_counts=[dict(total=0, safe=0, conflict=0) for _ in range(4)]) for _ in range(2)]
    for b, q, safe, conflict, begin, end in ((0, 0, 2, 0, 100, 200),
                                          (1, 0, 3, 0, 110, 210),
                                          (0, 1, 0, 4, 400, 500)):
        requests[b]['final_counts'][q] = dict(total=safe+conflict, safe=safe, conflict=conflict)
        core = cores[6 + 2*q + b]
        core[5:7] = [begin, end]  # UBmem whole worker interval.
        if conflict:
            core[10:12] = [begin, end]
    raw = dict(batch=2, workers_per_query=2, resolver_cores=3, token_bytes=width,
               clock_mhz=10, cores=cores)
    return raw, requests


@pytest.mark.parametrize('transport', ['urma', 'ubmem'])
@pytest.mark.parametrize('width', [656, 2624])
def test_global_span_and_actual_group_width(transport, width):
    result = final_transfer_scope(*capture(width), transport)
    assert result['entries'] == 9 and result['bytes'] == 9 * width
    assert result['span_us'] == 40
    assert result['GBps'] == pytest.approx(9 * width / 40 / 1000)
    assert sum(group['entries'] for group in result['groups']) == 9
    assert len(result['groups']) == 8  # Empty Q3/Q4 groups retained.
    assert 'not first WQE' in result['note']


def test_nonzero_work_requires_actual_timestamps():
    raw, requests = capture()
    raw['cores'][6][5] = 0
    with pytest.raises(ValueError, match='Get timestamps'):
        final_transfer_scope(raw, requests, 'urma')


def test_empty_tasks_are_na_not_zero_bandwidth():
    raw, requests = capture()
    for request in requests:
        request['final_counts'] = [dict(total=0, safe=0, conflict=0) for _ in range(4)]
    result = final_transfer_scope(raw, requests, 'urma')
    assert result['entries'] == 0 and result['GBps'] is None and result['span_us'] is None


def test_level_one_reports_final_and_combined_scope_without_summing_overlap(tmp_path, monkeypatch):
    # Core timestamp decoding has its own suite; this fixture isolates count/span aggregation.
    monkeypatch.setattr("diagnostics.decode_timeline", lambda raw: {"cores": []})
    raw, details = capture()
    details[0]['core_timeline_raw'] = raw
    details[0]['prefetch_scope'] = dict(bytes=656, first_doorbell=50,
        last_cq_observed=150, interval_cycles=100)
    atomic_json(tmp_path / 'full_attribution.json', dict(samples=[dict(
        round=0, final=9, algorithm=dict(totals={}, requests=[]), streaming_cycles=details,
        accounting=dict(stages_us=dict(resolve_update=100)))]))
    config = SimpleNamespace(iterations=1, level=1, stages=('fetch',), transport='urma')
    result = summarize(tmp_path, config)
    assert result['trace'] == 'COMPLETE'
    sample = result['captures'][0]['samples'][0]
    assert sample['combined_observed_bandwidth']['span_us'] == 45
    assert sample['combined_observed_bandwidth']['bytes'] == 10 * 656
    assert 'metric=final_get_bandwidth' in (tmp_path / 'diagnostic_summary.log').read_text()


def test_invalid_cross_core_order_cannot_produce_global_bandwidth(tmp_path, monkeypatch):
    raw, details = capture()
    details[0]['core_timeline_raw'] = raw
    def invalid(raw):
        raise ValueError('cross-core publication order violated')
    monkeypatch.setattr('diagnostics.decode_timeline', invalid)
    atomic_json(tmp_path / 'full_attribution.json', dict(samples=[dict(
        round=0, final=9, algorithm=dict(totals={}, requests=[]), streaming_cycles=details,
        accounting=dict(stages_us=dict(resolve_update=100)))]))
    config = SimpleNamespace(iterations=1, level=1, stages=('fetch',), transport='urma')
    result = summarize(tmp_path, config)
    assert result['trace'] == 'INCOMPLETE'
    assert 'final_get_bandwidth' not in result['captures'][0]['samples'][0]


def test_ubmem_trace_distinguishes_inline_and_separate_fetch():
    events = [{"name": "replay", "ph": "X", "ts": "1000.0", "dur": 100.0}]
    names = ("init", "li", "resolve", "fetch", "complete")
    events += [{"name": f"kvip_ubmem_656_{name}_kernel_0", "ph": "X", "ts": str(1001 + 10 * i),
                "dur": 5.0} for i, name in enumerate(names)]
    events += [{"name": f"kvip_ubmem_direct_{name}_kernel_0", "ph": "X", "ts": str(1052 + 8 * i),
                "dur": 4.0} for i, name in enumerate(("update_owned", "update_outputs", "apply_owned"))]
    result = account_ubmem(events, "replay", 90)
    assert result["fetch_separate"] and result["stages_us"]["resolve"] == 5
    assert result["stages_us"]["fetch"] == 5 and result["stages_us"]["update"] == 12
    old = account_ubmem([event for event in events if "_fetch_kernel" not in event["name"]], "replay", 90)
    assert not old["fetch_separate"] and old["stages_us"]["fetch"] == 0
    assert "error" in account_ubmem([event for event in events if "_resolve_kernel" not in event["name"]], "replay", 90)


def test_ubmem_trace_recognizes_fused_pipeline_without_phantom_fetch():
    events = [{"name": "replay", "ph": "X", "ts": 1000, "dur": 100}]
    for name, start, duration in (
        ("kvip_ubmem_656_init_kernel", 1001, 5),
        ("kvip_ubmem_656_li_kernel", 1007, 30),
        ("v9_streaming_resolve_update_656", 1038, 40),
    ):
        events.append({"name": name, "ph": "X", "cat": "kernel", "ts": start, "dur": duration})
    result = account_ubmem(events, "replay", 80)
    assert "error" not in result
    assert result["stages_us"]["resolve_update_fetch"] == 40
    assert result["stages_us"]["fetch"] == 0 and not result["fetch_separate"]
    assert result["kernel_span_us"] == 77
    assert "error" in account_ubmem(events[:-1], "replay", 80)


def test_ubmem_profile_finishes_schedule_and_keeps_incomplete_trace(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    import numpy as np
    from diagnostics import collect_ubmem

    class Profile:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.steps = 0

        def __enter__(self):
            return self

        def step(self):
            self.steps += 1
            if self.steps == 2:
                self.kwargs['on_trace_ready'](self)

        def export_chrome_trace(self, path):
            Path(path).write_text(json.dumps({'traceEvents': []}))

        def __exit__(self, *args):
            assert self.steps == 2

    profiles = []

    def profile(**kwargs):
        value = Profile(**kwargs)
        profiles.append(value)
        return value

    def schedule(**kwargs):
        assert kwargs == dict(wait=0, warmup=0, active=2, repeat=1)
        return object()

    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(profiler=SimpleNamespace(
        profile=profile, schedule=schedule, ProfilerActivity=SimpleNamespace(CPU=0, NPU=1))))
    monkeypatch.setitem(sys.modules, 'torch.profiler', SimpleNamespace(record_function=lambda name: nullcontext()))
    def event(**kwargs):
        return SimpleNamespace(record=lambda stream: None, synchronize=lambda: None,
                               elapsed_time=lambda end: 0.4)
    stream = SimpleNamespace(synchronize=lambda: None)
    torch = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda device: stream, Event=event,
                                                synchronize=lambda: None))
    rotating = SimpleNamespace(device=0, rows=SimpleNamespace(cpu=lambda: SimpleNamespace(
        numpy=lambda: np.zeros((1, 256), dtype=np.uint32))), library=object())
    # This fixture tests profiler lifecycle, not device algorithm decoding.
    # Preserve an algorithm capture failure alongside incomplete timestamps.
    monkeypatch.setattr('diagnostics.algorithm_before', lambda binding: {})
    monkeypatch.setattr('diagnostics.algorithm_after',
                        lambda binding, before: {'error': 'fixture counters unavailable'})
    checked = []
    collect_ubmem(torch, rotating, SimpleNamespace(replay=lambda stream: None), lambda: None,
                  lambda: checked.append(True), 2, tmp_path)
    assert len(checked) == 2 and profiles[0].steps == 2
    saved = json.loads((tmp_path / 'attribution.json').read_text())
    assert all('error' in sample['accounting'] for sample in saved)
    assert all(sample['algorithm']['error'] == 'fixture counters unavailable' for sample in saved)
    assert capsys.readouterr().out.count('UBMEM_TRACE_ERROR') == 2


def test_ubmem_trace_counts_sequence_phase_and_dispatch_suffixes():
    events = [dict(name='replay', ph='X', ts=1000, dur=100)]
    for name, start, duration in (
        ('kvip_ubmem_656_init_kernel_0', 1001, 3),
        ('kvip_ubmem_656_li_kernel_0', 1005, 10),
        ('kvip_ubmem_656_li_kernel_1_0', 1016, 5),
        ('kvip_ubmem_656_li_kernel_2_0', 1022, 20),
        ('kvip_ubmem_656_li_kernel_3_0', 1043, 7),
        ('v9_streaming_resolve_update_656_0', 1051, 30),
    ):
        events.append(dict(name=name, ph='X', ts=start, dur=duration, cat='kernel'))
    result = account_ubmem(events, 'replay', 85)
    assert result['stages_us']['li'] == 42
    assert result['inter_kernel_gap_us'] == 5


@pytest.mark.parametrize('source', sorted(SOURCE_ROOT.rglob('*.py')), ids=lambda path: str(path.relative_to(SOURCE_ROOT)))
def test_python_source_compiles(source):
    compile(source.read_bytes(), str(source), 'exec')
