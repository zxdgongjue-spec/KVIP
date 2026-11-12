"""CPU contracts for physical budgets, publication layout and exact FP8 cases."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from kvip.operators.lifetch.schedule import StreamingSchedule

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("batch", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("aic,aiv", ((32, 64), (28, 56)))
def test_independent_budget_and_publication_layout(batch, aic, aiv):
    li = batch * (2 if 2 * batch <= aic else 1)
    plan = StreamingSchedule(batch, aic, aiv, li)
    assert plan.tail_aiv == 3 * batch + 8
    assert plan.urma_queues == 2 * batch + 16
    from kvip.operators.lifetch.geometry import final_aivs

    assert final_aivs(batch, aic_count=aic) == 4
    size = batch * 32 + batch * 4 * 320
    offset = batch * 128 + batch * 4 * 288 * 4
    assert offset + batch * 4 * 128 == size * 4


def test_cpp_budget_and_shape_match_python(tmp_path):
    source = tmp_path / "budget.cpp"
    source.write_text(
        """#include <cassert>
#include "csrc/operators/lightning_indexer_with_fetch/include/runtime_shape.h"
#include "csrc/operators/common/lifetch/independent_schedule.h"
int main(){
 for(unsigned c: {28U,32U})for(unsigned b: {1U,2U,4U,8U,16U}){
  rotating::RuntimeShape s{};s.batch=b;s.history=131072;s.resident=16384;
  s.indexHeads=32;s.aicCount=c;s.splitQ=1;s.rounds=8;s.tokenBytes=656;
  assert(rotating::PrepareShape(&s)==0 && s.finalWorkers==4);
  assert(s.finalPostOffset+b*4*128==s.finalOwnerBytes);
  assert(rotating::ValidateStreamingBudget(b,c,c==28?56:64,s.liCores,2)==0);
 }
 assert(rotating::ValidateStreamingBudget(16,28,56,16,4)!=0);
 assert(rotating::ValidateStreamingBudget(16,28,64,16,4)==0);
 assert(rotating::ValidateStreamingBudget(1,8,32,2,4)==0);
 assert(rotating::ValidateStreamingBudget(1,32,8,32,1)!=0);
}
""".replace("#include <cassert>", "#include <cassert>\n#include <initializer_list>")
    )
    binary = tmp_path / "budget"
    subprocess.run(["c++", "-std=c++17", "-I", str(ROOT), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_case_groups_and_unrun_status(tmp_path):
    spec = importlib.util.spec_from_file_location("split_cases", ROOT / "scripts/run_split_core_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert [len(list(module.cases(g))) for g in ("small", "medium", "large", "all")] == [24, 24, 12, 60]
    assert module.main(["plan", "--output", str(tmp_path / "plan")]) == 0
    rows = json.loads((tmp_path / "plan/results.json").read_text())
    assert len({r["case"] for r in rows}) == 60
    assert all(r["status"] == "NOT_RUN" and r["metric"] is None for r in rows)


@pytest.mark.parametrize("heads", (32, 64))
@pytest.mark.parametrize("valid,shape", ((39322, 39424), (65536, 65536), (131072, 131072)))
@pytest.mark.parametrize("miss", (3, 5, 10, 15))
def test_exact_real_fp8_fixture(tmp_path, heads, valid, shape, miss):
    sys.path.insert(0, str(ROOT / "tests/operators/lifetch/support"))
    from real_fetch_inputs import prepare_full_chain_fixture

    root = tmp_path / "fixture"
    info = prepare_full_chain_fixture(root, shape, 16384, miss, index_heads=heads, exact_miss=True, valid_history=valid)
    keys = np.fromfile(root / "arg1.bin", np.uint8).reshape(shape, 128)
    scale = np.fromfile(root / "arg4.bin", np.float32)
    weights = np.fromfile(root / "arg2.bin", np.float32).reshape(4, heads)
    qs = np.fromfile(root / "arg3.bin", np.float32).reshape(4, heads)
    order = np.load(root / "physical_to_logical.npy")
    stored = np.load(root / "logical_scores.npy")[:, order]
    cold = set()
    for q in range(4):
        codes = keys[:, np.arange(heads) % 32 + q * 32]
        exponent = (codes >> 3).astype(np.int32)
        mantissa = (codes & 7).astype(np.float32)
        values = np.where(exponent == 0, mantissa * 2.0**-9, (1 + mantissa / 8) * np.exp2(exponent - 7)).astype(
            np.float32
        )
        raw = (values * (weights[q] * qs[q])).sum(-1) * scale
        bits = ((raw.view(np.uint32) + 0x8000) >> 16).astype(np.uint16)
        assert np.array_equal(bits, stored[q])
        threshold = np.partition(bits[:valid], -2048)[-2048]
        ids = np.flatnonzero(bits >= threshold)
        assert len(ids) == 2048 and (ids < valid).all()
        assert np.count_nonzero(ids >= 16384) == round(2048 * miss / 100)
        cold.update(ids[ids >= 16384].tolist())
    assert len(cold) == info["unique_misses_per_request"] == 2 * round(2048 * miss / 100)
    assert (stored[:, valid:] == 0).all()
    assert (order[:valid] < valid).all() and np.array_equal(order[valid:], np.arange(valid, shape))


@pytest.mark.parametrize("miss", (3, 5, 10, 15))
def test_partial_padding_without_exact_mode(tmp_path, miss):
    sys.path.insert(0, str(ROOT / "tests/operators/lifetch/support"))
    from real_fetch_inputs import prepare_full_chain_fixture

    root = tmp_path / "partial"
    prepare_full_chain_fixture(root, 39424, 16384, miss, index_heads=32, valid_history=39322)
    bits = np.load(root / "logical_scores.npy")[:, np.load(root / "physical_to_logical.npy")]
    assert (bits[:, 39322:] == 0).all()
    for row in bits:
        cutoff = np.partition(row[:39322], -2048)[-2048]
        winners = np.flatnonzero(row >= cutoff)
        assert len(winners) == 2048 and (winners < 39322).all()
