# SPDX-License-Identifier: Apache-2.0
"""CPU checks for fixed-list control; times from a fallback must never count."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from four_rank_controlled import (controlled_build, evidence, extend_state, list_digest,
                                  validate_physical_operands, window_inputs)


def original():
    rows = {f"r0_{tag}{i}": np.empty((0, 5), dtype="<i4") for tag in ("w", "q") for i in range(4)}
    rows["r0_w0"] = np.array([[0, 3, 9, 1968, 2], [1, 131, 8, 1968, 1]], dtype="<i4")
    rows["r0_q3"] = np.array([[3, 129, 8, 656, 0]], dtype="<i4")
    return rows


def test_members_verified_before_addresses_and_report_matches_each_segment():
    ref = original()
    ids, counts = window_inputs(ref, 1, 4)
    observed = ids.copy()
    observed[0, 0, :2] = [1, 0]
    status = np.array([[1, 0, 0, 0]])
    info = evidence(ref, ref, counts, observed, status)
    assert info["ordered_match"] and info["natural_window_order_mismatches"] == 1
    assert info["actual_window_sha256"] == list_digest(dict(reversed(list(ref.items()))), "w")
    after = {k: v.copy() for k, v in ref.items()}
    after["r0_q3"][0, 3] += 656
    bad = evidence(ref, after, counts, observed, status)
    assert not bad["ordered_match"] and bad["ordered_tail_mismatches"] == 1
    assert bad["actual_final_sha256"] != info["actual_final_sha256"]
    assert bad["first_mismatch"]["segment"] == "r0_q3"
    assert not evidence(ref, ref, counts, observed, np.array([[2,0,0,0]]))["ordered_match"]
    observed[0, 0, 0] = 3
    assert not evidence(ref, ref, counts, observed, status)["ordered_match"]
    cap = original()
    cap["r0_w1"] = np.array([[x,0,0,0,0] for x in range(257)], dtype="<i4")
    with pytest.raises(ValueError, match="count"):
        window_inputs(cap, 1, 1000)
    cap = original()
    cap["r0_w0"][0,0] = 1
    with pytest.raises(ValueError, match="duplicate"):
        window_inputs(cap, 1, 4)


def test_verify_physical_source_and_destination_of_each_record():
    ref = original()
    inp = SimpleNamespace(geometry=dict(history=4, resident=4, token_bytes=656),
                          logical_address_valid=np.ones((1,256)), captured_layout=np.array([[3,131,7,129]]))
    pages = SimpleNamespace(page_indices=np.array([[9,8]]))
    validate_physical_operands(ref, inp, pages)
    ref["r0_q3"][0, 2] = 9
    with pytest.raises(ValueError, match="r0_q3"):
        validate_physical_operands(ref, inp, pages)


def test_cold_overlay_preserves_native_sources_and_is_scoped(tmp_path):
    op = tmp_path / "csrc/operators/lightning_indexer_with_fetch"
    include = op / "include"
    include.mkdir(parents=True)
    shared = op.parent / "common/lifetch/sequence_schedule.h"
    shared.parent.mkdir(parents=True)
    shared.write_text("schedule")
    kernel = op / "op_kernel/prefetch.h"
    kernel.parent.mkdir()
    anchor = "                DataCopy(jobs,ids,((count+7)/8)*8);"
    kernel.write_text(anchor)
    source = tmp_path / "tests/operators/lifetch/support/large_two_cold_native"
    source.mkdir(parents=True)
    def build(overlay, lib, sdk, soc, policy):
        assert (overlay.parents[4]/shared.relative_to(tmp_path)).read_text() == "schedule"
        assert kernel.read_text() == anchor and overlay.parent != source.parent
        replaced = (overlay.parents[4]/kernel.relative_to(tmp_path)).read_text()
        assert "useFrozen ? 1U : 2U" in replaced
        assert "count == expectedCount && count == candidateCount" in replaced
        assert (overlay / "ordered_guard.inc").read_bytes().strip() == replaced.encode().strip()
        return SimpleNamespace()
    launch = controlled_build(build)(source, None, None, None, "four-ranks")
    assert len(launch.frozen_guard_sha256) == 64
    assert kernel.read_text() == anchor
    with pytest.raises(ValueError, match="only valid"):
        controlled_build(build)(source, None, None, None, "two-cold")


def test_pointer_trailer_routes_only_the_live_forward_context():
    torch = pytest.importorskip("torch")
    ctx = torch.arange(352, dtype=torch.int64).to(torch.uint8)
    words = torch.zeros(16, dtype=torch.int64)
    words[14] = ctx.data_ptr()
    work = torch.zeros(32, dtype=torch.int64)
    work[0] = ctx.data_ptr()
    calls = []
    f = SimpleNamespace(_context=ctx, bindings=words, workspace=work.view(torch.uint8),
                        terminal=object(), set_diagnostics=calls.append)
    s = SimpleNamespace(torch=torch, device="cpu", forward=f, history=131328, batch=20,
                        rotating=SimpleNamespace(runtime_shape=SimpleNamespace(rounds=4)),
                        large_window_launch=SimpleNamespace(frozen_guard_sha256="a"))
    extend_state(s, np.full((20,4,256),-1, dtype="<i4"), np.zeros((20,4), dtype="<i4"))
    assert f._context.numel() == ctx.numel() + 32
    assert torch.equal(f._context[:352], ctx)
    assert (f._context[352:].view(torch.int64).tolist() ==
            [s.frozen_binding[i].data_ptr() for i in (1,2,3,4)])
    assert int(words[14]) == int(work[0]) == f._context.data_ptr()
    assert ctx.data_ptr() != f._context.data_ptr()
    f.set_diagnostics(False)
    assert calls == [True, True]


def test_device_address_table_matches_logical_address_and_row_owner():
    torch = pytest.importorskip("torch")
    from four_rank_controlled import validate_bound_addresses
    records = original()
    addresses = torch.tensor([[0, 0, 0, 0], [1001, 1002, 1003, 1004]], dtype=torch.int64)
    logical = np.zeros((1, 256), dtype=np.uint64)
    logical[0, [3, 131, 129]] = [1001, 1002, 1004]
    ctx = torch.zeros(44, dtype=torch.int64)
    owner = torch.tensor([1], dtype=torch.int32)
    ctx[23], ctx[24], ctx[25] = owner.data_ptr(), addresses.data_ptr(), addresses.stride(0)
    state = SimpleNamespace(torch=torch, device="cpu", batch=1, logical_addresses=logical,
                            storage=SimpleNamespace(addresses=addresses),
                            forward=SimpleNamespace(_context=ctx.view(torch.uint8)),
                            workspace=SimpleNamespace(inputs=SimpleNamespace(row_owners=owner)))
    validate_bound_addresses(state, records)
    addresses[1,0] = 999
    with pytest.raises(ValueError, match="different registered memory"):
        validate_bound_addresses(state, records)


def test_report_only_reports_savings_when_both_complete_graphs_match(tmp_path, monkeypatch):
    import json
    import four_rank_accept as a

    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    monkeypatch.setattr(a, "CELLS", [(10, 32)])
    monkeypatch.setattr(a, "cached", lambda root, ref, group, batch, mode: ref)
    good = {"ordered_match": True, "actual_window_sha256": "w", "actual_final_sha256": "q"}
    output = dict(frozen_journal_sha256="j", expected_window_sha256="w", expected_final_sha256="q",
                  tail_text="t", compiler_sha256="c", guard_sha256="g", timed_journal_checks=[good]*10,
                  mean_us={"LI": 82.2, "tail": 15.6}, graph_mean_us=100.0)
    (old/"frozen.json").write_text(json.dumps(output))
    output.update(graph_mean_us=90.0, mean_us={"LI": 72.2, "tail": 15.6})
    (new/"frozen.json").write_text(json.dumps(output))
    assert a.frozen_report(new,old)
    report=json.loads((new/"controlled_acceptance.json").read_text())
    assert report["cells"][0]["controlled_savings_us"]["graph"] == 10.0
    output["timed_journal_checks"][9] = dict(good, actual_window_sha256="different")
    (new/"frozen.json").write_text(json.dumps(output))
    assert not a.frozen_report(new,old)
    report=json.loads((new/"controlled_acceptance.json").read_text())
    assert report["cells"][0]["verified_graph_calls"] == [10, 9]
    assert report["cells"][0]["controlled_savings_us"] is None
