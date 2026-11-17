import numpy as np
import pytest

from kvip.diagnostics.incremental_topk import IncrementalTopK, simulate, verify_rank


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("distribution", ["uniform", "ties", "zero", "ascending", "descending"])
def test_prefix_and_final_ranks(reverse, distribution):
    rng = np.random.default_rng(42)
    n, k = 8193, 128
    values = rng.integers(0, 65536 if distribution == "uniform" else 8, (4, n), dtype=np.uint16)
    if distribution == "zero":
        values.fill(0)
    if distribution in ("ascending", "descending"):
        values[:] = np.arange(n)[::1 if distribution == "ascending" else -1]
    scores = [values[q, :n-q] for q in range(4)]
    seed = [np.r_[np.arange(512), np.arange(n-260, len(s))] for s in scores]
    order = np.arange(512, n)[::-1 if reverse else 1]
    result = simulate(scores, seed, np.array_split(order, 8), np.arange(n) % 3 != 0,
                      k=k, budget=31, window_budget=7)
    assert result["total"] == len(result["selected"]) + result["waste"]
    assert result["useful"] + result["repair"] == len(result["selected"])
    flat = sum(result["publication"], [])
    assert len(flat) == len(set(flat)) <= 31
    assert all(a <= b for a, b in zip(result["thresholds"], result["thresholds"][1:]))


def test_budget_retry_without_new_arrivals_and_prune():
    k = 8
    values = np.arange(24, dtype=np.uint16)
    state = IncrementalTopK(values[:8], np.arange(8), k=k, chunk=3)
    eligible = np.ones(24, dtype=bool)
    first = state.pending(eligible, set())[:2]
    pending = state.pending(eligible, set(first))
    assert len(pending) == 6
    state.append(values[8:], np.arange(8, 24))
    assert not set(pending) & set(state.pending(eligible, set(first)))
    verify_rank(values, state.ids, k)


def test_zero_budget_is_all_repair_not_a_bandwidth_gain():
    values = [np.arange(256, dtype=np.uint16)] * 4
    result = simulate(values, [np.arange(32)] * 4, np.array_split(np.arange(32, 256), 8),
                      np.ones(256, dtype=bool), k=16, budget=0)
    assert result["prefetch"] == result["waste"] == result["useful"] == 0
    assert result["total"] == result["repair"] == 16


def test_native_publication_checker_rejects_omitted_strict_winner():
    from kvip.diagnostics.incremental_topk import check_publications, scan_windows, select
    n, r = 32768, 16384
    scores = np.tile(np.arange(n, dtype=np.uint16), (4, 1))
    ends = np.full(4, n)
    bounds = np.array([n-128, n])
    lists = np.zeros((4, 9, 2080), dtype=np.int32)
    seen = set(range(r)) | set(range(n-128, n))
    for epoch, window in enumerate([None, *scan_windows(n, r, ends, True)]):
        if window is not None:
            seen.update(map(int, window))
        ids = np.array(sorted(seen))
        _, top, kth = select(scores[0, ids], ids)
        lists[:, epoch, :3] = [2048, kth, epoch+1]
        lists[:, epoch, 32:] = top
    eligible = np.ones(n, dtype=bool)
    check_publications(scores, ends, r, bounds, lists, eligible, True)
    lists[0, 1, 32] = lists[0, 1, -1]
    lists[0, 1, 0] -= 1
    with pytest.raises(AssertionError, match="dropped a strict"):
        check_publications(scores, ends, r, bounds, lists, eligible, True)


def test_incremental_report_requires_oracle_and_uses_both_controls(tmp_path, capsys):
    import json
    from kvip.diagnostics import scan_report
    from kvip.diagnostics.scan_options import INCREMENTAL_VARIANTS
    cases = {}
    for name in INCREMENTAL_VARIANTS:
        candidate = name == "reverse-incremental"
        cases[name] = dict(
            geometry=dict(rounds=9, token_bytes=656),
            timing=dict(mean_us=80 if candidate else 100, p50_us=81 if candidate else 99),
            reads_samples=[dict(prefetch=5, unused_prefetch=1, unused_prefetch_bytes=656,
                final=3, bytes=5248, useful_transfer_bytes=4592, cold_demand=6, selected_unique=60)],
            selection_comparison=dict(total_replacements=2 if candidate else 0),
        )
    data = dict(status="pass", execution_order=list(INCREMENTAL_VARIANTS), cases=cases)
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="oracle has not passed"):
        scan_report.report(path, require_incremental=True)
    cases["reverse-incremental"]["incremental_check"] = dict(status="exact_prefix_ranks_and_pending")
    path.write_text(json.dumps(data))
    scan_report.report(path, require_incremental=True)
    out = capsys.readouterr().out
    assert "GATE reverse-incremental CANDIDATE" in out
    assert "vs_reverse_mean/p50=+20.00/+18.18%" in out
    assert "repair_delta_KiB=+0.00 waste_delta_KiB=+0.00 total_delta_KiB=+0.00" in out
    cases["reverse-incremental"]["timing"]["p50_us"] = 100
    path.write_text(json.dumps(data))
    scan_report.report(path, require_incremental=True)
    assert "GATE reverse-incremental KEEP_BASELINE" in capsys.readouterr().out


def test_frozen_archive_cpu_command(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from kvip.diagnostics.archive import TensorArchive
    from kvip.diagnostics.incremental_topk import main

    n, r = 32768, 16384
    manifest = dict(geometry=dict(batch=1, resident=r, heads=32, history=n))
    archive = TensorArchive(tmp_path, manifest)
    archive.save("post.score_keys", np.tile(np.arange(n, dtype=np.uint16), (1, 4, 1)))
    archive.save("pre.query_ends", np.array([[n-3,n-2,n-1,n]], dtype=np.int32))
    archive.save("pre.threshold_bounds", np.array([[n-128,n]], dtype=np.int32))
    archive.save("pre.tail_bounds", np.array([[n-64,n]], dtype=np.int32))
    archive.save("pre.row_owners", np.array([1], dtype=np.int32))
    archive.save("pre.hot_valid", np.ones((2,r),dtype=np.int32))
    archive.save("pre.reserved_hot", np.full((1,r),-1,dtype=np.int32))
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["incremental_topk", "--snapshot", str(tmp_path)])
    main()
    output = capsys.readouterr().out
    assert "CPU_INCREMENTAL forward PASS" in output
    assert "CPU_INCREMENTAL reverse PASS" in output


def test_full_bs16_q4_history_131584_with_eight_cold_windows():
    from kvip.diagnostics.incremental_topk import scan_windows

    rng = np.random.default_rng(17)
    h, r = 131584, 16384
    for row in range(16):
        ends = np.array([130048+row+q for q in range(4)])
        values = rng.integers(30000, 31024, (4,h),dtype=np.uint16)
        values[:, :r] += np.uint16(400)
        scores = [values[q,:end] for q,end in enumerate(ends)]
        seed = [np.r_[np.arange(r),np.arange(129920,len(s))] for s in scores]
        eligible = np.arange(h) >= r
        eligible[130048:h] = False
        result = simulate(scores,seed,scan_windows(h,r,ends,True),eligible)
        assert result["total"] == len(result["selected"]) + result["waste"]
        assert result["prefetch"] <= 512
        assert len(result["publication"]) == 9
