import numpy as np
import pytest

from kvip.diagnostics.incremental_topk import IncrementalTopK, simulate, verify_rank


@pytest.mark.parametrize("window_staging,prefilter", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("distribution", ["uniform", "ties", "zero", "ascending", "descending"])
def test_prefix_and_final_ranks(reverse, distribution, window_staging, prefilter):
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
                      k=k, budget=31, window_budget=7, window_staging=window_staging, prefilter=prefilter)
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


@pytest.mark.parametrize("owner_format", ["packed", "separate", "both"])
@pytest.mark.parametrize("mode", [[], ["--prefilter"], ["--tie-fast"]])
def test_frozen_archive_cpu_command(tmp_path, monkeypatch, capsys, owner_format, mode):
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
    if owner_format != "packed":
        archive.save("pre.row_owners", np.array([1], dtype=np.int32))
    if owner_format != "separate":
        metadata = np.zeros((5,4),dtype=np.int64)
        metadata[4].view(np.int32)[0] = 1
        archive.save("pre.metadata", metadata)
    archive.save("pre.hot_valid", np.ones((2,r),dtype=np.int32))
    archive.save("pre.reserved_hot", np.full((1,r),-1,dtype=np.int32))
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["incremental_topk", "--snapshot", str(tmp_path), *mode])
    main()
    output = capsys.readouterr().out
    assert "CPU_INCREMENTAL forward PASS" in output
    assert "CPU_INCREMENTAL reverse PASS" in output


def test_packed_owner_mapping_uses_int32_view_and_actual_owner(tmp_path):
    from kvip.diagnostics.archive import TensorArchive
    from kvip.diagnostics.incremental_topk import eligibility

    manifest = dict(geometry=dict(batch=2, resident=4, history=8))
    archive = TensorArchive(tmp_path, manifest)
    metadata = np.zeros((5,8),dtype=np.int64)
    metadata[4].view(np.int32)[:2] = [2,1]
    archive.save("pre.metadata", metadata)
    archive.save("pre.hot_valid", np.array([[0,0,0,0], [1,0,1,1], [0,1,1,1]],dtype=np.int32))
    archive.save("pre.reserved_hot", np.array([[-1,-1,0,-1], [-1,-1,-1,-1]],dtype=np.int32))
    archive.save("pre.tail_bounds", np.array([[6,8],[7,8]],dtype=np.int32))
    np.testing.assert_array_equal(eligibility(archive,0), [True,False,True,False,True,True,False,False])
    np.testing.assert_array_equal(eligibility(archive,1), [False,True,False,False,True,True,True,False])
    archive.save("pre.row_owners", np.array([1,2],dtype=np.int32))
    with pytest.raises(ValueError, match="disagrees"):
        eligibility(archive,0)


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


def test_window_staging_keeps_valid_zero_ties_before_rejected_positions():
    from kvip.diagnostics.incremental_topk import WindowTopK
    for positive in (0, 1, 17, 127, 128, 257):
        values = np.zeros(4096, dtype=np.uint16)
        values[1024:1024+positive] = 65535
        state = WindowTopK(values[:256], np.arange(256), k=128)
        state.append(values[256:], np.arange(256, 4096))
        verify_rank(values, state.ids, 128)
        assert np.count_nonzero(state.ids >= 256) == min(positive,128)


def test_window_report_requires_each_incremental_oracle(tmp_path, capsys):
    import json
    from kvip.diagnostics import scan_report
    from kvip.diagnostics.scan_options import INCREMENTAL_WINDOW_VARIANTS, options_for
    cases = {}
    for name in INCREMENTAL_WINDOW_VARIANTS:
        cases[name] = dict(
            geometry=dict(rounds=9, token_bytes=656), timing=dict(mean_us=100,p50_us=99),
            reads_samples=[dict(prefetch=5,unused_prefetch=1,unused_prefetch_bytes=656,
                final=3,bytes=5248,useful_transfer_bytes=4592,cold_demand=6,selected_unique=60)],
            selection_comparison=dict(total_replacements=0, ordered_output_sha256="same-final-ids"))
        if options_for(name)["incremental_topk"]:
            cases[name]["incremental_check"]=dict(status="exact_prefix_ranks_and_pending", prefix_lists_sha256="same-windows")
    data = dict(status="pass", execution_order=list(INCREMENTAL_WINDOW_VARIANTS), cases=cases)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    scan_report.report(path, require_incremental_window=True)
    output = capsys.readouterr().out
    assert "INCREMENT reverse-incremental->reverse-inc-window" in output
    assert "INCREMENT reverse-inc-window->reverse-inc-register" in output
    del cases["reverse-inc-register"]["incremental_check"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="reverse-inc-register.*oracle"):
        scan_report.report(path, require_incremental_window=True)


@pytest.mark.parametrize("count", [0, 1, 127, 128, 2047, 2048, 2049, 16384])
def test_prefilter_exact_ids_thresholds_and_dense_fallback(count):
    from kvip.diagnostics.incremental_topk import PrefilterWindowTopK, WindowTopK
    rng = np.random.default_rng(177)
    seed = np.full(16384, 30000, dtype=np.uint16)
    indices = np.arange(16384)
    sparse = PrefilterWindowTopK(seed, indices)
    dense = WindowTopK(seed, indices)
    for epoch in range(8):
        window = rng.integers(0, sparse.threshold+1, 16384, dtype=np.uint16)
        window[rng.permutation(16384)[:count]] = sparse.threshold+1
        ids = np.arange(16384) + (epoch+1)*16384
        sparse.append(window, ids)
        dense.append(window, ids)
        np.testing.assert_array_equal(sparse.ids, dense.ids)
        np.testing.assert_array_equal(sparse.scores, dense.scores)
        assert sparse.threshold == dense.threshold
    assert sparse.dense == (8 if count > 2048 else 0)
    assert sparse.empty == (8 if count == 0 else 0)


def test_prefilter_q4_lists_and_traffic_identical_to_dense():
    rng = np.random.default_rng(91)
    n = 131584
    scores = [rng.integers(30000, 31024, n-q, dtype=np.uint16) for q in range(4)]
    seed = [np.r_[np.arange(16384), np.arange(130048, len(s))] for s in scores]
    windows = np.array_split(np.arange(16384, n), 8)
    eligible = np.arange(n) % 3 != 0
    a = simulate(scores, seed, windows, eligible, window_staging=True)
    b = simulate(scores, seed, windows, eligible, prefilter=True)
    for key in ("thresholds", "publication", "selected", "prefetch", "useful", "waste", "repair", "total"):
        assert a[key] == b[key]


def test_prefilter_report_requires_both_oracles(tmp_path, capsys):
    import json
    from kvip.diagnostics import scan_report
    from kvip.diagnostics.scan_options import INCREMENTAL_PREFILTER_VARIANTS, options_for
    cases = {}
    for name in INCREMENTAL_PREFILTER_VARIANTS:
        cases[name] = dict(
            geometry=dict(rounds=9, token_bytes=656), timing=dict(mean_us=100,p50_us=99),
            reads_samples=[dict(prefetch=5,unused_prefetch=1,unused_prefetch_bytes=656,
                final=3,bytes=5248,useful_transfer_bytes=4592,cold_demand=6,selected_unique=60)],
            selection_comparison=dict(total_replacements=0, ordered_output_sha256="same-final-ids"))
        if options_for(name)["incremental_topk"]:
            cases[name]["incremental_check"]=dict(status="exact_prefix_ranks_and_pending", prefix_lists_sha256="same-windows")
    data = dict(status="pass", execution_order=list(INCREMENTAL_PREFILTER_VARIANTS), cases=cases)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    scan_report.report(path, require_incremental_prefilter=True)
    output = capsys.readouterr().out
    assert "INCREMENT reverse-inc-register->reverse-inc-prefilter" in output
    assert "CHECK window_prefix_ranks_and_pending=PASS" in output
    assert "ordered_TopK=identical" not in output
    del cases["reverse-inc-register"]["incremental_check"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reverse-inc-register.*oracle"):
        scan_report.report(path, require_incremental_prefilter=True)


@pytest.mark.parametrize("key", ["prefix_lists_sha256", "ordered_output_sha256"])
def test_prefilter_pair_rejects_different_exact_lists_or_final_order(key):
    from copy import deepcopy
    from kvip.diagnostics.replay_measure import verify_prefilter_equivalence
    sample = dict(incremental_check=dict(prefix_lists_sha256="prefix"),
                  selection_comparison=dict(ordered_output_sha256="final"))
    cases = {name:deepcopy(sample) for name in ("reverse-inc-register", "reverse-inc-prefilter")}
    verify_prefilter_equivalence(cases)
    section = "incremental_check" if key == "prefix_lists_sha256" else "selection_comparison"
    cases["reverse-inc-prefilter"][section][key] = "different"
    with pytest.raises(AssertionError, match=key):
        verify_prefilter_equivalence(cases)


@pytest.mark.parametrize("count", [0, 1, 31, 127, 128, 129, 2047, 2048, 2049, 16384])
def test_exact_tie_replacement_preserves_dense_selection_order(count):
    from kvip.diagnostics.incremental_topk import TieWindowTopK, WindowTopK
    rng = np.random.default_rng(178)
    seed = np.full(16384,30000,dtype=np.uint16)
    seed[:1920] += np.uint16(10)
    tie = TieWindowTopK(seed,np.arange(16384))
    full = WindowTopK(seed,np.arange(16384))
    fast = 0
    for epoch in range(12):
        scores = rng.integers(0,tie.threshold+1,16384,dtype=np.uint16)
        scores[rng.permutation(16384)[:count]] = tie.threshold+1
        ids = np.arange(16384) + 16384*(epoch+1)
        old_ties = np.count_nonzero(tie.scores == tie.threshold)
        fast += 0 < count < old_ties
        tie.append(scores,ids); full.append(scores,ids)
        np.testing.assert_array_equal(tie.ids,full.ids)
        np.testing.assert_array_equal(tie.scores,full.scores)
        assert tie.threshold == full.threshold
    assert tie.fast_ties == fast


@pytest.mark.parametrize("light", [False, True])
def test_state_report_checks_both_new_variants(tmp_path,capsys,light):
    import json
    from kvip.diagnostics import scan_report
    from kvip.diagnostics.scan_options import INCREMENTAL_STATE_VARIANTS,INCREMENTAL_LIGHT_VARIANTS,options_for
    variants = INCREMENTAL_LIGHT_VARIANTS if light else INCREMENTAL_STATE_VARIANTS
    required = {"require_incremental_light" if light else "require_incremental_state": True}
    candidate = "reverse-inc-light-ties" if light else "reverse-inc-ties"
    pair = "reverse-inc-light->reverse-inc-light-ties" if light else "reverse-inc-direct->reverse-inc-ties"
    cases = {}
    for name in variants:
        cases[name] = dict(
            geometry=dict(rounds=9,token_bytes=656),timing=dict(mean_us=100,p50_us=99),
            reads_samples=[dict(prefetch=5,unused_prefetch=1,unused_prefetch_bytes=656,
                final=3,bytes=5248,useful_transfer_bytes=4592,cold_demand=6,selected_unique=60)],
            selection_comparison=dict(total_replacements=0,ordered_output_sha256="same-final"))
        if options_for(name)["incremental_topk"]:
            cases[name]["incremental_check"]=dict(status="exact_prefix_ranks_and_pending",prefix_lists_sha256="same-prefix")
    path=tmp_path/"result.json"
    path.write_text(json.dumps(dict(status="pass",execution_order=list(variants),cases=cases)))
    scan_report.report(path,**required)
    output=capsys.readouterr().out
    assert "INCREMENT " + pair in output
    assert "window_lists_and_final_order=IDENTICAL" in output
    cases[candidate]["incremental_check"]["prefix_lists_sha256"]="different"
    path.write_text(json.dumps(dict(status="pass",execution_order=list(variants),cases=cases)))
    with pytest.raises(AssertionError,match="prefix_lists_sha256"):
        scan_report.report(path,**required)


def test_light_variants_require_direct_and_preserve_controls():
    from kvip.diagnostics.scan_options import INCREMENTAL_LIGHT_VARIANTS, options_for, validate_options
    assert INCREMENTAL_LIGHT_VARIANTS[:3] == ("forward", "reverse", "reverse-inc-register")
    for name in INCREMENTAL_LIGHT_VARIANTS:
        options = options_for(name)
        validate_options(options)
        assert options["incremental_light"] == name.startswith("reverse-inc-light")
    options = options_for("reverse-inc-light")
    options["incremental_direct"] = False
    with pytest.raises(ValueError, match="Light"):
        validate_options(options)
