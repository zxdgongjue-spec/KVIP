# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from kvip.diagnostics.consumer_ab import ARMS, command, diagnostic_wait, summarize
from kvip.diagnostics.replay import arguments


def test_real_update_and_low_route_are_enabled_in_every_arm(tmp_path):
    args = SimpleNamespace(baseline="a"*40, revision="b"*40, cache=tmp_path, source=tmp_path / "after",
                           result_dir=tmp_path, server="127.0.0.1:29084")
    for role in ARMS:
        a = arguments(command(args, 10, role)[4:])
        assert (a.iterations, a.warmup, a.check_auto_route, a.pipeline_variants) == (10, 2, True, ["auto-post-index"])
        assert a.verify_consumer_work and not a.high_post_index and not a.post_index_wait_first and not a.profile_dir
        assert a.calibrate_counter == (role in ("before", "after"))
        assert a.diagnostic_replays == 2 * a.calibrate_counter
        assert a.upstream_revision == (args.baseline if role.startswith("before") else args.revision)
        assert a.output == tmp_path / f"g10-{role}.json"
        assert a.snapshot == tmp_path / "skip-results.po80Nw/g10-input"


def synthetic_report(label, value, *, exact=True):
    rev = "a"*40 if label.startswith("before") else "b"*40
    new = label.startswith("after")
    identity = "X" if exact or not new else "Y"
    evidence = dict(event_sha256=identity, descriptor_sha256="D", event_count=100)
    f = [0]*128
    f[104:106] = [500, 4500]
    f[4] = 4500
    f[100] = 5900
    f2 = f.copy()
    f2[105] = 6000
    probe = dict(auto_route=dict(mode="high"),
                 metrics=dict(index_apply_total=dict(status="available", max_cycles=20500)),
                 raw=dict(fetch=[f]*15+[f2], workers_per_query=4, resolver_cores=3, cores=[[0, 6700]]*64))
    return dict(verified_consumer_work=dict(**evidence, ordered_index_verified=True),
        reads_samples=[dict(consumer_work=evidence, prefetch=99, final=1, unused_prefetch=0)]*10,
        _samples=[value+500]+[value]*9, library_sha256=rev, kvip_commit=rev,
        geometry=dict(token_bytes=656, token_count=64, batch=16, requested_windows=8),
        snapshot_sha256="snapshot", soc="Ascend950DT_9582", device=0, kvaf_build="build", diagnostics_commit="diag",
        graph_backend="kvaf.AclGraph", warmup=2, tuning=dict(low_miss_post_index=True, auto_route=True),
        high_post_index=False, post_index_wait_first=False, auto_route=dict(samples=[dict(mode="high")]*10),
        selection_comparison=dict(selected_set_sha256="s", ordered_output_sha256="o"),
        native_diagnostics=dict(samples=[probe, probe], included_in_timing=False, counter_calibration=dict(
            status="measured", included_in_timing=False, samples=[dict(cycles=20000000, event_us=20000)]*3)))


@pytest.mark.parametrize("exact,expected", [(True, "identical_work_pairs=18/18"), (False, "identical_work_pairs=0/18")])
def test_distinguish_drift_raw_steady_and_identical_work_pairs(exact, expected, capsys):
    cases = {k: synthetic_report(k, x, exact=exact) for k,x in zip(ARMS, (500, 490, 480, 490))}
    summarize(7, cases, "a"*40, "b"*40)
    output = capsys.readouterr().out
    assert "saved_min=0.000/0.000 saved_max=20.000/20.000" in output
    assert "drift_old=-10.000/-10.000 drift_new=-10.000/-10.000" in output
    assert expected in output
    assert ("paired_saved_us_mean/p50=10.000/10.000" if exact else "paired_saved_us_mean/p50=unavailable") in output
    assert "before_max_row_index_wait=5.500/5.500" in output
    assert "before_last_index_minus_last_get_or_prefetch_join=0.800/0.800" in output


def test_checks_fail_for_stale_or_unverified_index_or_changed_transport():
    cases = {k: synthetic_report(k, 600) for k in ARMS}
    cases["after"]["verified_consumer_work"]["descriptor_sha256"] = "stale"
    with pytest.raises(ValueError, match="persistent Index update"):
        summarize(7, cases, "a"*40, "b"*40)
    case = synthetic_report("after", 600)
    case["native_diagnostics"]["samples"][0]["raw"]["fetch"][0][104] = 0
    with pytest.raises(ValueError, match="timestamps"):
        diagnostic_wait(case)


def test_high_overlap_arms_reuse_public_completion_records(tmp_path, capsys):
    from kvip.diagnostics.consumer_ab import OVERLAP_ARMS

    args = SimpleNamespace(baseline="a"*40, revision="b"*40, cache=tmp_path, source=tmp_path/"new",
                           result_dir=tmp_path, server="127.0.0.1:29085", high_overlap=True)
    cases = {}
    for arm, usecs in zip(OVERLAP_ARMS, (500, 540, 501, 499, 542, 498)):
        opts = arguments(command(args, 7, arm)[4:])
        original = arm.startswith("original")
        old_binary = original or arm.startswith("before")
        assert opts.verify_consumer_work and opts.check_auto_route and opts.high_post_index == (not original)
        assert not opts.calibrate_counter and not opts.diagnostic_replays and not opts.post_index_wait_first
        assert opts.upstream_revision == (args.baseline if old_binary else args.revision)
        c = synthetic_report("before" if old_binary else "after", usecs)
        del c["native_diagnostics"]
        c["high_post_index"] = not original
        c["verified_consumer_work"]["index_positions_changed"] = 1868
        if not original:
            for s in c["reads_samples"]:
                s["high_post_index"] = dict(core_count=64, fallback_cores=0, final_get_by_core=[1]+[0]*63)
        cases[arm] = c
    summarize(7, cases, args.baseline, args.revision, True)
    result = capsys.readouterr().out
    assert "ORIGINAL_per_row_vs_candidate_us saved_min=-3.000/-3.000 saved_max=1.000/1.000" in result
    assert "identical_work_pairs=18/18" in result and "real_Index_order/TopK/KV/lifecycle/submitted_completed=PASS" in result
    cases["after"]["reads_samples"][1]["high_post_index"]["final_get_by_core"] = [0]*64
    with pytest.raises(ValueError, match="transport count"):
        summarize(7, cases, args.baseline, args.revision, True)
