# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

from kvip.diagnostics.consumer_ab import FIXED_FUSED_REVISION, command
from kvip.diagnostics.high_graph_account import print_account
from kvip.diagnostics.replay import arguments
from kvip.diagnostics.replay_measure import verified_high_trace
from test_graph_accounting import task


def test_independent_g7_profile_and_same_event_budget(tmp_path, capsys):
    opts = SimpleNamespace(cache=tmp_path, source=tmp_path/"candidate", revision="a"*40, baseline="b"*40,
                           intermediate=None, high_overlap=True, result_dir=tmp_path, server="127.0.0.1:28999")
    events = [task("KERNEL_AIVEC", t, us, name=name) for t, us, name in
              ((10, 5, "v9_init_kernel_0"), (16, 200, "v9_li_kernel_1"), (219, 95, "v9_streaming_resolve_update_4"))]
    (tmp_path/"trace_view.json").write_text(json.dumps(events))
    for arm, sha in (("original", FIXED_FUSED_REVISION), ("after", opts.revision)):
        a = arguments(command(opts, 7, arm, profile=True)[4:])
        assert (a.upstream_revision, a.iterations, a.diagnostic_replays, a.calibrate_counter, a.profile_metrics,
                a.profile_graph_event, a.output, a.high_post_index) == (
            sha, 2, 0, False, ["none"], True, tmp_path/f"g7-{arm}-level0.json", arm == "after")
        auto = SimpleNamespace(record=lambda reads: dict(mode="high", steps=[3], learned_owners=[2]),
                               requests=[[2,9,10,1]])
        counts = dict(prefetch=111, final=100, post_index=dict(active=False), consumer_work=dict(
            ordered_index_verified=True,event_sha256="e",descriptor_sha256="d"))
        if arm == "after":
            counts["high_post_index"] = dict(wait_first=False, fallback_cores=0, core_count=64, final_get_by_core=[100]+[0]*63)
        route = verified_high_trace(auto, counts, 3, arm == "after")
        c = dict(kvip_commit=sha,library_sha256=sha,libraries=[sha],snapshot_sha256="s",geometry="g",
                 selection_comparison="order",graph_backend="kvaf.AclGraph",tuning="h",high_post_index=a.high_post_index,
                 device=0,kvaf_build="k",soc="soc",diagnostics_commit="diag")
        exp = dict(auto_route=route, snapshot_sha256="s", production_source=dict(commit=sha), libraries=[sha])
        profile = dict(correctness="passed",graph_replays=1,included_in_headline_timing=False,metric="none",
                       graph_event_us=310,experiment=exp,directory=str(tmp_path))
        print_account(arm, dict(c,system_profiles={"passes":[profile]}), c)
        text=capsys.readouterr().out
        assert "span=304.000 init/LI/streaming=5.000/200.000/95.000" in text
        assert "gap/overlap=4.000/0.000 event_minus_span=6.000" in text
        counts["consumer_work"]["ordered_index_verified"] = False
        with pytest.raises(ValueError, match="oracle|persistent Index"):
            verified_high_trace(auto, counts, 3, arm == "after")
