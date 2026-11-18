# SPDX-License-Identifier: Apache-2.0
import json
from decimal import Decimal

import pytest

from kvip.diagnostics.graph_accounting import summarize, timeline
from test_post_index_plan_report import inputs


def task(kind, ts, dur, *, name=None, model=63, tid=13):
    start = str(Decimal("1794604239692877.760") + Decimal(str(ts)))
    return dict(
        name=name or kind, ph="X", pid=4, tid=tid, ts=start, dur=dur, args={"Task Type": kind, "Model Id": model}
    )


def trace(kernel):
    return [
        task("EVENT RECORD", 0, 0.028, model=0),
        task("MODEL_EXECUTE", 15, 3.2),
        task("KERNEL_MIX_AIC", 18, 10, name=kernel, tid=14),
        task("KERNEL_AIVEC", 28.25, 30, name="tail", tid=14),
        task("MODEL_WAIT_COMPLETE", 17, 43),
        task("EVENT RECORD", 61, 0.028, model=0),
    ]


def test_account_including_graph_dispatch_and_joins_on_overlapping_streams():
    row, kernels = timeline(trace("LI"), elapsed=62)
    assert row == dict(
        profile_event=62,
        kernel_span=40.25,
        kernel_gap=0.25,
        kernel_overlap=0,
        outside_kernel=21.75,
        execute_to_kernel=3,
        kernel_to_join=1.75,
        event_to_execute=15,
        join_to_event=1,
        event_alignment=1,
    )
    assert kernels == {"LI": 10.0, "tail": 30.0}


def test_overlap_marks_exported_kernel_time_once_and_missing_markers_are_unresolved():
    rows = trace("LI")[1:-1]
    rows.append(task("KERNEL_AIVEC", 22, 7, name="overlap", tid=15))
    values, kernels = timeline(rows, elapsed=62)
    assert values["kernel_overlap"] == 6.75 and values["kernel_gap"] == 0
    assert "event_to_execute" not in values and sum(kernels.values()) == 47
    rows = [e for e in rows if "KERNEL" in e["args"]["Task Type"]]
    assert timeline(rows, elapsed=62)[0] == {
        k: v for k, v in values.items() if k not in ("execute_to_kernel", "kernel_to_join")
    }


@pytest.mark.parametrize("extra", [None, "init-only", "torch-front"])
def test_paired_rounds_check_profile_identity_and_retained_work(tmp_path, capsys, extra):
    init_only = extra == "init-only"
    cases = inputs()
    for number in (1, 2):
        for role in (("serial", "before", extra, "after") if extra else ("serial", "before", "after")):
            case = cases["before" if role == extra else role]
            case.update(diagnostics_commit="h", libraries=[], queue_probe={"status": "unavailable"})
            case["graph_backend"] = "torch.npu.NPUGraph" if role in ("serial", "torch-front") else "kvaf.AclGraph"
            directory = tmp_path / f"p{number}-{role}"
            directory.mkdir()
            name = "QuantLightningIndexer" if role == "serial" else "v9_li_kernel_0"
            (directory / "trace_view.json").write_text(json.dumps(trace(name)))
            experiment = dict(graph_backend=case["graph_backend"], snapshot_sha256=case["snapshot_sha256"], production_source=case["production_source"])
            if role != "serial":
                experiment.update(
                    auto_route=dict(
                        mode="low", fallback=False, prefetch=0, final_get=case["reads_samples"][-1]["final"]
                    )
                )
            case.update(
                system_profiles=dict(
                    passes=[
                        dict(
                            correctness="passed",
                            graph_replays=1,
                            metric="none",
                            included_in_headline_timing=False,
                            experiment=experiment,
                            graph_event_us=62,
                            directory=str(directory),
                        )
                    ]
                )
            )
            (tmp_path / f"r{number}-{role}.json").write_text(json.dumps(case))
    summarize(tmp_path)
    output = capsys.readouterr().out
    assert "EDGE_DELTA after-serial mean_us event_to_execute=+0.000" in output
    assert "PAIRED after-before" in output
    assert ("ABLATION init-only-before" in output) == init_only
    assert ("ABLATION after-init-only" in output) == init_only
    assert ("FRONTEND torch-front-before" in output) == (extra == "torch-front")
    if extra == "torch-front":
        for number in (1, 2):
            (tmp_path / f"r{number}-after.json").unlink()
        with pytest.raises(ValueError, match="incomplete"):
            summarize(tmp_path)
        summarize(tmp_path, frontend_only=True)
        assert "FRONTEND torch-front-before" in capsys.readouterr().out
    if init_only:
        one_side = tmp_path / "r2-init-only.json"
        saved = one_side.read_bytes()
        one_side.unlink()
        with pytest.raises(ValueError, match="incomplete"):
            summarize(tmp_path)
        one_side.write_bytes(saved)
    path = tmp_path / "r2-before.json"
    data = json.loads(path.read_text())
    data["system_profiles"]["passes"][0]["experiment"]["auto_route"]["final_get"] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="wrong trace route/traffic"):
        summarize(tmp_path, frontend_only=extra == "torch-front")
