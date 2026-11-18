# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import runpy

import pytest


bracket = runpy.run_path(str(Path(__file__).parent / "tools/high_index_model/bracket.py"))


def sample(layout, values):
    from statistics import mean, median

    return (
        f"INDEX_MODEL_START B=16 cores=64 history=131328 resident=16384 layout={layout} events=5000 moves=10000\n"
        "DEVICE_INDEX_SAMPLES_US " + " ".join(map(str, values)) + "\n"
        f"DEVICE_INDEX_US mean={mean(values):.3f} p50={median(values):.3f} samples=6 warmup=1 diagnostics_excluded=true\n"
        "MODEL_EXACT_PASS events=5000 moved=10000 scope=index_only synthetic\n"
    )


def test_same_work_bracket_separates_mean_median_and_controls(tmp_path, capsys):
    old, new = "a" * 40, "b" * 40
    for name, rev in (("before", old), ("after", new)):
        path = tmp_path / name
        path.mkdir()
        (path / "model-source.json").write_text(
            json.dumps(
                dict(revision=rev, mode="npu", soc="Ascend950DT_9582", include="headers", harness_sha256="harness")
            )
        )
    for role, values in (("before", [55.0] * 6), ("after", [49.0] * 5 + [60.0]), ("before-end", [53.0] * 6)):
        (tmp_path / (role + ".log")).write_text(sample("even", values) + sample("skew", values))
    report = bracket["summarize"](tmp_path, "moves", old, new)
    assert json.loads((tmp_path / "summary.json").read_text()) == report
    assert report["comparisons"]["skew"]["at_most_50us"] == {"mean_us": False, "p50_us": True}
    assert report["comparisons"]["even"]["before_end_minus_before_us"]["p50_us"] == -2
    assert "at_most_50us=MISS/PASS" in capsys.readouterr().out


def test_rejects_changed_work_or_missing_success(tmp_path):
    path = tmp_path / "after.log"
    for broken in (
        sample("even", [50.0] * 6).replace("moved=10000", "moved=9999"),
        sample("even", [50.0] * 6).replace("MODEL_EXACT_PASS", "failure"),
    ):
        path.write_text(broken)
        with pytest.raises(ValueError):
            bracket["read_arm"](path, ("even",), "moves")
