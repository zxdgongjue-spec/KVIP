"""The customer matrix and generated entry points must agree before device runs."""

import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_customer_cartesian_product():
    module = load("run_split_core_cases")
    rows = list(module.cases())
    expected = set(itertools.product((1, 2, 4, 8, 16, 24, 32), (3, 5, 10, 15, 20, 50)))
    assert len(rows) == 42 and {(r["batch"], r["miss_percent"]) for r in rows} == expected
    assert len({r["case"] for r in rows}) == 42
    assert [len(list(module.cases(g))) for g in ("small", "medium", "large")] == [12, 12, 18]
    for r in rows:
        assert r["history"] == r["valid_history"] == 131072
        assert r["queries_per_request"] == 4 and r["total_queries"] == 4 * r["batch"]
        assert r["miss_per_query"] == {3: 61, 5: 102, 10: 205, 15: 307, 20: 410, 50: 1024}[r["miss_percent"]]
        assert r["actual_miss_percent"] == 100 * r["miss_per_query"] / 2048


@pytest.mark.parametrize("suite", ("smoke", "customer"))
def test_generated_entry_plan_without_cann(tmp_path, suite):
    generated = tmp_path / "generated"
    manifest = load("generate_split_core_sim").generate(generated)
    before = {p.name: p.read_bytes() for p in generated.iterdir()}
    load("generate_split_core_sim").generate(generated)
    assert before == {p.name: p.read_bytes() for p in generated.iterdir()}
    assert manifest["suites"] == {"smoke": 42, "customer": 42}
    output = tmp_path / "plan"
    env = {
        **os.environ,
        "KVIP_ACTION": "plan",
        "KVIP_ROOT": str(ROOT),
        "KVIP_PYTHON": sys.executable,
        "KVIP_OUTPUT": str(output),
    }
    subprocess.run(["bash", "-n", str(generated / f"{suite}.sh")], check=True)
    subprocess.run(["bash", str(generated / f"{suite}.sh"), "--shards", "8"], env=env, check=True, capture_output=True)
    rows = json.loads((output / "results.json").read_text())
    assert len(rows) == 42 and all(r["status"] == "NOT_RUN" and r["metric"] is None for r in rows)
    assert all(
        r["suite"] == suite and r["topk_shards"] == 8 and r["warmup"] == 0 and r["iterations"] == 1 for r in rows
    )
    assert any(r["batch"] == 32 and r["miss_percent"] == 50 and r["total_queries"] == 128 for r in rows)


def test_generated_comparison_keeps_all_condition_axes(tmp_path):
    generated = tmp_path / "generated"
    load("generate_split_core_sim").generate(generated)
    output = tmp_path / "comparison"
    env = {
        **os.environ,
        "KVIP_ACTION": "plan",
        "KVIP_ROOT": str(ROOT),
        "KVIP_PYTHON": sys.executable,
        "KVIP_OUTPUT": str(output),
    }
    subprocess.run(["bash", str(generated / "compare.sh")], check=True, env=env, capture_output=True)
    rows = [r for p in output.glob("*/results.json") for r in json.loads(p.read_text())]
    keys = {(r["case"], r["topk_shards"], r["aic_budget"], r["aiv_budget"], r["heads"]) for r in rows}
    assert len(rows) == len(keys) == 252
    assert all(r["status"] == "NOT_RUN" and r["aic_workers"] == 0 for r in rows)


@pytest.mark.parametrize("batch", (24, 32))
def test_native_large_batch_is_rejected_before_output_or_resource_use(tmp_path, batch):
    output = tmp_path / "must-not-exist"
    module = load("run_split_core_cases")
    with pytest.raises(SystemExit) as error:
        module.main(
            [
                "run",
                "--backend",
                "native",
                "--soc",
                "test-only-not-launched",
                "--case",
                f"b{batch}-h100-m50",
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2 and not output.exists()


def test_model_old_driver_capability_is_rejected_before_output(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = load("run_split_core_cases")
    driver = tmp_path / "old-model-driver"
    driver.touch()
    monkeypatch.setenv("LD_PRELOAD", "libruntime_camodel.so")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps({"scope": "AIV-only TopK", "queries_per_request": 4, "max_batch": 16})
        ),
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit) as error:
        module.main(["run", "--model-driver", str(driver), "--output", str(output)])
    assert error.value.code == 2 and not output.exists()


def test_checked_in_scripts_match_generator(tmp_path):
    output = tmp_path / "generated"
    load("generate_split_core_sim").generate(output)
    for p in output.iterdir():
        assert p.read_bytes() == (ROOT / "scripts/split_core_sim" / p.name).read_bytes()


@pytest.mark.parametrize(
    "script,args",
    (
        ("smoke.sh", ["--backend", "native"]),
        ("customer.sh", ["--backend=native"]),
        ("customer.sh", ["--suite=smoke"]),
        ("compare.sh", ["--shards", "1"]),
        ("compare.sh", ["--budget=32/64"]),
        ("compare.sh", ["--output", "override"]),
    ),
)
def test_generated_entry_rejects_reserved_overrides(tmp_path, script, args):
    generated = tmp_path / "generated"
    load("generate_split_core_sim").generate(generated)
    output = tmp_path / "must-not-exist"
    env = {
        **os.environ,
        "KVIP_ACTION": "plan",
        "KVIP_ROOT": str(ROOT),
        "KVIP_PYTHON": sys.executable,
        "KVIP_OUTPUT": str(output),
    }
    result = subprocess.run(["bash", str(generated / script), *args], env=env, capture_output=True)
    assert result.returncode == 2 and not output.exists()


def test_capability_query_ignores_loader_noise_without_model_preload(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = load("run_split_core_cases")
    caps = {"scope": "AIV-only TopK", "max_batch": 32, "queries_per_request": 4}
    monkeypatch.setenv("LD_PRELOAD", "libruntime_camodel.so")

    def capture(*args, **kwargs):
        assert "LD_PRELOAD" not in kwargs["env"]
        return SimpleNamespace(stdout="[SDK loader notice]\n" + json.dumps(caps) + "\n")

    monkeypatch.setattr(module.subprocess, "run", capture)
    assert module.driver_capabilities(tmp_path / "driver") == caps
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(caps) + "\n" + json.dumps(caps)),
    )
    with pytest.raises(ValueError):
        module.driver_capabilities(tmp_path / "driver")
