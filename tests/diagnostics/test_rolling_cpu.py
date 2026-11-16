import numpy as np
import pytest

from kvip.diagnostics.rolling_cpu import audit


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("kind", ["random", "ties", "extremes"])
def test_actual_window_audit_checks_every_prefix_and_operation_count(reverse, kind):
    rng = np.random.default_rng(42)
    scores = rng.integers(15000, 15300, (1, 4, 32768), dtype=np.uint16)
    if kind == "ties":
        scores.fill(15000)
    elif kind == "extremes":
        scores[..., ::2] = 0
        scores[..., 1::2] = 65535
    ends = np.array([[32601, 32602, 32603, 32604]])
    result = audit(scores, ends, 16384, np.array([[30720, 32512]]), 8, reverse)
    assert result["boundaries"] == 28
    assert result["cdf_tile"] == 4*128
    assert result["cdf_window"] == 28
    assert result["histogram_window"] < .53*result["histogram_tile"]
    assert np.all(np.diff(result["keys"], axis=1) >= 0)


def test_oversized_window_is_rejected_before_any_device_use():
    scores = np.zeros((1, 4, 65536), dtype=np.uint16)
    with pytest.raises(ValueError, match="exceeds native"):
        audit(scores, np.full((1, 4), 65536), 16384, np.array([[60000, 61000]]), 2, False)


def test_cpu_cli_reads_only_scores_and_bounds_not_kv(tmp_path, monkeypatch, capsys):
    import json
    from kvip.diagnostics import rolling_cpu
    from kvip.diagnostics.archive import TensorArchive

    manifest = dict(schema_version=2, status="complete", geometry=dict(
        batch=16, resident=16384, requested_windows=8, aic_count=32))
    archive = TensorArchive(tmp_path, manifest)
    archive.save("post.score_keys", np.zeros((16, 4, 32768), dtype=np.uint16))
    archive.save("pre.query_ends", np.full((16, 4), 32600, dtype=np.int32))
    archive.save("pre.threshold_bounds", np.tile([30720, 32512], (16, 1)).astype(np.int32))
    # The real replay validates every file. CPU threshold auditing must not
    # require reading or even opening unrelated, potentially large KV data.
    manifest["tensors"]["pre.hot_kv"] = dict(file="absent.npy", dtype="uint8",
        shape=[10], nbytes=10, sha256="unused")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("sys.argv", ["rolling_cpu", "--snapshot", str(tmp_path)])
    rolling_cpu.main()
    output = capsys.readouterr().out
    assert "CPU_ORACLE forward PASS queries=64 window_checks=448" in output
    assert "CPU_ORACLE reverse PASS queries=64 window_checks=448" in output
