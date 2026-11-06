# SPDX-License-Identifier: Apache-2.0
"""Host checks for benchmark input identity and truthful device-time summaries."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "benchmark_transports", Path(__file__).with_name("benchmark_transports.py")
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("width", (656, 2624))
def test_payload_matches_chunked_put_and_contains_owner_token_tags(width):
    tokens = np.array([0, 127, 128, 131071, 131327], dtype=np.uint64)
    actual = benchmark.payload(16, tokens, width)
    chunks = np.concatenate([benchmark.payload(16, chunk, width) for chunk in np.array_split(tokens, 3)])
    assert np.array_equal(actual, chunks)
    assert actual.dtype == np.uint8 and actual.shape == (len(tokens), width)
    assert np.array_equal(actual[:, :8].copy().view("<u8").reshape(-1), tokens + np.uint64(16 << 32))
    assert not np.array_equal(actual, benchmark.payload(15, tokens, width))
    reference = ((tokens[:, None] % 251 + np.arange(width)[None, :] + 16 * 17) % 251).astype(np.uint8)
    assert np.array_equal(actual[:, 8:], reference[:, 8:])


def test_statistics_preserve_samples_and_microsecond_units():
    result = benchmark.statistics([12, 8, 10, 30])
    assert result["samples_us"] == [12, 8, 10, 30]
    assert result["p50_us"] == 11
    assert result["count"] == 4 and result["min_us"] == 8 and result["max_us"] == 30
    for bad in ([], [0], [-1], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            benchmark.statistics(bad)


def test_actual_urma_prefetch_final_partition_keeps_fixed_workload():
    baseline = dict(
        prefetch_hot=3840, prefetch_cold=6485, unused_hot=3840, unused_cold=0, final=75, total=10400, unused=3840
    )
    replay = dict(
        prefetch_hot=3840, prefetch_cold=6503, unused_hot=3840, unused_cold=0, final=57, total=10400, unused=3840
    )
    assert baseline != replay
    assert benchmark.fixed_read_workload(baseline) == benchmark.fixed_read_workload(replay)
    # Validation must not rewrite the recorded stage distribution.
    assert replay["prefetch_cold"] == 6503 and replay["final"] == 57


@pytest.mark.parametrize(
    "change",
    (
        {"prefetch_cold": 6484, "total": 10399},
        {"prefetch_hot": 3841, "prefetch_cold": 6484},
        {"unused_hot": 3839, "unused": 3839},
        {"unused_hot": 3839, "unused_cold": 1},
    ),
)
def test_fixed_workload_rejects_changed_transfer_or_unused_work(change):
    baseline = dict(
        prefetch_hot=3840, prefetch_cold=6485, unused_hot=3840, unused_cold=0, final=75, total=10400, unused=3840
    )
    assert benchmark.fixed_read_workload(baseline) != benchmark.fixed_read_workload({**baseline, **change})


@pytest.mark.parametrize(
    "change",
    (
        {"final": -1},
        {"total": 1},
        {"unused": 1},
        {"unused_cold": 6486, "unused": 10326},
        {"prefetch_hot": True},
        {"final": 75.0},
    ),
)
def test_fixed_workload_rejects_invalid_counter_evidence(change):
    baseline = dict(
        prefetch_hot=3840, prefetch_cold=6485, unused_hot=3840, unused_cold=0, final=75, total=10400, unused=3840
    )
    with pytest.raises(ValueError):
        benchmark.fixed_read_workload({**baseline, **change})


def test_explicit_transport_and_padded_capacity_contract(tmp_path):
    common = [
        "--build-dir",
        str(tmp_path),
        "--soc",
        "Ascend950DT_9582",
        "--token-bytes",
        "2624",
        "--history",
        "131328",
        "--sequence-length",
        "131072",
        "--output",
        str(tmp_path / "result.json"),
    ]
    args = benchmark.arguments(["--transport", "ubmem", *common])
    assert args.history == 131328 and args.sequence_length == 131072
    assert args.batch == 16 and args.index_heads == 32 and args.resident == 8192
    assert args.warmup == 4 and args.iterations == 20
    with pytest.raises(SystemExit):
        benchmark.arguments(["--transport", "urma", *common])
    with pytest.raises(SystemExit):
        benchmark.arguments(["--transport", "ubmem", "--kvaf-server", "127.0.0.1:19000", *common])
    args = benchmark.arguments(["--transport", "urma", "--kvaf-server", "127.0.0.1:19000", *common])
    assert args.kvaf_server == "127.0.0.1:19000"
