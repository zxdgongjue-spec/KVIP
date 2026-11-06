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
