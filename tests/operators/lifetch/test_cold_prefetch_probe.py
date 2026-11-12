"""CPU-only fixtures for one real Cold TopK winner per active query."""

import sys
from pathlib import Path

import numpy as np
import pytest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

from real_fetch_inputs import prepare_full_chain_fixture  # noqa: E402
from prefetch_payload_audit import audit_prefetch_mapping  # noqa: E402
from test_prefetch_payload_audit import fixture, plan, check_payload  # noqa: E402


HISTORY = 8192
RESIDENT = 2048
ONE_MISS_PERCENT = 100.0 / 2048.0


def test_first_cold_tile_contains_one_final_winner_per_query(tmp_path):
    root = tmp_path / "fixture"
    info = prepare_full_chain_fixture(
        root, HISTORY, RESIDENT, ONE_MISS_PERCENT,
        index_heads=32, cold_probe_per_query=1,
    )
    assert info["final_misses_per_query"] == [1, 1, 1, 1]
    assert info["cold_probe_physical_tokens"] == [
        RESIDENT, RESIDENT + 16, RESIDENT + 32, RESIDENT + 48
    ]
    encoded = np.fromfile(root / "arg1.bin", dtype=np.uint8).reshape(HISTORY, 128)
    order = np.load(root / "physical_to_logical.npy")
    scores = np.load(root / "logical_scores.npy")[:, order]
    for query, token in enumerate(info["cold_probe_physical_tokens"]):
        # The TopK set is unique; one cold ID is inside the first cold tile.
        assert np.all(encoded[token, query * 32:(query + 1) * 32] == 0x44)
        cutoff = np.partition(scores[query], -2048)[-2048]
        selected = np.flatnonzero(scores[query] >= cutoff)
        assert len(selected) == 2048
        cold = selected[selected >= RESIDENT]
        assert cold.tolist() == [token]


@pytest.mark.parametrize("percent,heads", [
    (0.0, 32), (0.2, 32), (ONE_MISS_PERCENT, 64),
])
def test_cold_probe_rejects_uncontrolled_fixture(tmp_path, percent, heads):
    with pytest.raises(ValueError, match="one final miss"):
        prepare_full_chain_fixture(
            tmp_path / "fixture", HISTORY, RESIDENT, percent,
            index_heads=heads, cold_probe_per_query=1,
        )


def test_prefetch_audit_uses_direct_physical_slots_without_lru():
    data = fixture()
    source, output = data[-2:]
    output[0, 8] = source[0, 3]
    output[0, 9] = source[0, 12]
    output[1, 8] = source[1, 11]
    report = check_payload(plan(data, use_lru_slots=False), data)
    assert report["summary"]["validation"] == "PASS"
    assert [
        e["physical_slot"]
        for e in report["rows"][0]["mappings"]
    ] == [8, 9]
    assert report["summary"]["payload_tokens_checked"] == 3
