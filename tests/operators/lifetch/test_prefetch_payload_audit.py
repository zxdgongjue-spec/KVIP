"""CPU-only tests of P1-C0 row/CQ/mapping and HBM-vs-DDR payload audit."""

import sys
from pathlib import Path

import numpy as np
import pytest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))
from prefetch_payload_audit import (  # noqa: E402
    STATUS_MAGIC,
    audit_prefetch_mapping,
    verify_prefetch_payload,
)


def fixture():
    batch, history, resident, token_bytes = 2, 32, 8, 16
    mapping = np.full((batch, history), -1, np.int32)
    mapping[0, [3, 12]] = [resident, resident + 1]
    mapping[1, 11] = resident
    lru = np.broadcast_to(np.arange(2048, dtype=np.int32), (batch, 2048)).copy()
    rows = np.zeros((batch, 256), np.int32)
    rows[:, 8] = [2, 1]
    rows[:, 9] = [2, 1]
    rows[:, 16] = [2, 1]
    status = np.zeros((batch * 2, 8), np.uint64)
    status[:, 0] = STATUS_MAGIC
    status[::2, 1] = [2, 1]
    status[::2, 3] = [1, 1]  # two completed CQ groups, three tokens
    source = np.arange(batch * history * token_bytes, dtype=np.uint8).reshape(
        batch, history, token_bytes
    )
    output = np.zeros((batch, 16, token_bytes), np.uint8)
    output[0, 0] = source[0, 3]
    output[0, 1] = source[0, 12]
    output[1, 0] = source[1, 11]
    return mapping, lru, rows, status, source, output


def plan(data, **overrides):
    mapping, lru, rows, status, *_ = data
    args = {
        "resident": 8, "output_stride": 16, "rounds": 2,
        "selected_for_post": 3,
    }
    args.update(overrides)
    return audit_prefetch_mapping(mapping, lru, rows, status, [32, 32], **args)


def check_payload(result, data):
    source, output = data[-2:]
    return verify_prefetch_payload(
        result,
        read_hbm=lambda row, slots: output[row, slots],
        read_ddr=lambda row, tokens: source[row, tokens],
        token_bytes=16,
    )


def test_p1c0_reconciles_completed_groups_not_tokens_and_payload():
    data = fixture()
    report = check_payload(plan(data), data)
    assert report["errors"] == []
    assert report["summary"]["validation"] == "PASS"
    assert report["summary"]["requests"] == 3
    assert report["summary"]["mapping_count"] == 3
    assert report["summary"]["cq_completed_groups"] == 2
    assert report["summary"]["resident_candidate_count"] == 1
    assert report["summary"]["cold_candidate_count"] == 2
    assert report["summary"]["payload_tokens_checked"] == 3
    assert report["summary"]["payload_bytes_checked"] == 48


def test_p1c0_detects_wrong_hbm_payload():
    data = fixture()
    data[-1][0, 1, 5] ^= 0x40
    report = check_payload(plan(data), data)
    assert report["summary"]["validation"] == "FAIL"
    assert any("payload mismatch offset=5" in e for e in report["errors"])


def test_p1c0_detects_bad_physical_lru_collision():
    data = fixture()
    data[1][0, 1] = 0
    report = plan(data)
    assert report["summary"]["validation"] == "FAIL"
    assert any("physical HBM slot collision" in e for e in report["errors"])
    with pytest.raises(ValueError, match="refuse payload reads"):
        check_payload(report, data)


def test_p1c0_detects_missing_cqe_or_token_count_mismatch():
    data = fixture()
    data[3][0, 3] = 0
    report = plan(data)
    assert report["summary"]["validation"] == "FAIL"
    assert any("completed CQ groups=0" in e for e in report["errors"])

    data = fixture()
    data[2][0, 16] = 1
    assert any("sum(epochCount)" in e for e in plan(data)["errors"])


def test_p1c0_detects_key_bounds_and_unexpected_post_count():
    data = fixture()
    report = audit_prefetch_mapping(
        data[0], data[1], data[2], data[3], [11, 32],
        resident=8, output_stride=16, rounds=2, selected_for_post=5,
    )
    assert report["summary"]["validation"] == "FAIL"
    assert any("exceeds key length" in e for e in report["errors"])
    assert any("selectedForPost" in e for e in report["errors"])


def test_p1c0_rejects_unexpected_abi():
    data = fixture()
    with pytest.raises(ValueError, match="invalid P1-C0"):
        audit_prefetch_mapping(
            data[0], data[1], data[2][:, :32], data[3], [32, 32],
            resident=8, output_stride=16, rounds=2, selected_for_post=3,
        )
