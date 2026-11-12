# SPDX-License-Identifier: Apache-2.0
"""Historical trace readers verify bytes, shapes and cache identity before use."""

import pytest

from kvip.diagnostics.token_trace import load_token_trace
from score_archive_fixture import score_archive, write_archive


@pytest.mark.parametrize("real", [False, True])
def test_historical_score_archive_reader_roundtrip(tmp_path, real):
    source = score_archive(real=real)
    path = write_archive(source, tmp_path / "trace")
    result = load_token_trace(path)
    assert result.manifest["summary"] == source.manifest["summary"]
    assert result.manifest["summary"]["demand"]["cold_demand"] == 2
    assert result.manifest["real_inputs_saved"] is real
    result.close()
    assert not result.arrays


def test_reader_rejects_changed_archive_bytes(tmp_path):
    path = write_archive(score_archive(), tmp_path / "trace")
    archive = path / "scores.npz"
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_token_trace(path)
