# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import numpy as np
import pytest

from four_rank_accept import compare_counts, compare_journals
from four_rank_accept_worker import canonical_journal


def test_query_order_and_remote_aliases(tmp_path):
    inputs = SimpleNamespace(
        captured_layout=np.array([[3, 131, 129, 0]]),
        logical_address_valid=np.ones((1, 256)),
        geometry={"resident": 8},
    )
    pages = SimpleNamespace(page_indices=np.array([[9, 9]]))
    addresses = np.tile(0x12340000 + np.arange(128, dtype=np.uint64) * 656, 2)[None, :]
    src = np.array([[0, 1, 2, 3]])
    dst = np.array([[3, 1, 5, 6]])
    epochs = [[1, 0, 1, 0]]
    final = [[1, 1, 0, 0]]
    journal = canonical_journal(
        inputs, pages, addresses, src, dst, epochs, final, np.array([[3, 2, 0, 0]]), 4, 656
    )
    assert journal["r0_w0"].tolist() == [[0, 3, 9, 1968, 3]]
    assert journal["r0_w2"].tolist() == [[1, 131, 9, 1968, 1]]
    assert journal["r0_q0"].tolist() == [[3, 0, 9, 0, 6]]
    assert journal["r0_q1"].tolist() == [[2, 129, 9, 656, 5]]
    with pytest.raises(ValueError, match="Overlapping"):
        canonical_journal(inputs, pages, addresses, src, dst, epochs, final, np.array([[3, 3, 0, 0]]), 4, 656)
    dirs = [tmp_path / "a", tmp_path / "b"]
    for i, p in enumerate(dirs):
        p.mkdir()
        (p / "journal.json").write_text(json.dumps(dict(tail_text="x", compiler_sha256="h", chunks=8)))
        if i:
            journal["r0_q1"][0, -1] = 7
        np.savez(p / "journal.npz", **journal)
    check = compare_journals(*dirs)
    assert check["matched_segments"] == 7 and check["differences"][0]["segment"] == "r0_q1"


def test_acceptance_requires_ten_individual_call_count_matches():
    a = dict(row_window_prefetch=[[1, 2, 3, 4]], row_query_final=[[5, 0, 0, 0]])
    b = dict(row_window_prefetch=[[1, 2, 3, 4]], row_query_final=[[0, 5, 0, 0]])
    assert compare_counts([a, b] * 5, [b, a] * 5) == (0, True)
