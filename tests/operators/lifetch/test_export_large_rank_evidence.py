# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from export_large_rank_evidence import BATCHES, GROUPS, IDENTITY_KEYS, PACKED, REFERENCE, collect


def test_partial_serial_archive_on_same_card(tmp_path):
    area = tmp_path / "peer-baseline"
    current = area / "candidate"
    good = area / "partial-control"
    other_card = area / "other-card"
    for root, native, card in ((current, "new", 4), (good, REFERENCE, 4), (other_card, REFERENCE, 3)):
        root.mkdir(parents=True)
        for k, v in dict(native=native, host="b24", machine_id="host", physical_npu=card,
                         selection_oracle="captured_logical_scores", kvaf="sdk", kvaf_tree="tree").items():
            (root / f"{k}.txt").write_text(str(v))
        for g in GROUPS:
            for b in BATCHES:
                if root == good and b != 32:
                    continue
                arm = "fused" if root == current else "serial"
                path = root / f"g{g}/bs{b}/block1" / arm
                path.mkdir(parents=True)
                (path / "placement_verified").touch()
                (path / "thp.json").write_text(json.dumps(dict(status="measured", page_mode="thp", ub_device="n",
                    coverage_min_pct=99.9, numa_pages_overlapping_vmas={"3": 100})))
                identity = dict.fromkeys(IDENTITY_KEYS, "same")
                identity.update(group=g, batch=b, warmup=4, calls=10, protocol=PACKED, host="b24",
                                physical_npu=card, native=native)
                if root == good and g == 2:
                    identity["sequence_sha256"] = "other requests"
                record = dict(identity=identity, arm=arm, correctness="PASS", calls=10, get_workers=2,
                              full_KV_Index_checked_calls=10, checked_queries=40*b, checked_rows=10*b,
                              samples=[{}]*10, graph_event_us=3000, graph_mean_us=300,
                              mean_us=dict(LI=200, tail=90))
                (path / "packed_result.json").write_text(json.dumps(record))
    collect(current, tmp_path)
    archive, = tmp_path.glob("b24-npu4-rank-evidence.*/rank-evidence.tar.gz")
    with tarfile.open(archive) as bundle:
        data = json.load(bundle.extractfile("comparison.json"))
        assert len(data["compared"]) == 4 and len(data["missing_serial_cells"]) == 16
        assert len(data["rejected"]) == 1
        assert not any("other-card" in p for p in bundle.getnames())
