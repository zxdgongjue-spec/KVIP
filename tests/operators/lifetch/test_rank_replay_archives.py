# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from rank_replay_archives import BATCHES, CONTROL_REV, GROUPS, PROTOCOL, SOURCE_REV, discover


def marker(root, key, value):
    (root / (key + ".txt")).write_text(value)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_select_completed_same_card_pair_or_use_standalone(tmp_path):
    current, root, control = [tmp_path / n for n in ("current", "arbitrary-input", "fixed-q4-packed.local")]
    for path in (current, root, control):
        path.mkdir()
        for key, value in {"host": "localhost", "machine_id": "h20", "physical_npu": "4",
                           "kvaf": "sdk", "kvaf_tree": "tree"}.items():
            marker(path, key, value)
    marker(root, "native", SOURCE_REV)
    marker(control, "native", CONTROL_REV)
    marker(control, "packed_source", str(root))
    save(control / "packed_summary.json", dict(protocol=PROTOCOL, blocks=2, cells=[
        dict(group=g, batch=b) for g in GROUPS for b in (1, 2, 4, 8, 16, *BATCHES)]))
    assert discover(tmp_path, current) == (None, None)
    for b in BATCHES:
        for g in GROUPS:
            cell = Path(f"g{g}/bs{b}")
            save(root / cell / "fused/ARM_RESULT.json", {})
            for fname in ("selected.npy", "ARM_RESULT.json", "thp.json", "path/path_result.json"):
                save(root / cell / "serial" / fname, {})
            for n in (1, 2):
                for arm in ("serial", "fused"):
                    for filename in ("packed_result.json", "placement_verified", "thp.json"):
                        save(control / cell / f"block{n}" / arm / filename, {})
    assert discover(tmp_path, current) == (control, root)
    for key, wrong, right in (("machine_id", "h24", "h20"), ("physical_npu", "3", "4")):
        marker(control, key, wrong)
        assert discover(tmp_path, current) == (None, None)
        marker(control, key, right)
    (control / "packed_source.txt").unlink()
    assert discover(tmp_path, current) == (None, None)
