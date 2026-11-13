import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from convert_trace import convert
from collect_fit_targets import collect


def test_legacy_conversion_and_metrics(tmp_path):
    source = tmp_path/'pid1-sample0000.pt'
    selected = torch.arange(2048, dtype=torch.int32).repeat(1,4,1)
    torch.save(dict(metadata=dict(schema=1,sample=0), before=dict(
        row_owners=torch.tensor([0]), logical_ids=torch.arange(4096).reshape(1,-1),
        hot_valid=torch.ones(1,2048,dtype=torch.uint8), weights=torch.ones(1,4,2,dtype=torch.bfloat16)),
        after=dict(logical_topk=selected)), source)
    root = tmp_path/'converted'
    m = convert(source, root/'capture-00000')
    assert m['arrays']['before.weights']['encoding'] == 'raw_bytes'
    with np.load(root/'capture-00000/arrays.npz') as z:
        assert z['before.weights'].nbytes == 16
    (root/'index.json').write_text(json.dumps(dict(status='COMPLETE',captures=[dict(bundle='capture-00000')])))
    rows = collect(root)
    assert rows[0]['selected_unique'] == 2048 and rows[0]['initial_cold'] == 0
    assert rows[0]['queries'][1]['cross_q_duplicates'] == 2048
    assert rows[0]['temporal'] is None and rows[0]['prefetch'] is None
    with pytest.raises(ValueError, match='already exists'):
        convert(source, root/'capture-00000')
    with (root/'capture-00000/arrays.npz').open('ab') as f:
        f.write(b'x')
    with pytest.raises(ValueError, match='checksum'):
        collect(root)


def test_current_schema_preserved(tmp_path):
    support = Path(__file__).resolve().parents[1] / 'support'
    sys.path.insert(0, str(support))
    from test_trace_inputs import capture_file
    from trace_inputs import load_score_capture
    source_dir = tmp_path/'source'
    source_dir.mkdir()
    source = capture_file(source_dir)
    convert(source, tmp_path/'output', support.parents[3])
    capture = load_score_capture(tmp_path/'output/schema4/manifest.json')
    assert capture.executable
    assert capture.arrays['real.keys'].shape == (1,4096,128)
