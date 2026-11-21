"""Merged snapshots must preserve Q unpack order, DP page aliases and cold positions."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
import query_owner_input as owner_input
from workloads import CaptureReference


def test_recover_merged_input(tmp_path, monkeypatch):
    n, h, r = 16, 4096, 2048
    region, slot = 2**32, 128 * 656
    manifest = dict(geometry=dict(batch=n, history=h, resident=r, heads=32,
                                   token_bytes=656, output_stride=r+512),
                    tensors={}, ddr=dict(region_base=region, slot_bytes=slot),
                    metadata=dict(ddr_subarenas=[dict(dp_rank=i, slot_offset=i*32, region_bytes=32*slot)
                                                  for i in range(8)]))
    def save(field, value, dtype=None):
        path = tmp_path / (field + '.npy')
        np.save(path, np.ascontiguousarray(value).view(np.uint8).reshape(-1))
        manifest['tensors'][field] = dict(file=path.name, dtype=dtype or value.dtype.name,
                                          shape=value.shape, nbytes=value.nbytes,
                                          sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    meta = np.ones((5, 64), dtype=np.int64)
    order = np.arange(64).reshape(8,8)[:,::-1].ravel()
    meta[0] = meta[2] = order
    meta[4].view(np.int32)[:n] = np.arange(1, n+1)
    save('pre.metadata', meta)
    layout = np.tile(np.arange(h)[::-1], (n+1,1)).astype(np.int64)
    save('pre.logical_ids', layout)
    save('pre.hot_valid', np.ones((n+1,r), dtype=np.int32))
    addr = np.zeros_like(layout)
    for row in range(n):
        addr[row+1] = region + (row//2*h + layout[row+1])*656
        addr[row+1, :128] = 0
    save('pre.addresses', addr)
    save('pre.lru_slots', np.tile(np.arange(r, dtype=np.int32), (n,1)))
    save('pre.query_ends', np.full(64,h,dtype=np.int32))
    save('pre.tail_bounds', np.full((n,2),h,dtype=np.int32))
    save('pre.keys', np.zeros((n+1,h,128), dtype=np.uint8))
    save('pre.key_scales', np.ones((n+1,h), dtype=np.float32))
    save('pre.query', np.zeros((64,32,128), dtype=np.uint8), 'float8_e4m3fn')
    save('pre.query_scale', np.ones((64,32), dtype=np.float32))
    save('pre.weights', np.zeros((64,32), dtype=np.uint16), 'bfloat16')
    save('post.score_keys', np.full((n,4,h),0x8000,dtype=np.uint16))
    top = np.arange(2048)[None,:] + np.arange(64)[:,None]%4
    save('post.logical_output', (h-1-top[order]).astype(np.int32))
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    refs = [CaptureReference(i,1,900,2,path,((f'r{i}a',3),(f'r{i}b',2))) for i in range(8)]
    monkeypatch.setitem(owner_input.COUNTS,2,(16*2051,16*3))
    recovered = owner_input.merged_round(refs)
    assert recovered.page_keys[0] == recovered.page_keys[1]
    assert recovered.page_keys[0][1] == (0,1) and recovered.page_keys[2][1] == (1,1)
    assert recovered.page_keys[-1][-1] is None
    assert recovered.identities[2] == (1,'r1a',3)
    assert np.array_equal(recovered.captured_layout,layout[1:])
    path = tmp_path/'pre.keys.npy'
    with path.open('r+b') as f:
        f.seek(-1,2)
        f.write(b'x')
    with pytest.raises(ValueError,match='checksum'):
        owner_input.merged_round(refs)
