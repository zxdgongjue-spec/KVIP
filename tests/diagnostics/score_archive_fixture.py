"""Small historical schema-4 archives without invoking a retired recorder."""

import json
from types import SimpleNamespace

import numpy as np

from kvip.diagnostics.archive import file_digest
from kvip.diagnostics.token_trace import trace_summary
from kvip.operators.lifetch.geometry import final_aivs, li_geometry


def score_archive(*, real=False, rank=0, scheduler_round=51, width=2624):
    b, h, r, heads, stride = 2, 8192, 4096, 32, 4480
    ids = np.tile(np.arange(h, dtype=np.int32), (b, 1))
    owners = np.array([2, 1], dtype=np.int32)
    metadata = np.zeros((5, b * 4), dtype=np.int64)
    metadata[0] = np.arange(b * 4)
    metadata[1:4:2] = 1
    metadata[2] = [4, 5, 6, 7, 0, 1, 2, 3]
    metadata[4].view(np.int32)[:b] = owners
    ends = np.tile(np.arange(8065, 8069, dtype=np.int32), (b, 1))
    topk = np.empty((b, 4, 2048), dtype=np.int32)
    for row in range(b):
        for q in range(4):
            topk[row, q] = [*range(2046), 5000, ends[row, q] - 1]
    selected = topk.reshape(b * 4, 2048)[metadata[2]]
    post_ids = ids.copy()
    post_ids[:, [2046, 5000]] = post_ids[:, [5000, 2046]]
    slots = np.empty_like(selected)
    for tnd, packed in enumerate(metadata[2]):
        row = packed // 4
        inverse = np.argsort(post_ids[row])
        slots[tnd] = inverse[selected[tnd]]
        abc = selected[tnd] >= 8064
        slots[tnd, abc] = r + selected[tnd, abc] % (stride - r)
    scores = np.full((b, 4, h), 0x8000, dtype=np.uint16)
    keys = np.zeros((b, h, 128), dtype=np.uint8)
    records = np.zeros((b, 4, 3, 4), dtype=np.int32)
    for row in range(b):
        for q in range(4):
            scores[row, q, topk[row, q]] = 0xBF80
            scores[row, q, 0] = 0xC000
            scores[row, q, 5001] = 0xBF00
            scores[row, q, ends[row, q] :] = 0xC100
            keys[row, topk[row, q], q] = 0x38
            keys[row, 0, q] = 0x40
            keys[row, 5001, q] = 0x30
            records[row, q, 0] = [r, r, 0x8000, 1]
            records[row, q, -1] = [ends[row, q], h, 0xBF80, 4]
    scheduling = li_geometry(b, "auto", 8, 32)
    windows = scheduling["intermediate_fetches"]
    arrays = {
        "pre.row_owners": owners,
        "pre.logical_ids": ids,
        "pre.hot_valid": np.ones((b, r), dtype=np.uint8),
        "pre.metadata": metadata,
        "pre.query_ends": ends,
        "pre.tail_bounds": np.tile([8064, 8068], (b, 1)).astype(np.int32),
        "pre.lru_slots": np.tile(np.arange(2048, dtype=np.int32), (b, 1)),
        "pre.address_valid": (ids < 8064).astype(np.uint8),
        "pre.ddr_page_slots": np.tile(np.r_[np.arange(63), -1], (b, 1)).astype(np.int32),
        "post.logical_ids": post_ids,
        "post.hot_valid": np.ones((b, r), dtype=np.uint8),
        "post.logical_output": selected,
        "post.slot_output": slots,
        "post.topk": topk,
        "post.prefetch_records": np.array([[row, p, p] for row in range(b) for p in (5000, 5001)], dtype=np.int32),
        "post.final_records": np.empty((0, 3), dtype=np.int32),
        "post.read_counts": np.tile([2, 0, 2 * width], (b, 1)).astype(np.int64),
        "post.score_keys": scores,
        "post.threshold_records": records,
        "post.window_posted": np.tile([2] + [0] * (windows - 1), (b, 1)).astype(np.int32),
        "post.window_processed_ends": np.full((b, windows, 4), h, dtype=np.int32),
        "post.prefetch_ordinals": np.tile([0, 1], b).astype(np.int32),
    }
    if real:
        query = np.zeros((b, 4, heads, 128), dtype=np.uint8)
        weights = np.zeros((b, 4, heads), dtype=np.uint16)
        weights[:, :, 0] = 0x3F80
        for q in range(4):
            query[:, q, 0, q] = 0x38
        arrays.update(
            {
                "real.keys": keys,
                "real.key_scales": np.ones((b, h), dtype=np.float32),
                "real.query": query,
                "real.weights": weights,
                "real.query_scale": np.ones((b, 4, heads), dtype=np.float32),
            }
        )
    slot_bytes = ((width * 128 + 4095) // 4096) * 4096
    manifest = dict(
        schema_version=4,
        kind="kvaf_decode_score_trace",
        content="scores",
        status="complete",
        source_transport="ubmem",
        real_inputs_saved=real,
        validation=dict(passed=True, kind="score_trace_integrity"),
        geometry=dict(
            batch=b,
            history=h,
            resident=r,
            output_stride=stride,
            token_bytes=width,
            tokens_per_block=128,
            slot_bytes=slot_bytes,
            region_bytes=128 * slot_bytes,
            owners=3,
            token_count=b * 4,
            heads=heads,
            aic_count=32,
            requested_windows=8,
            rounds=windows,
            service_lanes=scheduling["service_aivs_per_window"],
            final_workers=final_aivs(b, "auto", 32),
            prefetch_limit=512,
            window_limit=64,
            split_q=True,
            threshold_only=False,
            rolling_threshold=False,
            page_mode="thp",
            numa_node=2,
            soc="Ascend950DT_9582",
        ),
        metadata=dict(
            dp_rank=rank,
            dp_size=2,
            group=2,
            content="scores",
            capture_real_inputs=real,
            capture_mode="eager_diagnostic",
            layer_names=[f"model.layers.{i}" for i in range(width // 656)],
            geometry=dict(batch=b),
            context=dict(round=scheduler_round, global_decode_confirmed=True, min_context=8064, max_context=8068),
            requests=[
                dict(request_id=f"request{i}", owner=int(owner), query_begin=i * 4, query_stop=(i + 1) * 4)
                for i, owner in enumerate(owners)
            ],
        ),
    )
    trace = SimpleNamespace(manifest=manifest, arrays=arrays, array=lambda name: arrays[name])
    manifest["summary"] = trace_summary(trace)
    return trace


def write_archive(trace, directory):
    directory.mkdir(parents=True)
    path = directory / "scores.npz"
    np.savez_compressed(path, **trace.arrays)
    trace.manifest["archive"] = dict(file=path.name, sha256=file_digest(path), bytes=path.stat().st_size)
    trace.manifest["arrays"] = {name: dict(dtype=a.dtype.name, shape=list(a.shape)) for name, a in trace.arrays.items()}
    (directory / "manifest.json").write_text(json.dumps(trace.manifest))
    return directory
