# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from kvip.diagnostics.frozen_high_tail import ARMS, ORDERS, descriptor_slots, device_spans, paired_values
from kvip.diagnostics.high_index_profile import verify_permutation


def test_identical_load_counterbalanced_contrasts_keep_raw_sign_and_drift():
    samples = []
    for i, order in enumerate(ORDERS):
        start = 120 + 7 * i
        times = dict(get_only=start, overlap=start - 1 + i, wait=start + 25)
        samples.append(dict(order=order, arms={arm: {"joined_us": times[arm]} for arm in order}))
    for arm in ARMS:
        assert [order.index(arm) for order in ORDERS].count(0) == 2
    report = paired_values(samples, "joined_us")
    assert report["exposed"] == dict(mean=1.5, p50=1.5, minimum=-1, maximum=4, second_half_minus_first=3)
    with pytest.raises(ValueError):
        paired_values(samples * 2, "joined_us")


def test_oracle_identifies_hot_source_freed_before_residency_publish():
    flags = np.zeros((1, 16, 32), np.int32)
    flags[0, [0, 12], 1] = 1
    ids = np.zeros((1, 10240), np.int32)
    dst = ids.copy()
    ids[0, [0, 2048]] = [4, 4]
    dst[0, [0, 2048]] = [0, 1]
    before = np.arange(6).reshape(1, 6)
    after = np.array([[1, 4, 2, 3, 0, 5]])
    cleared = []
    verify_permutation(flags, ids, dst, before, after, [0], resident=2, invalidated_slots=cleared)
    assert cleared == [[0]]


def test_completed_get_slots_and_phase_orders():
    posts = np.zeros((1, 4, 32), np.int32)
    posts[0, 1, 0], posts[0, 1, 4] = 2, 4
    assert descriptor_slots(posts, np.array([50, 50, 50, 50, 2, 3]), 4).tolist() == [2, 3]
    ticks = np.zeros((64, 32), np.int64)
    for arm, order in (("get_only", [3, 7, 13]), ("wait", [3, 7, 4, 5, 6, 13]), ("overlap", [3, 4, 5, 6, 7, 13])):
        ticks.fill(0)
        ticks[:, order] = np.arange(1, len(order) + 1)
        assert device_spans(ticks, arm, 1)["joined_us"] == len(order) - 1
        ticks[62, 13] = 1
        with pytest.raises(ValueError):
            device_spans(ticks, arm, 1)


def test_live_tail_orchestration_checks_dirty_output_and_reuses_identical_requests(tmp_path, monkeypatch):
    """Use CPU tensors to exercise scheduling and validation. No hardware timing assertion."""
    import ctypes
    from types import SimpleNamespace as NS

    torch = pytest.importorskip("torch")
    from kvip.diagnostics import frozen_high_tail as tail
    from kvip.diagnostics.high_index_profile import event_work, descriptor_fingerprint
    from kvip.diagnostics.snapshot import persistent_tensors

    b, h, width = 16, 6, 4
    src = torch.zeros((b, 10240), dtype=torch.int32)
    src[:, 2048] = 4
    dst = torch.zeros_like(src)
    flags = torch.zeros((b, 16, 32), dtype=torch.int32)
    flags[:, [0, 12, 13, 14, 15], 0] = 1
    flags[:, 12, 1] = 1
    final = torch.zeros((b, 4, 32), dtype=torch.int32)
    final[:, :, 4] = (torch.arange(64) * 2048).reshape(b, 4)
    final[:, 0, 0] = 1
    indices = torch.arange(b * 8192, dtype=torch.int32)
    dslots = torch.zeros_like(indices)
    dslots[::8192] = torch.arange(b, dtype=torch.int32)
    offset = torch.zeros_like(indices)
    addresses = indices.to(torch.int64) + 100000
    out = torch.arange(b * width, dtype=torch.uint8).reshape(b, width)
    ids = torch.arange(h, dtype=torch.int32).repeat(b, 1)
    rot = NS(
        runtime_shape=NS(tokenBytes=width),
        final_posts=final,
        blocks=indices,
        offsets=offset,
        destinations=dslots,
        fallback_addresses=addresses,
        row_owners=torch.arange(b),
        output=out,
    )
    store = NS(
        keys=(ids + 42).to(torch.uint8),
        scales=(ids + 2).float(),
        addresses=ids.long() + 100,
        logical_ids=ids,
        hot_valid=torch.ones((b, 2), dtype=torch.int32),
    )
    selector = NS(storage=store, rotating=rot)
    persist = persistent_tensors(selector)
    seeds = [(t, t.clone()) for t in persist.values()]

    def apply():
        for name, t in persist.items():
            if name not in ("hot_valid", "output"):
                t[:, [0, 4]] = t[:, [4, 0]]

    apply()
    headers = torch.zeros((b, 32), dtype=torch.int32)
    headers[:, 3] = 1
    fw = NS(
        flags=flags,
        events_src=src,
        events_dst=dst,
        history=h,
        resident=2,
        batch=b,
        workspace=torch.zeros(256, dtype=torch.uint8),
        worker_status=headers,
        _high_index_storage=torch.zeros(128, dtype=torch.uint8),
        post_index=torch.zeros((64, 16), dtype=torch.int64),
        times=torch.zeros(b * 256 + 64 * 32, dtype=torch.int64),
    )

    def reset():
        for t, s in seeds:
            t.copy_(s)

    payload = out.clone()
    bundle = NS(forward=fw, rotating=rot, selector=selector, seeds=seeds, restored={"new_base": 100000}, reset=reset)
    native = NS()
    progress = []

    def fake_launch(cores, stream, arg):
        word = ctypes.c_uint32.from_address(arg + 96).value
        arm = "get_only" if word & 0x800000 else "wait" if word & 0x02000000 else "overlap"
        progress.append(arm)
        assert torch.equal(torch.bitwise_xor(out, 255), payload) and cores == 64
        out.copy_(payload)
        if arm != "get_only":
            apply()
        rec = fw.post_index
        rec[:, 14] = 2
        rec[48:, 1] = rec[48:, 15] = 1
        rec[:, 2] = 2 if arm == "wait" else 0
        if word & 0x80000000:
            ts = fw.times[b * 256 :].reshape(64, 32)
            phases = {"get_only": [3, 7, 13], "wait": [3, 7, 4, 5, 6, 13], "overlap": [3, 4, 5, 6, 7, 13]}[arm]
            ts[:, phases] = torch.arange(1, len(phases) + 1)
            ts[:16, 10] = 2
        return 0

    native.kvip_replay_fixed_tail = fake_launch
    monkeypatch.setattr(tail.ctypes, "CDLL", lambda _: native)

    class Event:
        def __init__(self, **kwargs):
            pass

        def synchronize(self):
            pass

        def record(self, stream):
            pass

        def elapsed_time(self, end):
            return 0.1

    monkeypatch.setattr(torch, "npu", NS(Event=Event), raising=False)
    ref = event_work(flags.numpy(), src.numpy(), dst.numpy(), history=h)
    ref["descriptor_sha256"] = descriptor_fingerprint(
        final.numpy(),
        [indices.numpy(), offset.numpy(), dslots.numpy()],
        addresses.numpy(),
        final_get=16,
        arena_base=100000,
    )
    ref.update(
        final_get_by_core=[0] * 48 + [1] * 16,
        fallback=False,
        moved=[2] * 16 + [0] * 48,
        preplan={"row_pairs": [1] * 16, "row_moves": [2] * 16},
    )
    (tmp_path / "revision.json").write_text('{"revision":"test"}')
    result = tail.measure_frozen(
        torch, NS(npu_stream=1, synchronize=lambda: None), bundle, tmp_path / "lib.so", "test", ref, 1
    )
    assert result["event_count"] == 16 and result["moved"] == 32 and len(progress) == 39
    assert result["joined"]["exposed"]["p50"] == 3
