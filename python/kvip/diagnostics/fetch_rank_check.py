# SPDX-License-Identifier: Apache-2.0
"""Verify Fetch-side selection and measure actual doorbells outside headline timing."""
import ctypes
import hashlib

import numpy as np

from kvip.diagnostics.archive import cpu_array
from kvip.diagnostics.fetch_window_topk import publication
from kvip.diagnostics.incremental_topk import TOP, eligibility, inputs, scan_windows, select
from kvip.diagnostics.recent_threshold import recent_positions
from kvip.diagnostics.window_audit import CachedArchive


def oracle_row(scores, ends, bounds, resident):
    windows = scan_windows(scores.shape[-1], resident, ends, True)
    table = []
    for q, end in enumerate(ends):
        seed = recent_positions(resident, bounds, int(end))
        state = select(scores[q, seed], seed)
        seeded = np.zeros(scores.shape[-1], dtype=bool)
        seeded[seed] = True
        versions = [state]
        for ids in windows:
            ids = ids[(ids < end) & ~seeded[ids]]
            state = select(np.r_[state[0], scores[q, ids]], np.r_[state[1], ids])
            versions.append(state)
        table.append(versions)
    return table


def check_state(ranks, audit, tables, eligible, *, fixed, actual_posts=None):
    digest = hashlib.sha256()
    sent_digest = hashlib.sha256()
    ranks_checked = 0
    submits = []
    for row, table in enumerate(tables):
        versions = audit[row, :, 0].astype(int)
        if (np.any(np.diff(versions) < 0) or np.any(versions < 0)
                or np.any(versions > np.arange(1, 10))
                or (fixed and not np.array_equal(versions, np.arange(1, 10)))):
            raise ValueError(f"Invalid ready versions row={row}: {versions.tolist()}")
        for q in range(4):
            for epoch in range(versions[-1]):
                r = ranks[row, q, epoch]
                kth, strict = map(int, r[:2])
                scores, ids, cutoff = table[q][epoch]
                if kth != cutoff or strict != int(np.sum(scores > cutoff)):
                    raise ValueError(f"Kth/count mismatch row={row} q={q} epoch={epoch}: {kth}/{strict}")
                positions = np.r_[np.arange(strict), np.arange(TOP-1, strict-1, -1)]
                observed_ids = r[32:32+TOP][positions]
                observed_scores = r[32+TOP:].view('<u2')[positions]
                if not np.array_equal(observed_ids, ids) or not np.array_equal(observed_scores, scores):
                    raise ValueError(f"Ordered score/ID mismatch row={row} q={q} epoch={epoch}")
                digest.update(observed_ids.astype('<u4').tobytes())
                digest.update(observed_scores.astype('<u2').tobytes())
                ranks_checked += 1
        posted = set()
        row_lists = []
        for epoch, version in enumerate(versions):
            lists = []
            if version:
                for q in range(4):
                    ids = table[q][version-1][1]
                    lists.append(ids[eligible[row, ids]].tolist())
            wanted = publication(lists, posted, 64, 512)
            n = int(audit[row, epoch, 1])
            observed = audit[row, epoch, 32:32+n].tolist()
            if n != len(wanted) or observed != wanted:
                raise ValueError(f"Q4 submission mismatch row={row} epoch={epoch} version={version} "
                                 f"counts={n}/{len(wanted)} first={observed[:4]}/{wanted[:4]}")
            if actual_posts is not None and actual_posts[row][epoch] != observed:
                raise ValueError(f"Submitted WQE IDs differ from admission row={row} epoch={epoch}")
            sent_digest.update(np.asarray([n, *observed], dtype='<u4').tobytes())
            row_lists.append(observed)
        submits.append(row_lists)
    return dict(ranks=ranks_checked, lists=len(tables)*9, prefix_sha=digest.hexdigest(),
                post_sha=sent_digest.hexdigest(), versions=audit[:, :, 0].tolist(), posts=submits)


def native_call(library, name, signature, *args):
    fn = getattr(library, name)
    fn.argtypes, fn.restype = signature, ctypes.c_uint32
    rc = fn(*args)
    if rc:
        raise RuntimeError(f"{name} returned {rc}")


def clock_rate(torch, rot, stream):
    # Two durations cancel fixed launch and event overhead. Report a slope only
    # when repeated long/short pairs agree. Never use a hardcoded device clock.
    ptr = ctypes.c_void_p
    buf = torch.zeros(8, dtype=torch.int64, device=rot.device)

    def sample(ticks):
        buf[0] = ticks
        stream.synchronize()
        a, b = [torch.npu.Event(enable_timing=True) for _ in range(2)]
        a.record(stream)
        native_call(rot.library, 'kvip_fetch_clock', [ptr, ptr], buf.data_ptr(), stream.npu_stream)
        b.record(stream)
        b.synchronize()
        return int(buf[1].item()), 1000 * a.elapsed_time(b)

    sample(50000)
    rates = []
    for _ in range(5):
        short, long = sample(50000), sample(500000)
        if long[1] > short[1] and long[0] > short[0]:
            rates.append((long[0]-short[0])/(long[1]-short[1]))
    if len(rates) < 3 or np.ptp(np.percentile(rates, [25, 75])) > .08*np.median(rates):
        raise ValueError(f"Device timer calibration unstable: {rates}")
    return float(np.median(rates))


def stats(values):
    return dict(n=len(values), mean=float(np.mean(values)), p50=float(np.median(values))) if values else dict(n=0)


def posted_ids(torch, rot):
    rows = cpu_array(torch, rot.rows)
    sent = cpu_array(torch, rot.fetch_wqe_ids)
    result = []
    for row, state in enumerate(rows):
        offset = 0
        lists = []
        for epoch, n in enumerate(state[16:25]):
            n = int(n)
            ids = sent[row, epoch, :n].tolist()
            if len(ids) != n or any(v < 0 for v in ids):
                raise ValueError(f"Incomplete posted slot history row={row}")
            lists.append(ids)
            offset += n
        if offset != state[9] or len(set(sum(lists, []))) != offset:
            raise ValueError(f"Duplicate or missing posted slots row={row}")
        result.append(lists)
    return result


def marker_summary(records, posts, final, rate):
    data = np.ascontiguousarray(records).view('<u8').reshape(9, -1, 16)
    first, useful, returned = [], [], []
    for epoch in range(1, 9):
        for lane, record in enumerate(data[epoch]):
            entry, back = map(int, record[13:15])
            if not 0 < entry <= back:
                raise ValueError(f"Missing rotation interval epoch={epoch} lane={lane}")
            returned.append((back-entry)/rate)
            batches = [b for b in range(lane, len(posts), data.shape[1]) if posts[b][epoch]]
            if int(record[10]) != sum(len(posts[b][epoch]) for b in batches) or len(batches) > 3:
                raise ValueError("Unexpected WQE count or batched doorbell geometry")
            if batches:
                t0, t1 = map(int, record[11:13])
                if not entry <= t0 <= t1 <= back:
                    raise ValueError("Posted doorbells lie outside rotation interval")
                first.append((t0-entry)/rate)
                times = ([t0] if len(batches) == 1 else
                         [t0, t1] if len(batches) == 2 else [t0, int(record[15]), t1])
                if times != sorted(times):
                    raise ValueError("Doorbell sequence is not monotonic")
                for j, b in enumerate(batches):
                    if final[b].intersection(posts[b][epoch]):
                        useful.append((times[j]-entry)/rate)
                        break
    return dict(first=stats(first), useful=stats(useful), returned=stats(returned), ticks_per_us=rate)


def inspect_replay(torch, stream, rot, archive):
    posts = posted_ids(torch, rot)
    winners = [set(row.reshape(-1).tolist()) for row in cpu_array(torch, rot.arguments[9])]
    result = dict(markers=marker_summary(cpu_array(torch, rot.events), posts, winners, clock_rate(torch, rot, stream)))
    if not rot.fetch_rank:
        return result
    cpu = CachedArchive(archive.directory, archive.manifest)
    try:
        g, scores, ends, bounds = inputs(cpu)
        eligible = np.stack([eligibility(cpu, row) for row in range(g['batch'])])
        tables = [oracle_row(s, end, bound, g['resident']) for s, end, bound in zip(scores, ends, bounds)]
    finally:
        cpu.close()

    def read(fixed):
        return check_state(cpu_array(torch, rot.fetch_ranks), cpu_array(torch, rot.fetch_audit),
                           tables, eligible, fixed=fixed, actual_posts=None if fixed else posts)

    result['live'] = read(False)
    frozen = torch.from_numpy(np.where(eligible, -1, 0).astype('<i4')).to(rot.device)
    stream.synchronize()
    ptr = ctypes.c_void_p
    native_call(rot.library, 'kvip_fetch_fixture', [ptr, ptr, ptr, ctypes.c_uint32],
                rot.bindings.data_ptr(), frozen.data_ptr(), stream.npu_stream, len(tables))
    stream.synchronize()
    fixed = read(True)
    result['fixed'] = {k: fixed[k] for k in ('ranks', 'lists', 'prefix_sha', 'post_sha')}
    return result
