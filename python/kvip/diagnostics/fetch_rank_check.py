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


class RankMismatch(ValueError):
    def __init__(self, row, query, epoch, actual, expected, cutoff, expected_ids):
        self.where = (row, query, epoch)
        kth, strict = map(int, actual[:2])
        ids = actual[32:32+TOP] & np.uint32(0x7fffffff)
        missing = np.setdiff1d(expected_ids, ids)
        extra = np.setdiff1d(ids, expected_ids)
        super().__init__(f"rank row={row} q={query} epoch={epoch} "
                         f"kth={kth}/{cutoff} strict={strict}/{expected} "
                         f"missing={len(missing)}:{missing[:6].tolist()} "
                         f"extra={len(extra)}:{extra[:6].tolist()}")


def bitmap_evidence(words, row_scores, ends, bounds, resident, query, epoch, cutoff):
    """Compare completed producer bits with the scores in physical Index order."""
    if epoch == 0:
        return "seed"
    history = row_scores.shape[-1]
    physical = scan_windows(history, resident, ends, True)[epoch-1]
    logical = scan_windows(history, resident, ends, False)[epoch-1]
    valid = ((physical < ends[query]) &
             ((physical < bounds[0]) | (physical >= bounds[1])))
    wanted = (row_scores[query, physical] > cutoff) & valid
    stored = ((words[logical//16] >> ((logical % 16)*2)) & 1).astype(bool)
    wanted_tiles = wanted.reshape(-1, 128).any(axis=1)
    stored_tiles = (words.reshape(-1, 8)[logical[::128]//128] != 0).any(axis=1)
    lost = wanted_tiles & ~stored_tiles
    return (f"completed_bitmap missing_bits={np.count_nonzero(wanted & ~stored)} "
            f"missing_nonempty_tiles={np.count_nonzero(lost)} "
            f"first_physical_tiles={(physical[::128][lost][:6]//128).tolist()}")


def check_state(ranks, all_versions, posts, tables, eligible, *, fixed, lazy=False):
    digest = hashlib.sha256()
    sent_digest = hashlib.sha256()
    ranks_checked = 0
    submits = []
    for row, table in enumerate(tables):
        versions = all_versions[row].astype(int)
        if (np.any(np.diff(versions) < 0) or np.any(versions < 0)
                or np.any(versions > np.arange(1, 10))
                or (fixed and not np.array_equal(versions, np.arange(1, 10)))):
            raise ValueError(f"Invalid ready versions row={row}: {versions.tolist()}")
        for q in range(4):
            for epoch in range(versions[-1]):
                r = ranks[row, q, epoch]
                if lazy and r[2] == 0:
                    continue
                if lazy and r[2] != epoch+1:
                    raise ValueError("Invalid lazy rank validity marker")
                kth, strict = map(int, r[:2])
                scores, ids, cutoff = table[q][epoch]
                expected = int(np.count_nonzero(scores > cutoff))
                if (kth, strict) != (cutoff, expected):
                    raise RankMismatch(row, q, epoch, r, expected, cutoff, ids)
                positions = np.r_[np.arange(strict), np.arange(TOP-1, strict-1, -1)]
                observed_ids = r[32:32+TOP][positions] & np.uint32(0x7fffffff)
                observed_scores = r[32+TOP:].view('<u2')[positions]
                if not np.array_equal(observed_ids, ids) or not np.array_equal(observed_scores, scores):
                    raise RankMismatch(row, q, epoch, r, expected, cutoff, ids)
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
            observed = posts[row][epoch]
            n = len(observed)
            if n != len(wanted) or observed != wanted:
                raise ValueError(f"Q4 submission mismatch row={row} epoch={epoch} version={version} "
                                 f"counts={n}/{len(wanted)} first={observed[:4]}/{wanted[:4]}")
            sent_digest.update(np.asarray([n, *observed], dtype='<u4').tobytes())
            row_lists.append(observed)
        submits.append(row_lists)
    return dict(ranks=ranks_checked, lists=len(tables)*9, prefix_sha=digest.hexdigest(),
                post_sha=sent_digest.hexdigest(), versions=all_versions.tolist(), posts=submits)


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
    return dict(n=len(values), mean=float(np.mean(values)), p50=float(np.median(values))) if len(values) else dict(n=0)


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

    # Compare the actual WQEs with the independent oracle. The version rides
    # in the normal row flush so timing never waits for an extra admission dump.
    versions = cpu_array(torch, rot.rows)[:, 80:116:4]
    # The fixture overwrites the rank bank. Check the live result first and
    # retain its failure while checking the SAME device-produced inputs at rest.
    def evidence(exc):
        if not isinstance(exc, RankMismatch) or not rot.library.kvip_fetch_bitmap() or getattr(rot, "fetch_packets_enabled", False):
            return str(exc)
        b, q, epoch = exc.where
        history = g['history']
        offset = len(tables)*(4*TOP*6+1024) + (b*4+q)*(history//4)
        words = cpu_array(torch, rot.arguments[13].view(torch.uint8).reshape(-1)[
            offset:offset+history//4]).view('<u4')
        return f"{exc} {bitmap_evidence(words, scores[b], ends[b], bounds[b], g['resident'], q, epoch, tables[b][q][0][2])}"

    live_ranks = cpu_array(torch, rot.fetch_ranks)
    profile = getattr(rot.library, 'kvip_fetch_window_rank', None)
    if profile and profile():
        samples = [live_ranks[b, q, e, :32] for b in range(len(tables))
                   for q in range(4) for e in range(1, int(versions[b, -1]))
                   if not getattr(rot, 'fetch_packets_enabled', False) or live_ranks[b,q,e,2] == e+1]
        if samples:
            headers = np.ascontiguousarray(samples)
            times = headers.view('<u8')[:, 2:7] / result['markers']['ticks_per_us']
            result['rank_stages'] = dict(
                scope='instrumented_per_query_cold_window_excludes_post',
                load=stats(times[:, 0]), scan_collect=stats(times[:, 1]), merge=stats(times[:, 2]),
                store=stats(times[:, 3]), admit=stats(times[:, 4]),
                blocks=int(headers[:, 18].sum()), candidates=int(headers[:, 19].sum()),
                merges=int(headers[:, 20].sum()))
    live_error = None
    try:
        result['live'] = check_state(live_ranks, versions, posts, tables, eligible, fixed=False,
                                     lazy=getattr(rot, 'fetch_packets_enabled', False))
    except ValueError as exc:
        live_error = evidence(exc)
    frozen = torch.from_numpy(np.where(eligible, -1, 0).astype('<i4')).to(rot.device)
    stream.synchronize()
    ptr = ctypes.c_void_p
    native_call(rot.library, 'kvip_fetch_fixture', [ptr, ptr, ptr, ctypes.c_uint32],
                rot.bindings.data_ptr(), frozen.data_ptr(), stream.npu_stream, len(tables))
    stream.synchronize()
    audit = cpu_array(torch, rot.fetch_audit)
    if np.any(audit[:, :, 1] > 64) or np.any(audit[:, :, 1] < 0):
        raise ValueError("Fixture exceeded the window budget")
    lists = [[a[32:32+int(a[1])].tolist() for a in row] for row in audit]
    try:
        fixed = check_state(cpu_array(torch, rot.fetch_ranks), audit[:, :, 0], lists, tables, eligible, fixed=True)
    except ValueError as exc:
        raise ValueError(f"live={live_error or 'PASS'}; frozen={evidence(exc)}") from exc
    if live_error:
        raise ValueError(f"live={live_error}; frozen=PASS ({fixed['ranks']} ranks, {fixed['lists']} lists)")
    result['fixed'] = {k: fixed[k] for k in ('ranks', 'lists', 'prefix_sha', 'post_sha')}
    return result
