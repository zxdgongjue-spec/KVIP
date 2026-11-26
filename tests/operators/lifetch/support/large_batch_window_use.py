# SPDX-License-Identifier: Apache-2.0
"""Reconcile a single split-window diagnostic with the request's final physical winners."""

import math

import numpy as np


def window_use(events, src, physical_winners, prefetch_rows, clock_mhz):
    sources = np.asarray(src)
    winners = np.asarray(physical_winners)
    totals = np.asarray(prefetch_rows)
    event_words = np.asarray(events, dtype=np.uint64)
    batch = len(totals)
    if (
        not batch or totals.shape != (batch,) or sources.ndim != 2 or len(sources) != batch
        or winners.shape != (batch, 4, 2048) or event_words.size != 5 * batch * 16
        or not math.isfinite(clock_mhz) or clock_mhz <= 0
    ):
        raise ValueError("Invalid five-window diagnostic layout or clock")

    # Events are epoch-major. The journal is request-major, with all prefetch
    # admissions in submission order followed by that request's terminal GETs.
    records = event_words.reshape(5, batch, 16)
    posted = [0] * 5
    useful = [0] * 5
    lead_ticks = 0
    for row in range(batch):
        count = int(totals[row])
        if count < 0 or count > sources.shape[1]:
            raise ValueError("Prefetch count exceeds the request journal")
        prefix = sources[row, :count]
        if np.any(prefix < 0) or len(np.unique(prefix)) != count:
            raise ValueError("Prefetch contains invalid or duplicate physical sources")
        # A GET used by several Qs counts once, just as it does in the SQ.
        used = np.isin(prefix, winners[row].ravel())
        start = 0
        for epoch, record in enumerate(records[:, row]):
            if int(record[2]) >> 32 != epoch or int(record[3]) & 0xffffffff != 1:
                raise ValueError("Missing or incomplete prefetch event")
            end = start + int(record[10])
            if end > count:
                raise ValueError("Window admissions exceed the request's prefetch prefix")
            hits = int(np.count_nonzero(used[start:end]))
            if epoch == 2 and hits:
                last_doorbell = int(record[12])
                next_ready = int(records[3, row, 4])
                if not 0 < last_doorbell <= next_ready:
                    raise ValueError("Missing midpoint doorbell or next window timestamp")
                lead_ticks += hits * (next_ready - last_doorbell)
            posted[epoch] += end - start
            useful[epoch] += hits
            start = end
        if start != count:
            raise ValueError("Window admissions do not cover the request's prefetch prefix")

    return dict(posted_by_epoch=posted, useful_by_epoch=useful,
                useful_mid_lead_to_next_ready_us=lead_ticks / clock_mhz / useful[2] if useful[2] else None)
