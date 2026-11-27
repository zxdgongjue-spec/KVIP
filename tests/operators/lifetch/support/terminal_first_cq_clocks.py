# SPDX-License-Identifier: Apache-2.0
from math import isfinite


def measure_first(stamps, counts, order, mhz, enabled):
    """Software-unretired first final prefixes of the sixteen initial four-Q teams."""
    if sorted(map(int, order)) != list(range(len(counts))) or not isfinite(mhz) or mhz <= 0:
        raise ValueError("Missing initial team order or invalid clock calibration")
    events, pilot_delays, pilot_polls, partner_dates = [], [], [], []
    first_rows = set(map(int, order[:16]))
    start = min(int(stamps[r, q, 0]) for r in first_rows for q in range(4))
    team_done = [max(int(stamps[r, q, 10]) for q in range(4)) for r in first_rows]
    if start <= 0 or min(team_done) < start:
        raise ValueError("Missing initial team execution timestamps")
    for r, row in enumerate(counts):
        heavy = sum(row) >= 256
        for q, n in enumerate(row):
            if not n:
                continue
            posted, polling, receipt = map(int, stamps[r,q,27:30])
            if not 0 < posted <= polling <= receipt:
                raise ValueError("Incomplete first final CQ clock interval")
            paired = q ^ 1
            pilot = (r ^ q) & 1 == 0
            if enabled and heavy and not pilot and row[paired] and posted < int(stamps[r,paired,29]):
                raise ValueError("Partner issued before its pilot consumed the first CQ")
            if r in first_rows:
                first = (n + 15)//16 if heavy and n > 1 else n
                events.extend(((posted,1,first),(receipt,-1,-first)))
                if heavy and pilot:
                    pilot_delays.append((receipt-posted)/mhz)
                    pilot_polls.append((receipt-polling)/mhz)
                if heavy and not pilot:
                    partner_dates.append((receipt-start)/mhz)
    ledgers = [0,0]
    peaks = [0,0]
    for tick, dq, dn in sorted(events, key=lambda v:(v[0],-v[1])):
        for k, change in enumerate((dq,dn)):
            ledgers[k] += change
            peaks[k] = max(peaks[k],ledgers[k])
    if ledgers != [0,0]:
        raise ValueError("Unbalanced first-prefix clocks")
    return dict(pilot_delays_us=pilot_delays, pilot_poll_us=pilot_polls,
                partner_done_us=partner_dates, first_team_done_us=(min(team_done)-start)/mhz,
                peak_Q=peaks[0], peak_tokens=peaks[1])
