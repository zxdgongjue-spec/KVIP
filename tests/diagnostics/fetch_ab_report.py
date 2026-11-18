# SPDX-License-Identifier: Apache-2.0
"""Condensed real-hardware gate for the fixed URMA Fetch-rank experiment."""
import hashlib
import json
import sys
from pathlib import Path

from kvip.diagnostics.scan_economy_report import control
from kvip.diagnostics.scan_options import options_for

NAMES = ('forward', 'reverse', 'reverse-fetch-bitmap', 'reverse-fetch-dense')
KEYS = ('mean_us', 'p50_us')


def pair(values):
    return '/'.join(f'{float(v):+.2f}' for v in values)


def report(path, phase, native, manifest):
    data = json.loads(path.read_text())
    if data['status'] != 'pass':
        raise ValueError(f'Replay incomplete: {path}')
    cases = data['cases']
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for name, case in cases.items():
        opts, g, t = options_for(name), case['geometry'], case['tuning']
        if (case['snapshot_sha256'] != digest or case['production_source']['commit'] != native
                or (g['batch'], g['token_bytes'], g['rounds']) != (16, 656, 9)
                or (t['window_limit'], t['prefetch_limit']) != (64, 512)
                or any(t.get(key, False) != opts.get(key, False)
                       for key in ('reverse_cold', 'fetch_bitmap', 'fetch_dense'))):
            raise ValueError(f'Wrong source, input, geometry or compile settings: {name}')
    if phase == 'check':
        details = {n: cases[n]['native_diagnostics']['samples'][0]['fetch_ab'] for n in NAMES}
        a, b = (details[n] for n in NAMES[-2:])
        if a['fixed'] != b['fixed'] or (a['fixed']['ranks'], a['fixed']['lists']) != (576,144):
            raise ValueError('A/B fixed-frontier rank or Q4 submission differs')
        cmp = matches = raw_equal = 0
        for av, bv, ap, bp in zip(a['live']['versions'], b['live']['versions'],
                                   a['live']['posts'], b['live']['posts']):
            same_history = True
            for x, y, u, v in zip(av, bv, ap, bp):
                same_history &= x == y
                raw_equal += u == v
                cmp += 1
                matches += same_history
                if same_history and u != v:
                    raise ValueError('Live Q4 posts differ at identical progress and admission history')
        print(f'CHECK fixed_ordered_Top2048=576/576 fixed_Q4_lists=144/144 '
              f'live_vs_own_oracle=PASS live_equal_lists={raw_equal}/{cmp} '
              f'live_same_progress_history={matches}/{cmp}')
        for name in NAMES:
            m = details[name]['markers']
            def fmt(key):
                x = m[key]
                return f"{x['mean']:.2f}/{x['p50']:.2f}" if x['n'] else 'NA/NA'
            print(f"WINDOW {name} mean/P50_us useful_post={fmt('useful')} "
                  f"return_LI={fmt('returned')} nonempty_post={fmt('first')} "
                  f"useful/nonempty/all={m['useful']['n']}/{m['first']['n']}/{m['returned']['n']}")
        return
    for name in NAMES:
        a, b = (cases[n]['timing'] for n in (name, name+'-end'))
        _, drift = control(cases, name)
        print(f'GRAPH {name} mean/P50_us run1={pair(a[k] for k in KEYS)} '
              f'run2={pair(b[k] for k in KEYS)} drift_pct={pair(drift[k] for k in KEYS)}')
    for name in NAMES[-2:]:
        _, own = control(cases, name)
        net = []
        for base in NAMES[:2]:
            best, drift = control(cases, base)
            for k in KEYS:
                slow = max(cases[n]['timing'][k] for n in (name,name+'-end'))
                net.append(100*(1-slow/best[k])-max(abs(own[k]),abs(drift[k])))
        print(f"GATE {name} {'AHEAD' if min(net)>0 else 'KEEP_BASELINES'} "
              f"net_mean/P50_pp vs_forward={pair(net[:2])} vs_reverse={pair(net[2:])}")


if __name__ == '__main__':
    path, phase, native, manifest = sys.argv[1:]
    report(Path(path), phase, native, Path(manifest))
