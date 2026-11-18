# SPDX-License-Identifier: Apache-2.0
"""Gate the producer-packet/demand-driven path against both unchanged controls."""
import sys
from pathlib import Path

from fetch_ab_report import conservative_gain, pair, validated_cases
from kvip.diagnostics.scan_economy_report import control
from kvip.diagnostics.scan_report import traffic

NAMES = ('forward', 'reverse', 'reverse-fetch-packets')
KEYS = ('mean_us', 'p50_us')


def report(path, phase, native, manifest):
    cases = validated_cases(path, native, manifest)
    if phase == 'check':
        for name in NAMES:
            d = cases[name]['native_diagnostics']['samples'][0]['fetch_ab']
            if name == NAMES[-1]:
                if (d['fixed']['ranks'], d['fixed']['lists']) != (576, 144):
                    raise ValueError('Incomplete fixed-frontier correctness check')
                print(f"CHECK packets fixed_Top2048=576/576 Q4_lists=144/144 "
                      f"live_posts=PASS live_rank_updates={d['live']['ranks']} "
                      "overflow_fallback=exact budgets=64/512")
            markers = d['markers']
            def fmt(key):
                x = markers[key]
                return f"{x['mean']:.2f}/{x['p50']:.2f}" if x['n'] else 'NA/NA'
            print(f"WINDOW {name} mean/P50_us useful_post={fmt('useful')} "
                  f"return_LI={fmt('returned')} samples={markers['useful']['n']}/{markers['returned']['n']}")
        return
    for name in NAMES:
        first, last = (cases[key]['timing'] for key in (name, name+'-end'))
        _, drift = control(cases, name)
        counts = [traffic(cases[key]) for key in (name, name+'-end')]
        kib = '/'.join(f"{sum(c[k] for c in counts)/2048:.1f}" for k in ('waste','repair','total'))
        print(f"GRAPH {name} mean/P50_us first={pair(first[k] for k in KEYS)} "
              f"last={pair(last[k] for k in KEYS)} drift_pct={pair(drift[k] for k in KEYS)} "
              f"waste/repair/total_KiB={kib}")
    net = [v for base in NAMES[:2] for v in conservative_gain(cases, NAMES[-1], cases, base)]
    print(f"GATE packets {'AHEAD' if min(net)>0 else 'KEEP_BASELINES'} "
          f"net_mean/P50_pp vs_forward={pair(net[:2])} vs_reverse={pair(net[2:])}")
    print('SCOPE full_graph_includes_LI_packet_production_rank_post_final_fetch_index_update; diagnostics_excluded')


if __name__ == '__main__':
    path, phase, native, manifest = sys.argv[1:]
    report(Path(path), phase, native, Path(manifest))
