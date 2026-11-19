#!/usr/bin/env python3
"""Three full chains, sharing inputs and serial LIDU/Get; no staged fused candidates."""
import json
import math
import statistics

import full_chain_compare as common


def plan(a, output):
    for job in common.plan(a, output):
        if job['case'] != 'legacy':
            continue
        job['name'] = f"r{job['round']}_pf{job['prefetch']}_three_way"
        job['case'] = 'three_way'
        job['argv'][job['argv'].index('--output') + 1] = str(output / job['name'])
        job['argv'] += ['--compare-optimized-serial', '--serial-history-parts', str(a.serial_history_parts)]
        yield job


def read_run(output, job, a):
    rows = common.read_run(output, job, a)
    for row in rows:
        report = json.loads(common.Path(row['report']).read_text())
        for prefix, name in (('serial_after', 'optimized_native_chain'), ('li_before', 'native_li_only'),
                             ('li_after', 'optimized_li_only')):
            for metric in common.METRICS:
                value = report['timing'][name][metric]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise RuntimeError(f"Missing/invalid {name} in {row['report']}")
                row[f'{prefix}_{metric}'] = value
        geometry = report.get('optimized_serial_geometry')
        if not geometry or not geometry.get('pack_in_timing'):
            raise RuntimeError('Optimized LI adapter scope is missing: ' + row['report'])
        row['optimized_serial_geometry'] = json.dumps(geometry, sort_keys=True)
        misses = report['optimized_serial_misses_per_request']
        if (not isinstance(misses, list) or len(misses) != row['batch']
                or misses != report.get('native_misses_per_request')):
            raise RuntimeError('Serial actual miss work differs: ' + row['report'])
        row['optimized_serial_misses_per_request'] = json.dumps(misses)
        row['optimized_serial_selection'] = json.dumps(report.get('optimized_serial_selection'))
    return rows


def summarize(output, rows):
    common.write_csv(output / 'full_chain.csv', rows)
    groups = {}
    for row in rows:
        if not row['discard']:
            groups.setdefault((row['prefetch'], row['batch'], row['miss_percent']), []).append(row)
    summary = []
    for key, members in sorted(groups.items()):
        if len({r['round'] for r in members}) != len(members):
            raise RuntimeError('Duplicate measured round')
        item = dict(zip(('prefetch', 'batch', 'miss_percent'), key))
        item['rounds'] = len(members)
        for source, name in (('serial', 'serial_before'), ('serial_after', 'serial_after'),
                             ('fused', 'fused_legacy'), ('li_before', 'li_before'), ('li_after', 'li_after')):
            for metric in common.METRICS:
                item[f'{name}_{metric}_round_mean'] = statistics.mean(r[f'{source}_{metric}'] for r in members)
        for name, numerator, denominator in (('serial_after_over_before', 'serial_after', 'serial'),
                                             ('fused_legacy_over_serial_before', 'fused', 'serial'),
                                             ('serial_after_over_fused_legacy', 'serial_after', 'fused'),
                                             ('li_after_over_before', 'li_after', 'li_before')):
            values = [r[f'{numerator}_p50_us'] / r[f'{denominator}_p50_us'] for r in members]
            item[name + '_p50_mean'] = statistics.mean(values)
            item[name + '_p50_per_round'] = json.dumps(values)
        for prefix in ('serial', 'serial_after', 'fused'):
            item[prefix + '_p50_round_stdev_us'] = statistics.stdev(r[prefix + '_p50_us'] for r in members) if len(members) > 1 else 0
        summary.append(item)
    common.write_csv(output / 'summary.csv', summary)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['Three full chains | us | mean of measured-round p50 | r0 discarded | ratios < 1 faster',
             ' pf batch miss% SerialBefore SerialAfter LegacyFused LI_before LI_after after/before after/fused']
    for r in summary:
        lines.append(f"{r['prefetch']:3} {r['batch']:5} {r['miss_percent']:5g} "
            f"{r['serial_before_p50_us_round_mean']:12.2f} {r['serial_after_p50_us_round_mean']:11.2f} "
            f"{r['fused_legacy_p50_us_round_mean']:11.2f} {r['li_before_p50_us_round_mean']:9.2f} {r['li_after_p50_us_round_mean']:8.2f} "
            f"{r['serial_after_over_before_p50_mean']:12.3f} {r['serial_after_over_fused_legacy_p50_mean']:11.3f}")
    (output / 'summary.log').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main(argv=None):
    return common.run_experiment(common.arguments(argv, serial=True), make_plan=plan, read=read_run,
        summary=summarize, prefix='li_serial_vs_fused',
        note='SDK LI -> LIDU/Get; numeric direct LI -> same LIDU/Get; legacy fused. Same original Q/K/scales, restored initial state. Serial unique miss counts checked equal. Numeric LI pack is timed. r0 discarded.')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('ERROR: ' + str(error), file=common.sys.stderr)
        raise SystemExit(1)
