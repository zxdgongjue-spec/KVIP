"""Same-replay logical AIV timelines; no combining maxima from different cores."""
import argparse
import json
from pathlib import Path


def decode(raw):
    batch, workers = raw['batch'], raw['workers_per_query']
    resolver_cores = raw.get('resolver_cores', 2)
    cores, pu, fetch = raw['cores'], raw['producer_update'], raw['fetch']
    mhz = raw['clock_mhz']
    if not mhz or mhz <= 0 or len(cores) != resolver_cores * batch + 4 * workers:
        raise ValueError('missing clock calibration or core records')
    if any(not c[0] or c[1] < c[0] for c in cores):
        raise ValueError('missing/nonmonotonic core entry or exit marker')
    origin = min(c[0] for c in cores)
    result = []
    for block, c in enumerate(cores):
        stages = []
        def add(name, start, end):
            start, end = int(start), int(end)
            if not start or not end or end < start or start < c[0] or end > c[1]:
                raise ValueError(f'block={block} invalid {name}: {start}..{end}, core={c[:2]}')
            stages.append(dict(name=name, start_cycles=start, end_cycles=end,
                               start_us=(start-origin)/mhz, end_us=(end-origin)/mhz,
                               duration_us=(end-start)/mhz))
        if block < resolver_cores * batch:
            b = block // resolver_cores
            p, u, f = pu[b][:32], pu[b][32:], fetch[b]
            if block % resolver_cores == 0:
                role = f'resolver+finalize b{b}'
                add('input_and_bitmap', p[1], p[2])
                add('initialize', p[2], p[3])
                for q in range(4):
                    t = p[10+q*5:15+q*5]
                    for i, name in enumerate(('resolve','output','descriptor','publish')):
                        add(f'Q{q+1}.{name}', t[i], t[i+1])
                add('prefetch_CQ_join', f[96], f[100])
                for i, name in enumerate(('index_update','prefetch_ready','Q1_fetch','Q2_fetch','Q3_fetch','Q4_fetch')):
                    add('wait.'+name, f[104+2*i], f[105+2*i])
                if f[116]: add('prefetch_residency', f[116], f[117])
                add('residency', f[97], f[98])
                add('status', f[98], f[99])
            elif block % resolver_cores == 1:
                role = f'index_update b{b}'
                add('initialize_and_wait_prefetch', u[10], u[12])
                for q in range(5):
                    if q: add(f'wait.Q{q}', u[11+2*q], u[12+2*q])
                    add('apply.prefetch' if q == 0 else f'apply.Q{q}', u[12+2*q], u[13+2*q])
                add('metadata', u[22], u[23])
            else:
                role = f'output_pack b{b}'
                add('wait.prefetch_publication', c[0], c[2])
                add('input_and_output_masks', c[2], c[3])
                for q in range(4):
                    t = c[4+q*6:10+q*6]
                    for i, name in enumerate(('logical_lookup', 'wait.resolver', 'locations_and_pack', 'output_and_sync', 'publish')):
                        add(f'Q{q+1}.{name}', t[i], t[i+1])
        else:
            q, lane = divmod(block-resolver_cores*batch, workers)
            role = f'fetch Q{q+1} lane{lane}'
            for phase, name in enumerate(('safe','conflict')):
                t = c[2+phase*5:7+phase*5]
                if not t[0]:
                    if phase == 0 and lane < batch: raise ValueError(f'block {block}: missing safe phase')
                    continue
                for i, step in enumerate(('prepare','input_layout','queue_setup','KVAF_Get')):
                    add(f'{name}.{step}', t[i], t[i+1])
            if c[12]:
                add('finalize.wait_index', c[12], c[13])
                add('finalize.residency_and_ack', c[13], c[14])
                add('finalize.diagnostic_write', c[14], c[15])
            for b in range(lane, batch, workers):
                ft = fetch[b][q*24:q*24+24]
                add(f'request{b}.wait_Q_ready', ft[0], ft[1])
                add(f'request{b}.metadata', ft[1], ft[2])
                published = pu[b][14+q*5]
                if ft[1] < published:
                    raise ValueError('cross-core timestamps violate producer publication order')
            # Shared group counters are copied to each assigned request; report once.
            if lane < batch:
                f = fetch[lane][q*24:q*24+24]
                counters = dict(safe_post_cycles=f[12], safe_CQ_cycles=f[13],
                                conflict_post_cycles=f[14], conflict_CQ_cycles=f[15])
            else:
                counters = {}
        intervals = sorted((s['start_cycles'], s['end_cycles']) for s in stages)
        cursor = c[0]
        gaps = []
        for start, end in intervals:
            if start > cursor: gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < c[1]: gaps.append((cursor, c[1]))
        for i, (start, end) in enumerate(gaps): add(f'control_gap{i}', start, end)
        stages.sort(key=lambda x: (x['start_cycles'], x['end_cycles']))
        item = dict(block=block, role=role, start_cycles=c[0], end_cycles=c[1],
                    start_us=(c[0]-origin)/mhz, end_us=(c[1]-origin)/mhz,
                    duration_us=(c[1]-c[0])/mhz, stages=stages)
        if block >= resolver_cores*batch:
            item['KVAF_counters'] = counters
            item['requests'] = list(range(lane, batch, workers))
            item['published_misses'] = [raw['flags'][b][q+1][1] for b in item['requests']]
        result.append(item)
    return dict(origin_cycles=origin, clock_mhz=mhz, cores=result,
                last_block=max(result, key=lambda x:x['end_cycles'])['block'],
                observed_span_us=(max(c[1] for c in cores)-origin)/mhz)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    a = p.parse_args()
    files = sorted(a.root.rglob('full_attribution.json'))
    if not files: raise SystemExit('FAIL: no full_attribution.json')
    output, lines = [], ['Same-replay logical AIV blocks; us from calibrated device counter.',
        't=0 is earliest instrumented core entry. Entry/exit exclude marker flush and kernel launch/retirement.',
        'Wait intervals show observed waiting, including polling; do not assume pure network time.']
    for path in files:
        samples = json.loads(path.read_text())['samples']
        if not samples: raise ValueError(f'no samples: {path}')
        for sample in samples:
            raw = sample['streaming_cycles'][0]['core_timeline_raw']
            timeline = decode(raw)
            timeline.update(source=str(path), round=sample['round'])
            output.append(timeline)
            last = timeline['cores'][timeline['last_block']]
            kernel = sample.get('accounting', {}).get('stages_us', {}).get('resolve_update')
            lines.append(f"ROUND {sample['round']} source={path.parent} profiler_kernel_us={kernel} observed_span_us={timeline['observed_span_us']:.3f} last_block={last['block']} ({last['role']})")
            lines.append('block | role | start_us | end_us | duration_us')
            for c in timeline['cores']:
                lines.append(f"{c['block']} | {c['role']} | {c['start_us']:.3f} | {c['end_us']:.3f} | {c['duration_us']:.3f}")
            lines.append('LAST CORE stages: name | start_us | end_us | duration_us')
            for s in last['stages']:
                lines.append(f"{s['name']} | {s['start_us']:.3f} | {s['end_us']:.3f} | {s['duration_us']:.3f}")
    (a.root/'core_timeline.json').write_text(json.dumps(output, indent=2))
    detailed = []
    for r in output:
        detailed.append(f"ROUND {r['round']} source={r['source']}")
        for c in r['cores']:
            detailed.append(f"block={c['block']} {c['role']} {c['start_us']:.3f}..{c['end_us']:.3f} us")
            for s in c['stages']:
                detailed.append(f"  {s['name']}: {s['start_us']:.3f}..{s['end_us']:.3f} duration={s['duration_us']:.3f} us")
            if 'published_misses' in c: detailed.append('  requests/misses: '+str(list(zip(c['requests'], c['published_misses']))))
            if 'KVAF_counters' in c: detailed.append('  '+json.dumps(c['KVAF_counters']))
    (a.root/'core_timeline_details.log').write_text('\n'.join(detailed)+'\n')
    text='\n'.join(lines)+'\n'
    (a.root/'core_timeline_summary.log').write_text(text)
    print(text, end='')
    print(f"Full per-core stages: {a.root/'core_timeline_details.log'}")


if __name__ == '__main__':
    main()
