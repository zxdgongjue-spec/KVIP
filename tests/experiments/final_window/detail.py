"""Preserve level-3 device records and print compact, per-role timing tables."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import struct


def save_detail(forward, rotating, valid, output, clock_mhz, terminal=False):
    names = ('times', 'flags', 'query_errors', 'worker_status')
    data = {name: getattr(forward, name).cpu().tolist() for name in names}
    for name in ('events', 'rank_progress', 'final_posts', 'rows', 'locations'):
        data[name] = getattr(rotating, name).cpu().tolist()
    data.update(batch=forward.batch, workers=forward.q_workers,
                resolver_cores=forward.resolver_cores, clock_mhz=clock_mhz,
                service_lanes=rotating.runtime_shape.serviceLanes,
                windows=rotating.rounds, prefetch_tokens=rotating.prefetch_limit,
                prefetch_rank_start=rotating.prefetch_rank_start,
                active_rows=valid.any(dim=1).cpu().tolist(), terminal=terminal)
    path = Path(output) / 'level3.json'
    path.write_text(json.dumps(data))
    return path


def summarize(path):
    d = json.loads(Path(path).read_text())
    b, workers, roles = d['batch'], d['workers'], d['resolver_cores']
    mhz = d['clock_mhz']
    if not mhz or mhz <= 0:
        raise ValueError('Missing measured device-counter calibration')
    times = d['times']
    table = defaultdict(list)
    def span(role, phase, start, end):
        if not start and not end:
            return
        if not start or end < start:
            raise ValueError(f'Invalid {role}/{phase} timestamps: {start}..{end}')
        table[role, phase].append((end-start)/mhz)
    for row in range(b):
        if not d['active_rows'][row]:
            continue
        for q in range(4):
            t = times[b*64+(row*4+q)*16:b*64+(row*4+q+1)*16]
            for start,end,phase in ((0,1,'mapping wait'),(1,2,'barrier'),(2,3,'load'),
                    (3,4,'classify first'),(4,5,'wait second'),(5,6,'classify second'),
                    (10,11,'bitmap/compact miss'),(11,7,'publish')):
                span('final classification',phase,t[start],t[end])
        p = times[row*64:row*64+32]
        u = times[row*64+32:row*64+64]
        f = times[b*128+row*128:b*128+(row+1)*128]
        span('resolver', 'input', p[1], p[2])
        span('resolver', 'initialize', p[2], p[3])
        for q in range(4):
            t = p[10+q*5:15+q*5]
            for i, phase in enumerate(('resolve','output','descriptor','publish')):
                span('resolver', f'Q{q+1}.{phase}', t[i], t[i+1])
        span('resolver', 'prefetch CQ', f[96], f[100])
        for i, phase in enumerate(('update','prefetch','Q1 fetch','Q2 fetch','Q3 fetch','Q4 fetch')):
            span('resolver', f'wait {phase}', f[104+2*i], f[105+2*i])
        span('resolver', 'prefetch residency', f[116], f[117])
        span('resolver', 'residency', f[97], f[98])
        span('resolver', 'status', f[98], f[99])
        for q in range(5):
            span('index', 'apply prefetch' if q == 0 else f'apply Q{q}', u[12+2*q], u[13+2*q])
        span('index', 'metadata', u[22], u[23])
    for block in range(roles*b+4*workers):
        c = times[b*256+block*32:b*256+(block+1)*32]
        if block < roles*b:
            row, role = divmod(block, roles)
            if not d['active_rows'][row]:
                continue
            name = ('resolver','index','output')[role]
            if role == 2:
                for q in range(4):
                    t = c[4+q*6:10+q*6]
                    for i, phase in enumerate(('lookup','wait resolver','pack','output','publish')):
                        span(name,f'Q{q+1}.{phase}',t[i],t[i+1])
        else:
            q, lane = divmod(block-roles*b,workers)
            name = f'fetch Q{q+1}'
            for offset, prefix in ((2,'safe'),(7,'conflict')):
                t = c[offset:offset+5]
                for i, phase in enumerate(('prepare','layout','queue','Get')):
                    span(name,f'{prefix}.{phase}',t[i],t[i+1])
            for i, phase in enumerate(('wait index','residency/ack','diagnostic write')):
                span(name,phase,c[12+i],c[13+i])
        span(name,'core lifetime',c[0],c[1])
    if d['terminal']:
        records = [record for row in d['rank_progress'] for record in row]
        for block, record in enumerate(records):
            for start, end, phase in ((16,20,'TopK through service join'),(16,17,'TopK/classify'),(17,18,'wait TopK barrier'),
                    (18,19,'service phase'),(19,20,'wait service barrier'),
                    (21,22,'admission'),(22,23,'service setup/post'),(23,24,'location patch')):
                if start == 16 and not d['active_rows'][block//4]:
                    continue
                if start in (18,21,22,23) and not any(d['active_rows'][block::d['service_lanes']]):
                    continue
                if start in (18,21,22,23) and block >= d['service_lanes']:
                    continue
                span('terminal',phase,record[start],record[end])
    lines = [f"Effective config: bound BS={b}; active={sum(d['active_rows'])}; windows={d['windows']}; prefetch={d['prefetch_tokens']}; rank start={d['prefetch_rank_start']}",
             'Final detail: one separate instrumented replay; mean/max over recorded cores or active rows, us.',
             'Nested and parallel intervals must not be summed.',
             'Role | Phase | Mean | Max | N']
    for (role, phase), values in table.items():
        lines.append(f'{role} | {phase} | {statistics.mean(values):.2f} | {max(values):.2f} | {len(values)}')
    def flatten(value):
        for item in value:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item
    raw = bytes(flatten(d['events']))
    windows = defaultdict(lambda: defaultdict(list))
    for offset in range(0,len(raw),128):
        lane=(offset//128)%d['service_lanes']
        if not any(d['active_rows'][lane::d['service_lanes']]):
            continue
        begin,end,core,epoch,complete,skipped,*v=struct.unpack_from('<QQIIII12Q',raw,offset)
        if not begin or not complete or skipped:
            continue
        wait,scan,candidate,post,bitmap,candidates,posted,db0,db1,inc0,inc1,counts=v
        if end < begin:
            raise ValueError('Nonmonotonic service timestamps')
        values = dict(service=(end-begin)/mhz,scan=scan/mhz,filter=candidate/mhz,post=post/mhz,posted=posted)
        if inc0 and end >= inc0:
            values['combined']=(end-inc0)/mhz
        if inc0 and inc1 >= inc0:
            values['TopK']=(inc1-inc0)/mhz
        for name,value in values.items():
            windows[epoch][name].append(value)
    if not windows:
        raise ValueError(f'No executed window timestamps in {path}')
    if d['terminal'] and not table.get(('terminal','service setup/post')):
        raise ValueError(f'Missing terminal service stamps in {path}')
    lines += ['Window detail: mean/max over executed lanes, us; posted is summed.',
              'Epoch | TopK through service | TopK | Service | Scan | Filter | Post | Posted']
    for epoch, values in sorted(windows.items()):
        cells=[]
        for name in ('combined','TopK','service','scan','filter','post'):
            v=values[name]
            cells.append(f'{statistics.mean(v):.2f}/{max(v):.2f}' if v else '--')
        lines.append(f'{epoch} | '+ ' | '.join(cells)+f" | {sum(values['posted'])}")
    text = '\n'.join(lines)
    Path(path).with_name('final_detail.log').write_text(text+'\n')
    print(text)
    return table


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', nargs='+', type=Path)
    for path in parser.parse_args().records:
        print(f'Source: {path}')
        summarize(path)
