#!/usr/bin/env python3
"""Summarize single-score Model captures; issue windows are not A5 kernel time."""
import argparse
import csv
import json
from pathlib import Path
import re


def summarize(folders):
    cases=[];cores=[]
    for folder in folders:
        folder=Path(folder)
        m=json.loads((folder/'manifest.json').read_text())
        if len(m['rows'])!=1:raise ValueError('one isolated score capture required')
        r=m['rows'][0];g=r['geometry']
        if r['stage']!='score' or r['status']!='MODEL_PASSED' or r['exit_code']!=0:
            raise ValueError('incomplete/failed/non-score capture')
        if int(m['logged_core_mask'],16)!=0xffffffff:raise ValueError('all physical groups must be logged')
        s=json.loads((folder/r['name']/'verified-kernel-instructions/instructions.json').read_text())
        if not s.get('capture_complete'):raise ValueError('unverified capture')
        row=dict(case=folder.name,history=m['history'],work_grain=g['work_grain'],schedule=r['schedule'],
                 whole_model_ticks=r['whole_model_ticks'],active_segments=g['active_segments'],
                 score_launches=g['score_launches'],issue_window_scope='first to last logged issue; includes gaps; not active cycles or completion time')
        own=[]
        for c in s['coverage']:
            if not c['instructions']:continue
            match=re.fullmatch(r'core(\d+)\.(cube|vec)core([01])',c['core'])
            if not match:raise ValueError('unknown physical core identity')
            core=int(match[1]);kind='AIC' if match[2]=='cube' else 'AIV'
            known=set(c['kernel_counts'])-{'unmapped'}
            if len(known)!=1 or not known<=({'balanced_aic','balanced_aiv'} if r['schedule']=='balanced' else {'score_aic','score_aiv'}):
                raise ValueError('multiple/unknown kernels in issue window')
            if r['schedule']=='balanced':tasks=g['task_map'][core::g['capacity']]
            else:
                # This experiment's waves captures must be single-launch T1.
                if g['score_launches']!=1:raise ValueError('multiple waves require per-launch boundaries')
                tasks=g['task_map'][core:core+1]
            work=sum(t['end']-t['begin'] for t in tasks)
            if not work:raise ValueError('instructions on an unassigned core')
            span=c['last_issue_tick']-c['first_issue_tick']
            item=dict(case=folder.name,history=m['history'],work_grain=g['work_grain'],core=c['core'],kind=kind,
                      history_work=work,init_count=sum(t['end']>t['begin'] for t in tasks),
                      issued_instructions=c['instructions'],first_issue_tick=c['first_issue_tick'],last_issue_tick=c['last_issue_tick'],
                      issue_span_ticks=span,issue_span_per_128_tokens=span/(work/128),
                      instructions_per_128_tokens=c['instructions']/(work/128),unparsed_lines=c['unparsed_lines'])
            cores.append(item);own.append(item)
        expected_aic=g['capacity'] if r['schedule']=='balanced' else g['tasks']
        if sum(x['kind']=='AIC' for x in own)!=expected_aic or sum(x['kind']=='AIV' for x in own)!=2*expected_aic:
            raise ValueError('incomplete core coverage')
        row.update(max_core_history=max(x['history_work'] for x in own),min_core_history=min(x['history_work'] for x in own),
                   history_load_efficiency=m['batch']*m['history']/(expected_aic*max(x['history_work'] for x in own)),
                   max_aic_issue_span=max(x['issue_span_ticks'] for x in own if x['kind']=='AIC'),
                   max_aiv_issue_span=max(x['issue_span_ticks'] for x in own if x['kind']=='AIV'),
                   all_core_issue_envelope=max(x['last_issue_tick'] for x in own)-min(x['first_issue_tick'] for x in own))
        cases.append(row)
    return cases,cores


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folders',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cases,cores=summarize(a.folders);a.output.mkdir(parents=True,exist_ok=False)
    for name,rows in (('cases',cases),('cores',cores)):
        (a.output/(name+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
        with (a.output/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(a.output)


if __name__=='__main__':main()
