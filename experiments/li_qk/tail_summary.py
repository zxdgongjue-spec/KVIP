#!/usr/bin/env python3
"""Collect complete post-score Model captures without mixing binary/fixture scopes."""
import argparse
import csv
import json
from pathlib import Path


def collect(folders):
    rows=[];phases=[]
    for folder in folders:
        m=json.loads((folder/'manifest.json').read_text())
        if m['status']!='MODEL_PASSED' or int(m['logged_core_mask'],16)!=0xffffffff:
            raise ValueError('complete all-core Model capture required: '+str(folder))
        binary=m['build_manifest']['binary_sha256'];fixture=m['fixture']['hashes']['oracle']
        for r in m['rows']:
            report=r['report']
            if r['status']!='MODEL_PASSED' or r['exit_code']!=0 or report.get('status')!='PASS' or report.get('tail_p50_us') is not None:
                raise ValueError('failed/incomplete Model row: '+r['name'])
            if not all(report.get(x) for x in ('oracle','guards','inputs_unchanged')) or report.get('fetch_launches')!=0:
                raise ValueError('wrong experiment scope or correctness contract')
            if any(report[k]!=r[k] for k in ('batch','history','resident','parts','resident_parts','stage','cold_parallel','final_split')):
                raise ValueError('report/manifest configuration mismatch')
            item=dict(group=str(folder),control=r['name'].split('_',1)[0],
                **{k:r[k] for k in ('batch','history','resident','parts','resident_parts','stage','whole_model_ticks','issued_instructions')},
                binary_sha256=binary,oracle_sha256=fixture)
            # A combined pair may live in two directories, but must share
            # geometry, exact binary, fixture and stage scope.
            key=tuple(item[k] for k in ('batch','history','resident','parts','resident_parts','stage','binary_sha256','oracle_sha256'))
            item['_pair']=key;rows.append(item)
            evidence=folder/r['name']/'verified-phases/phases.json'
            if evidence.exists():
                data=json.loads(evidence.read_text())
                if sum(x['count'] for x in data['totals'])!=r['issued_instructions']:
                    raise ValueError('phase instruction denominator mismatch')
                for x in data['totals']:
                    phases.append(dict(group=str(folder),control=item['control'],**x))
    for r in rows:
        bases=[x for x in rows if x['_pair']==r['_pair'] and x['control']=='C0']
        if len(bases)>1:raise ValueError('ambiguous C0 baseline')
        r['ticks_over_C0']=r['whole_model_ticks']/bases[0]['whole_model_ticks'] if bases else None
        r['instructions_over_C0']=r['issued_instructions']/bases[0]['issued_instructions'] if bases else None
    for r in rows:del r['_pair']
    return dict(scope='complete dynamic Model captures; ticks are not A5 microseconds; counts are not time shares',rows=rows,phases=phases)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folders',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=collect(a.folders);a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for key in ('rows','phases'):
        if not result[key]:continue
        with (a.output/(key+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(result[key][0]));w.writeheader();w.writerows(result[key])
    print(a.output/'rows.csv')


if __name__=='__main__':main()
