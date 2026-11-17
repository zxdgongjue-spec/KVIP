#!/usr/bin/env python3
"""Derive loaded Model kernel PC ranges from ELF and actual entry records."""
import argparse
import gzip
import json
from pathlib import Path
import re
import subprocess
from run import sha
from instructions import analyze

ENTRY=re.compile(r'PC: (0x[0-9a-fA-F]+).*?MOV_XD_SPR\s+XD:X0=.*?SPR:PARA_BASE')

def elf(path,objdump):
    table=subprocess.check_output([objdump,'-t',str(path)],text=True)
    headers=subprocess.check_output([objdump,'-h',str(path)],text=True)
    functions={m[2]:(int(m[0],16),int(m[1],16)) for m in re.findall(r'^([0-9a-f]+)\s+\w+\s+F\s+\.text\s+([0-9a-f]+)\s+(\S+)',table,re.M)}
    text=re.search(r'\s+\.text\s+([0-9a-f]+)\s+',headers)
    if not text or not functions:raise ValueError('unrecognized ELF symbol/section output')
    return functions,int(text[1],16),dict(path=str(path),sha256=sha(path),symbols=table,sections=headers)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--objdump',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.folder=a.folder.resolve();a.build=a.build.resolve()
    manifest=json.loads((a.folder.parent/'manifest.json').read_text())
    row=next(r for r in manifest['rows'] if r['name']==a.folder.name)
    build=json.loads((a.build/'build-manifest.json').read_text())
    if (build.get('status')!='BUILT' or build['binary_sha256']!=manifest['build_manifest']['binary_sha256']
            or sha(a.build/'build/li_qk')!=build['binary_sha256']):
        raise ValueError('PC mapping requires the exact captured build')
    score,ssize,sevidence=elf(a.build/'build/li_score_merge_obj_dir/device.o',a.objdump)
    post,psize,pevidence=elf(a.build/'build/li_post_merge_obj_dir/device_aiv.o',a.objdump)
    logs=sorted(a.folder.glob('*.instr_log.dump'))+sorted(a.folder.glob('*.instr_log.dump.gz'))
    entries={}
    for path in logs:
        values=[];opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',errors='replace') as f:
            for line in f:
                m=ENTRY.search(line)
                if m:values.append(int(m[1],16))
        entries[path.name]=values
    aiv=next((v for k,v in entries.items() if 'veccore0' in k and v),None)
    if not aiv:raise ValueError('AIV kernel entry absent; enable an AIV core in Model log mask')
    balanced=row['schedule']=='balanced';stage=row['stage'];regions=[];bases={}
    if stage in ('all','score'):
        prefix='li_qk_score_balanced_' if balanced else 'li_qk_score_'
        fn=next(n for n in score if n.startswith(prefix) and n.endswith('_mix_aiv') and ('balanced' in n)==balanced)
        bases['score']=aiv[0]-score[fn][0]
        cubes=next((v for k,v in entries.items() if 'cubecore' in k and v),None)
        if cubes:
            cfn=next(n for n in score if n.startswith(prefix) and n.endswith('_mix_aic') and ('balanced' in n)==balanced)
            if cubes[0]-score[cfn][0]!=bases['score']:raise ValueError('AIC/AIV image base mismatch')
        for n,(off,size) in score.items():
            label=('balanced_' if 'balanced' in n else 'score_')+('aiv' if n.endswith('_mix_aiv') else 'aic')
            if not n.startswith('li_qk_score_'):continue
            regions.append(dict(kernel=label,begin=hex(bases['score']+off),end=hex(bases['score']+off+size)))
        last=max(o+n for o,n in score.values())
        regions.append(dict(kernel='shared_score_vf',begin=hex(bases['score']+last),end=hex(bases['score']+ssize),shared_owners=['score_aiv','balanced_aiv']))
    if stage in ('all','scatter','topk'):
        first='li_qk_scatter_' if stage=='scatter' or (stage=='all' and not row['geometry']['canonical_alias']) else 'li_qk_topk_'
        fn=next(n for n in post if n.startswith(first));off=post[fn][0]
        candidates=[v for v in aiv if 'score' not in bases or not bases['score']<=v<bases['score']+ssize]
        if not candidates:raise ValueError('consumer entry absent')
        bases['post']=candidates[0]-off
        for n,(off,size) in post.items():
            label='scatter' if n.startswith('li_qk_scatter_') else 'topk'
            end=psize if label=='topk' else off+size
            regions.append(dict(kernel=label,begin=hex(bases['post']+off),end=hex(bases['post']+end)))
    regions=[r for r in regions if int(r['begin'],16)<int(r['end'],16)]
    result=analyze(a.folder,regions)
    result['capture_status']=row['status']
    result['capture_complete']=row['status']=='MODEL_PASSED' and row.get('exit_code')==0
    result['binary_sha256']=build['binary_sha256']
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'pc-map.json').write_text(json.dumps(regions,indent=2)+'\n')
    (a.output/'pc-map-proof.json').write_text(json.dumps(dict(entries=entries,bases={k:hex(v) for k,v in bases.items()},score=sevidence,post=pevidence),indent=2)+'\n')
    (a.output/'instructions.json').write_text(json.dumps(result,indent=2)+'\n');print(a.output)

if __name__=='__main__':main()
