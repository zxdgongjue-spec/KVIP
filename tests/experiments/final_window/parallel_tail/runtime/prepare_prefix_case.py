"""Measure v537 with real-call ordering: prior-prefetch Index is not updated.

Constructed simulator inputs and fake transport remain; this is an integration
regression fixture, not the hardware runner. All 84 prefix and 146 final Index
changes are included in the timed kernel and audited after execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one source site: {old[:100]}')
    return text.replace(old, new)


def prepare(destination):
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here.parent / 'prepare_candidate.py'),
                    'v537', str(destination)], check=True)
    emitter = destination / 'emit.h'
    text = emitter.read_text()
    for number in (0, 1):
        old = f't{number}=s{number};l{number}=asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(logical)+2*t{number});'
        text = replace_once(text, old, '/* Prefix Index is updated only after output readers join. */')
    emitter.write_text(text)
    kernel = destination / 'kernel.cpp'
    text = kernel.read_text()
    text = replace_once(text,
        '// Previous admissions are assumed updated already in this fixture.\n'
        ' uint32_t first=initialCount+(total-initialCount)*q/4,last=initialCount+(total-initialCount)*(q+1)/4,n=last-first;',
        '// Include the actual prior-prefetch Index work, after all output readers join.\n'
        ' uint32_t first=total*q/4,last=total*(q+1)/4,n=last-first;')
    kernel.write_text(text)
    # The standalone v537 audit read the saved TopK prefix. The real caller
    # also consumes ctx->selected, so publish and independently audit that
    # public output instead of silently reusing the saved-prefix measurement.
    post = destination / 'parallel_post.h'
    text = post.read_text()
    text = replace_once(text, 'uint32_t generation,uint32_t history){',
                        'uint32_t generation,uint32_t history,__gm__ uint32_t* selected){')
    text = replace_once(text, '   outputSlots[i]=UINT32_MAX;',
                        '   outputSlots[i]=UINT32_MAX;\n'
                        '   asc_stcg(selected+i,tagged==UINT32_MAX?UINT32_MAX:token);')
    post.write_text(text)
    text = kernel.read_text()
    text = replace_once(text, 'flags,generation,s.history);',
                        'flags,generation,s.history,reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+q*2048);')
    kernel.write_text(text)
    driver = destination / 'driver.cpp'
    text = driver.read_text()
    text = replace_once(text,
        'c.changes=put(std::vector<uint32_t>(10240,UINT32_MAX));c.changeSlots=put(std::vector<uint32_t>(10240,UINT32_MAX));',
        'std::vector<uint32_t> prefixSources(10240,UINT32_MAX),prefixDestinations(10240,UINT32_MAX);\n'
        ' for(uint32_t i=0;i<84;++i){prefixSources[i]=journal.tokens[i];prefixDestinations[i]=lru[i];}\n'
        ' c.changes=put(prefixSources);c.changeSlots=put(prefixDestinations);')
    text = replace_once(text, 'if((mode==8 || mode==9)){\n  for(uint32_t i=0;i<84;++i)',
                        'if(false){ // Preserve pre-update Index inputs in every mode.\n  for(uint32_t i=0;i<84;++i)')
    marker = ' printf("audit_index=%zu\\n",bad-checkpoint);checkpoint=bad;'
    prefix_audit = '''
 for(uint32_t i=0;i<84;++i){
  uint32_t token=journal.tokens[i],physical=lru[i];
  bad+=finalLogical[physical]!=logical[token] || finalLogical[token]!=logical[physical];
  bad+=finalAddresses[physical]!=addresses[token] || finalAddresses[token]!=addresses[physical];
  bad+=finalScales[physical]!=scales[token] || finalScales[token]!=scales[physical];
  for(uint32_t j=0;j<16;++j)
   bad+=finalKeys[physical*16+j]!=keys[token*16+j] || finalKeys[token*16+j]!=keys[physical*16+j];
 }
'''
    text = replace_once(text, marker, prefix_audit + marker)
    marker = ' printf("audit_topk=%zu\\n",bad-checkpoint);checkpoint=bad;'
    text = replace_once(text, marker,
                        ' auto publicSelected=get<uint32_t>(c.selected,8192);\n'
                        ' for(uint32_t i=0;i<8192;++i)bad+=publicSelected[i]!=selected[i];\n' + marker)
    # Repeated execution restores exactly the same unupdated prefix records.
    text = replace_once(text, '   restore(c.mapping,initialMapping);',
                        '   restore(c.mapping,initialMapping);\n'
                        '   restore(c.changes,prefixSources);restore(c.changeSlots,prefixDestinations);')
    driver.write_text(text)
    (destination / 'runtime_prefix_sources.json').write_text(json.dumps({
        'base': 'v537', 'prefix_changes_in_timed_kernel': 84,
        'final_changes': 146, 'real_urma': False,
        'public_selected_output_audited': True,
        'sha256': {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                   for name in ('kernel.cpp', 'driver.cpp', 'emit.h', 'parallel_post.h')},
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.output.resolve())
