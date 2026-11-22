"""Generate full native-body boundary workloads, retaining the state oracle.

The unmodified implementation is expected to reject counts above 2048. The
generator does not widen device buffers or suppress that error; those changes
belong to the implementation under test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_status_warp_model import prepare
from capacity_codegen import adapt_capacity, bind_model_capacity
from overlap_codegen import adapt_parallel_overlap
from private_topk import replace_once


def generate(output, changes, candidate=False):
    if not (changes == 230 or 2048 <= changes <= 2100 or changes == 4000):
        raise ValueError("Workload requires 230, 2048..2100 or 4000 changes")
    prepare(output)
    driver = output / 'driver.cpp'
    text = driver.read_text()
    # Keep the same 146 final misses and vary prior assignments only. Relocate
    # their source range so every source remains inside the captured history.
    prior = changes - 146 if changes <= 2100 else 1902
    final = changes - prior
    if changes == 4000:
        # Keep the prior mapping journal within its real 2048-entry capacity.
        # Extra traffic comes from four disjoint final Q candidate ranges.
        text=replace_once(text, 'counts[4]={11,0,94,87},begins[4]={0,0,11,59}',
                          'counts[4]={525,525,524,524},begins[4]={0,525,1050,1574}')
    substitutions = {84: prior, 146: final, 230: changes, 596: prior + 512}
    text = re.sub(r'\b(?:84|146|230|596)\b',
                  lambda match: str(substitutions[int(match.group())]), text)
    if candidate:
        adapt_capacity(output)
        text=replace_once(text, 'c.reservedHot=put(std::vector<uint32_t>(R,UINT32_MAX));',
                          'std::vector<uint32_t> initialReserved(R,UINT32_MAX);for(uint32_t i=0;i<lru.size();++i)initialReserved[lru[i]]=i;c.reservedHot=put(initialReserved);')
        text=replace_once(text, ' auto audit=[&](uint32_t completed)->size_t {', ''' auto audit=[&](uint32_t completed)->size_t {
 if(get<uint32_t>(c.reservedHot,R)!=initialReserved)throw std::runtime_error("overflow reservation not retired");
 auto originalLru=lru;
 std::vector<uint32_t> expectedLru=originalLru;
 std::vector<bool> protectedSlots(R,false);
 for(auto slot:originalLru)protectedSlots[slot]=true;
 for(auto token:get<uint32_t>(c.selected,8192))if(token<R && initialMapping[token]==UINT32_MAX)protectedSlots[token]=true;
 for(uint32_t slot=0;slot<R && expectedLru.size()<EXPECTED_CHANGES;++slot)if(!protectedSlots[slot])expectedLru.push_back(slot);
 const auto& lru=expectedLru;
'''.replace('EXPECTED_CHANGES',str(changes)))
        text=replace_once(text, 'physical<10000||physical>=12048',
                          'std::find(lru.begin(),lru.end(),physical)==lru.end()')
        text=replace_once(text, 'mapping[token]=R+physical-10000;',
                          'mapping[token]=R+uint32_t(std::find(lru.begin(),lru.end(),physical)-lru.begin());')
        text=text.replace('mapping[token]>=R+2048',f'mapping[token]>=R+{changes}')
        text=text.replace('ordinal<2048?lru[ordinal]', f'ordinal<{changes}?lru[ordinal]')
    driver.write_text(text)
    if candidate:
        bind_model_capacity(output)
        adapt_parallel_overlap(output)
    # Keep timing evidence tied to the generated device code, even while the
    # candidate generator is edited for subsequent experiments.
    sources = ('kernel.cpp', 'helpers.h', 'parallel_post.h', 'terminal_finish.h',
               'capacity.h', 'overlap_update.h', 'driver.cpp')
    manifest = {
        'changes': changes, 'prior': prior, 'final_unique': final,
        'candidate': candidate,
        'sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                   for name in sources if (output / name).is_file()},
    }
    (output / 'case.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--changes', type=int, required=True)
    parser.add_argument('--candidate', action='store_true')
    args = parser.parse_args()
    generate(args.output.resolve(), args.changes, args.candidate)
