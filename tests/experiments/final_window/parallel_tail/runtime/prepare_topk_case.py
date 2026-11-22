"""Exercise live final-TopK budget semantics and one padded Q in the simulator."""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_prefix_case import prepare, replace_once
from private_topk import adapt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--inactive-q2', action='store_true')
    args = parser.parse_args()
    destination = args.output.resolve()
    prepare(destination)
    adapt(destination / 'csrc/operators/lightning_indexer_with_fetch/op_kernel')
    kernel = destination / 'kernel.cpp'
    kernel.write_text(replace_once(kernel.read_text(), 'op.TerminalQuery(q);',
                                  'op.TerminalQueryRuntime(q);'))
    driver = destination / 'driver.cpp'
    text = replace_once(driver.read_text(),
                        'c.windowLimit=mode?8192:1024;c.prefetchRankStart=2048;',
                        'c.windowLimit=128;c.prefetchRankStart=1024;')
    # Q3/Q4 each have >32 misses. The final call must not inherit the ordinary
    # window's per-Q limit of 128/4. All 146 unique final misses must be posted.
    if args.inactive_q2:
        text = replace_once(text, 'c.queryEnds=put(std::vector<uint32_t>(4,H));',
                            'c.queryEnds=put(std::vector<uint32_t>{H,0,H,H});')
        marker = ' for(uint32_t q=0;q<4;++q){std::vector<uint32_t> want;'
        text = replace_once(text, marker,
                            ' std::fill(selected.begin()+K,selected.begin()+2*K,UINT32_MAX);\n' + marker)
        text = replace_once(text, '  std::vector<uint32_t> got(selected.begin()+q*K,selected.begin()+(q+1)*K);',
                            '  if(q==1)want.assign(K,UINT32_MAX);\n'
                            '  std::vector<uint32_t> got(selected.begin()+q*K,selected.begin()+(q+1)*K);')
        marker = 'physical=token;if(token>=H){++bad;continue;}'
        text = replace_once(text, marker,
                            'physical=token;if(token==UINT32_MAX){bad+=logicalOut[q*K+i]!=UINT32_MAX||slots[q*K+i]!=UINT32_MAX;continue;}if(token>=H){++bad;continue;}')
    driver.write_text(text)
    (destination / 'runtime_topk_sources.json').write_text(json.dumps({
        'window_limit': 128, 'rank_start': 1024, 'expected_unique_posts': 146,
        'inactive_q2': args.inactive_q2,
        'scope': 'constructed simulator, fake transport, not end-to-end acceptance',
        'sha256': {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                   for name in ('kernel.cpp', 'driver.cpp', 'parallel_post.h', 'emit.h')},
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
