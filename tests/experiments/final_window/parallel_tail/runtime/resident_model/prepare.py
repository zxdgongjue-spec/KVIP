"""Inject the observed resident-source record into the complete native-body model."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from prepare_status_warp_model import prepare
from private_topk import replace_once


def generate(output, pattern="isolated"):
    prepare(output)
    driver = output / 'driver.cpp'
    text = replace_once(driver.read_text(), 'journal.tokens[i]=H-596+i;',
                        'journal.tokens[i]=i==1?5732:H-596+i;')
    marker = ' restore(c.mapping,initialMapping);'
    # First publication only; subsequent replays restore the same lru vector.
    at = text.index(marker)
    text = text[:at] + ' lru[1]=4820;restore(c.lruSlots,lru);\n' + text[at:]
    text = replace_once(text, ' lru[1]=4820;restore(c.lruSlots,lru);',
                        ' lru[1]=4820;restore(c.lruSlots,lru);\n std::vector<uint32_t> inverse(R,UINT32_MAX);for(uint32_t i=0;i<lru.size();++i)inverse[lru[i]]=i;restore(c.reservedHot,inverse);')
    text = replace_once(text, ' printf("audit_index=%zu\\n",bad-checkpoint);checkpoint=bad;',
                        ' bad+=get<uint32_t>(c.hotValid,R)[5732]!=0;\n printf("audit_index=%zu\\n",bad-checkpoint);checkpoint=bad;')
    if pattern in ("chain", "cycle"):
        text = replace_once(text, 'i==1?5732:H-596+i', 'i==1?5732:i==2?4820:H-596+i')
        if pattern == "cycle":
            text = replace_once(text, 'lru[1]=4820;', 'lru[1]=4820;lru[2]=5732;')
        start = text.index(' for(uint32_t token=prior;token<prior+146;++token){if(mapping[token]<R')
        end = text.index(' printf("audit_index=', start)
        text = text[:start] + r"""
 std::vector<uint32_t> sources, destinations;
 for(uint32_t i=0;i<84;++i){sources.push_back(journal.tokens[i]);destinations.push_back(lru[i]);}
 std::vector<uint32_t> ordered(146);
 for(uint32_t token=prior;token<prior+146;++token)ordered[mapping[token]-R-84]=token;
 for(uint32_t i=0;i<146;++i){sources.push_back(ordered[i]);destinations.push_back(lru[84+i]);}
 std::vector<uint32_t> expected(H);std::iota(expected.begin(),expected.end(),0);
 std::vector<uint32_t> displaced,freed;
 for(size_t i=0;i<sources.size();++i){
  expected[destinations[i]]=sources[i];
  if(std::find(sources.begin(),sources.end(),destinations[i])==sources.end())displaced.push_back(destinations[i]);
  if(std::find(destinations.begin(),destinations.end(),sources[i])==destinations.end())freed.push_back(sources[i]);
 }
 for(size_t i=0;i<freed.size();++i)expected[freed[i]]=displaced[i];
 for(uint32_t dst=0;dst<H;++dst){uint32_t src=expected[dst];
  bad+=finalLogical[dst]!=logical[src] || finalAddresses[dst]!=addresses[src] || finalScales[dst]!=scales[src];
  for(uint32_t j=0;j<16;++j)bad+=finalKeys[16*dst+j]!=keys[16*src+j];
 }
 auto validity=get<uint32_t>(c.hotValid,R);
 for(uint32_t dst=0;dst<R;++dst){bool freedHot=std::find(freed.begin(),freed.end(),dst)!=freed.end();bad+=validity[dst]!=uint32_t(!freedHot);}
""" + text[end:]
    driver.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--pattern', choices=('isolated', 'chain', 'cycle'), default='isolated')
    args=parser.parse_args()
    generate(args.output.resolve(), args.pattern)
