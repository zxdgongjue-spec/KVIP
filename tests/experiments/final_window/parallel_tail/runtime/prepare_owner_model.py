"""Native model with live owner=1 and a poisoned unused owner=0 bank.

Preserves native kernel code. Audits both selected owner contents and the
untouched bank, WQE output-owner stride, and two graph-equivalent executions.
"""
import argparse
from pathlib import Path

from prepare_native_model import prepare_model
from private_topk import replace_once


def prepare_owner_model(destination):
    prepare_model(destination)
    path = destination / 'driver.cpp'
    text = path.read_text()
    helper = '''template<class T> uint64_t put_owner(const std::vector<T>& value){
 std::vector<T> banks(value.size()*2,T(0x5a));
 std::copy(value.begin(),value.end(),banks.begin()+value.size());return put(banks);
}
template<class T> void check_unused_owner(uint64_t address,size_t count){
 for(auto value:get<T>(address,count))if(value!=T(0x5a))throw std::runtime_error("unused owner bank modified");
}
'''
    text = replace_once(text, 'int main(int argc,char**argv){try{', helper + 'int main(int argc,char**argv){try{')
    text = replace_once(text, 'c.hotValid=put(std::vector<uint32_t>(R,1));',
                        'c.hotValid=put_owner(std::vector<uint32_t>(R,1));c.rowOwners=put(std::vector<uint32_t>{1});')
    text = replace_once(text, 'a.keys=put(keys);a.scales=put(scales);a.logical=put(logical);a.addresses=row.addresses;',
                        'a.keys=put_owner(keys);a.scales=put_owner(scales);a.logical=put_owner(logical);a.addresses=put_owner(addresses);')
    text = replace_once(text, 'c.tokenAddresses=row.addresses;c.addressStride=H;',
                        'c.tokenAddresses=a.addresses;c.addressStride=H;')
    text = replace_once(text, 'physical=(local-args.output)/656;',
                        'physical=(local-args.output)/656-c.outputStride;')
    text = replace_once(text, 'local!=args.output+uint64_t(lru[mapping[token]-R])*656;',
                        'local!=args.output+uint64_t(c.outputStride+lru[mapping[token]-R])*656;')
    for field, dtype, count in (('keys','uint64_t','H*16'),('scales','uint32_t','H'),
                                ('logical','uint64_t','H'),('addresses','uint64_t','H')):
        text = replace_once(text, f'get<{dtype}>(a.{field},{count})',
                            f'get<{dtype}>(a.{field}+uint64_t({count})*sizeof({dtype}),{count})')
        text = replace_once(text, f'restore(a.{field},{field});',
                            f'restore(a.{field}+uint64_t({count})*sizeof({dtype}),{field});')
    audit = ''' check_unused_owner<uint64_t>(a.keys,H*16);
 check_unused_owner<uint32_t>(a.scales,H);
 check_unused_owner<uint64_t>(a.logical,H);
 check_unused_owner<uint64_t>(a.addresses,H);
 check_unused_owner<uint32_t>(c.hotValid,R);
 printf("audit_owner_isolation=0\\n");
'''
    text = replace_once(text, ' auto generations=get<uint32_t>(a.flags,16*32);',
                        audit + ' auto generations=get<uint32_t>(a.flags,16*32);')
    path.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    prepare_owner_model(parser.parse_args().output.resolve())
