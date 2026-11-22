"""Two simultaneously active requests using the unchanged native kernel.

Each request has independent SQ/CQ state, output routes, Index and scratch.
Reuse the complete single-request oracle for each row after the shared launch.
This test performs one launch; repeated-generation coverage is a separate test.
"""
import argparse
from pathlib import Path
from prepare_native_model import prepare_model
from private_topk import replace_once


def prepare_batch_model(destination):
    prepare_model(destination)
    p = destination / 'driver.cpp'
    text = p.read_text()
    text = replace_once(text, 'int main(int argc,char**argv){try{', '''template<class T> uint64_t duplicate_bank(uint64_t address,size_t count){
 auto first=get<T>(address,count);auto both=first;both.insert(both.end(),first.begin(),first.end());return put(both);
}
int main(int argc,char**argv){try{''')
    text = replace_once(text, 'uint32_t mode=argc>1?std::stoul(argv[1]):0;',
                        'uint32_t mode=1;')
    insertion = '''
 auto multiC=c;auto multiA=a;multiC.shape.batch=2;
 multiC.rowOwners=put(std::vector<uint32_t>{0,1});
'''
    for name, dtype, count in (
        ('selected','uint32_t','8192'),('locations','uint32_t','8192'),
        ('sourceBlocks','uint32_t','8192'),('sourceOffsets','uint32_t','8192'),
        ('mapping','uint32_t','H'),('lruSlots','uint32_t','2048'),
        ('progress','uint32_t','128'),('queryEnds','uint32_t','4'),
        ('hits','uint16_t','4*H/8'),
        ('hotValid','uint32_t','R'),('reservedHot','uint32_t','R'),
        ('mappingJournal','rotating::MappingJournal','1'),
        ('changes','uint32_t','10240'),('changeSlots','uint32_t','10240'),
        ('rows','rotating::Row','1'),('rowStatus','uint32_t','32'),
        ('owners','rotating::FinalPost','4'),('finalStatus','KvafBatchGetCoreStatus','2'),
        ('changeCount','uint32_t','1')):
        insertion += f' multiC.{name}=duplicate_bank<{dtype}>(c.{name},{count});\n'
    for name,dtype,count in (
        ('keys','uint64_t','H*16'),('scales','uint32_t','H'),
        ('logical','uint64_t','H'),('addresses','uint64_t','H'),
        ('queryErrors','uint32_t','128'),('queryBitmaps','uint32_t','2048'),
        ('times','uint64_t','128'),('flags','uint32_t','1024'),
        ('eventsSrc','uint32_t','8192'),('eventsDst','uint32_t','128'),
        ('completedRows','rotating::Row','1')):
        insertion += f' multiA.{name}=duplicate_bank<{dtype}>(a.{name},{count});\n'
    insertion += '''
 multiC.tokenAddresses=multiA.addresses;
 const uint32_t prefixBytes=4*K*6+1024,coveredBytes=(H+255)/256*32;
 std::vector<uint8_t> multiInc(2*(512+prefixBytes+coveredBytes)+4096);
 for(uint32_t b=0;b<2;++b){
  memcpy(multiInc.data()+b*512,inc.data(),512);
  memcpy(multiInc.data()+1024+b*prefixBytes,inc.data()+512,prefixBytes);
 }
 multiC.incremental=put(multiInc);
 auto multiSq=duplicate_bank<uint32_t>(sq,4*8192*16);
 auto multiDoorbell=duplicate_bank<uint64_t>(doorbell,64);
 auto multiQueue=duplicate_bank<uint64_t>(queue,32);
 auto multiEntity=entity;multiEntity.insert(multiEntity.end(),entity.begin(),entity.end());
 for(uint32_t q=0;q<8;++q){multiEntity[q*24+7]=multiSq+uint64_t(q)*8192*64;multiEntity[q*24+8]=multiDoorbell+q*128;}
 auto entityPointer=put(multiEntity);
 auto multiTemplates=templates;multiTemplates.insert(multiTemplates.end(),templates.begin(),templates.end());
 for(uint32_t q=0;q<8;++q){auto e=reinterpret_cast<const uint32_t*>(multiEntity.data())+q*48;
  for(uint32_t j=0;j<4;++j)multiTemplates[q*32+24+j]=e[14+j];
 }
 multiA.eventsDst=put(multiTemplates);
 std::vector<KvafExecuteBatchGetKernelArgs> multiArgs(8,args);
 for(uint32_t q=0;q<8;++q){multiArgs[q].channel_entity=entityPointer+q*192;multiArgs[q].queue_state=multiQueue+q*64;}
 multiC.args=put(multiArgs);
 auto multiOutputSlots=duplicate_bank<uint32_t>(outputSlots,8192);
 auto multiUpdateStatus=duplicate_bank<uint32_t>(updateStatus,1);
 auto multiUpdate=update;multiUpdate[6]=multiUpdateStatus;
 multiUpdate[20]=put(std::vector<int64_t>{3,1,0,2,7,5,4,6});
 multiUpdate[21]=put(std::vector<int64_t>(8,1));
 multiUpdate[22]=multiA.eventsSrc;multiUpdate[23]=multiOutputSlots;
 multiA.updateBindings=put(multiUpdate);
 multiA.context=put(std::vector<rotating::Context>{multiC});
'''
    text = replace_once(text, ' auto deviceArg=put(std::vector<pipeline_ab::Args>{a});',
                        insertion + ' auto deviceArg=put(std::vector<pipeline_ab::Args>{multiA});')
    text = replace_once(text, 'binding.queueArgs=c.args;', 'binding.queueArgs=multiC.args;')
    text = replace_once(text, 'binding.templates=a.eventsDst;binding.flags=a.flags;binding.times=a.times;binding.logical=a.eventsSrc;',
                        'binding.templates=multiA.eventsDst;binding.flags=multiA.flags;binding.times=multiA.times;binding.logical=multiA.eventsSrc;')
    local = ''' auto c=multiC;auto a=multiA;
 auto sq=multiSq+uint64_t(b)*4*8192*64;
 auto doorbell=multiDoorbell+uint64_t(b)*64*8;
 auto queue=multiQueue+uint64_t(b)*32*8;
 auto args=multiArgs[0];args.output+=uint64_t(b)*multiC.outputStride*656;
 auto outputSlots=multiOutputSlots+uint64_t(b)*8192*4;
 auto updateStatus=multiUpdateStatus+b*4;
'''
    for name,dtype,count in (
        ('selected','uint32_t','8192'),('locations','uint32_t','8192'),
        ('mapping','uint32_t','H'),('mappingJournal','rotating::MappingJournal','1'),
        ('rows','rotating::Row','1')):
        local += f' c.{name}+=uint64_t(b)*({count})*sizeof({dtype});\n'
    for name,dtype,count in (
        ('keys','uint64_t','H*16'),('scales','uint32_t','H'),
        ('logical','uint64_t','H'),('addresses','uint64_t','H'),
        ('queryErrors','uint32_t','128'),('times','uint64_t','128'),
        ('flags','uint32_t','1024'),('eventsSrc','uint32_t','8192'),
        ('completedRows','rotating::Row','1')):
        local += f' a.{name}+=uint64_t(b)*({count})*sizeof({dtype});\n'
    text = replace_once(text,' auto audit=[&](uint32_t completed)->size_t {',
                        ' auto audit=[&](uint32_t completed,uint32_t b)->size_t {\n'+local)
    text = replace_once(text, ' auto prefixout=get<uint8_t>(c.incremental,inc.size());',
                        ''' auto prefixout=inc;
 auto prefixData=get<uint8_t>(multiC.incremental+1024+uint64_t(b)*prefixBytes,prefixBytes);
 std::copy(prefixData.begin(),prefixData.end(),prefixout.begin()+512);''')
    text = replace_once(text, 'aclrtlaunch_v9_parallel_terminal(4,stream,(void*)terminal)',
                        'aclrtlaunch_v9_parallel_terminal(8,stream,(void*)terminal)')
    text = replace_once(text, 'if(audit(repeat+1))throw std::runtime_error("per-launch full audit failed");',
                        'for(uint32_t b=0;b<2;++b){printf("request=%u\\n",b);if(audit(repeat+1,b))throw std::runtime_error("per-row full audit failed");}')
    text = replace_once(text, 'get<uint64_t>(a.times,256)', 'get<uint64_t>(a.times,128)')
    p.write_text(text)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    prepare_batch_model(p.parse_args().output.resolve())
