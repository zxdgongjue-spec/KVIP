#ifdef NDEBUG
#undef NDEBUG
#endif
#include "init_partition.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace rotating;

template<uint32_t Words>
void CheckPartitions(uint32_t parts) {
    std::vector<uint32_t> writers(Words,0);
    for(uint32_t part=0;part<parts;++part) {
        auto range=PartitionInitWords<Words>(part,parts);
        assert(range.begin<=range.end && range.end<=Words);
        assert(range.begin%32==0 && range.end%32==0);
        for(uint32_t offset=range.begin;offset<range.end;offset+=4096) {
            const uint32_t count=std::min(uint32_t(4096),range.end-offset);
            assert(count && count%8==0); // DataCopy count is in uint32 words.
            for(uint32_t word=offset;word<offset+count;++word)++writers[word];
        }
    }
    assert(std::all_of(writers.begin(),writers.end(),[](uint32_t count){return count==1;}));
}

template<uint32_t History>
void CheckTail(uint32_t parts,uint32_t begin,uint32_t end,uint32_t capacity) {
    constexpr uint32_t resident=16384,slots=8192;
    std::vector<uint32_t> expected(History,UINT32_MAX),actual(History,0xdeadbeef);
    // Reference the existing InitializeTailMappings contract directly: at
    // most output tail capacity consecutive tokens, bounded by H and end.
    if(capacity)for(uint32_t i=0;i<capacity;++i) {
        const uint64_t token=uint64_t(begin)+i;
        if(token<end && token<History)expected[token]=resident+slots+token%capacity;
    }
    std::vector<uint32_t> order(parts);
    std::iota(order.begin(),order.end(),0);
    std::mt19937 rng(History+parts+begin);std::shuffle(order.begin(),order.end(),rng);
    // A shard installs its tail immediately; other shard clears can run later.
    // Disjoint cache-line ownership must preserve the same final mapping.
    for(auto part:order) {
        auto owned=PartitionInitWords<History>(part,parts);
        std::fill(actual.begin()+owned.begin,actual.begin()+owned.end,UINT32_MAX);
        if(!capacity)continue;
        uint64_t capacityEnd=uint64_t(begin)+capacity;
        uint32_t tailEnd=uint64_t(end)>capacityEnd?uint32_t(capacityEnd):end;
        auto tail=InitTailIntersection(owned,begin,tailEnd,History);
        for(uint32_t i=0;i<tail.end-tail.begin;++i) {
            const uint32_t token=tail.begin+i;
            assert(token>=owned.begin && token<owned.end);
            actual[token]=resident+slots+token%capacity;
        }
    }
    assert(actual==expected);
}

template<uint32_t History>
void CheckGeometry(uint32_t parts) {
    CheckPartitions<History>(parts);
    CheckPartitions<8192>(parts);
    CheckPartitions<10240>(parts);
    CheckTail<History>(parts,0,0,0);
    CheckTail<History>(parts,History-256,History,384);
    CheckTail<History>(parts,History-256,History+256,384);
    CheckTail<History>(parts,History,History+512,384);
    CheckTail<History>(parts,UINT32_MAX-128,UINT32_MAX,384);
    // Force an ABC range across every shard boundary, including unaligned ends.
    for(uint32_t part=1;part<parts;++part) {
        uint32_t edge=PartitionInitWords<History>(part,parts).begin;
        if(edge>=128)CheckTail<History>(parts,edge-128,edge+129,384);
    }
}

int main() {
    for(uint32_t batch=1;batch<=16;++batch)for(uint32_t aics=batch;aics<=32;++aics) {
        const uint32_t parts=2*aics/batch,cores=batch*parts;
        assert(parts>=1 && cores<=2*aics);
        std::vector<uint32_t> seen(batch*parts);
        for(uint32_t core=0;core<cores;++core)++seen[(core%batch)*parts+core/batch];
        assert(std::all_of(seen.begin(),seen.end(),[](uint32_t count){return count==1;}));
    }
    // Uneven partitions, B16/AIC32's four writers, and B1's full 64 writers.
    for(uint32_t parts:{1U,2U,3U,4U,5U,21U,32U,64U}) {
        CheckGeometry<8448>(parts);
        CheckGeometry<33024>(parts);
        CheckGeometry<131328>(parts);
        CheckGeometry<210944>(parts);
    }
    std::cout<<"parallel init partitions: coverage, DMA alignment and ABC equivalence PASS\n";
}
