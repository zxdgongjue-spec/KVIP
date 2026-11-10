# SPDX-License-Identifier: Apache-2.0
"""Execute the production TND output publisher under its bounded AIV schedule."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("batch,aics", ((1, 1), (1, 32), (2, 32), (16, 16), (16, 32)))
def test_output_rows_preserve_mapping_and_have_one_writer(tmp_path, batch, aics):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the output publication model")
    root = Path(__file__).resolve().parents[3]
    kernel = root / "csrc/operators/lightning_indexer_with_fetch_ubmem/op_kernel/index_cache_update.cpp"
    text = kernel.read_text()
    start = text.index("__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishOutputs")
    brace = text.index("{", start)
    end, depth = brace + 1, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    function = text[start:end]
    source = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#define __simt_vf__
#define __aicore__
#define __gm__
#define LAUNCH_BOUND(x)
struct {uint32_t x;} threadIdx;
namespace rotating {
constexpr uint32_t kOutputCores=4*SIM_BATCH<2*V9_AIC_COUNT?4*SIM_BATCH:2*V9_AIC_COUNT;
}
constexpr uint32_t rows=SIM_BATCH*4,top=2048,history=128;
uint32_t currentCore=0;
int32_t* outputBase=nullptr;
std::vector<uint32_t> writes(2*rows*top),writers(2*rows*top);
template<class T>T asc_ldcg(T* p) {return *p;}
void asc_stcg(int32_t* p,int32_t value) {
    auto index=p-outputBase;
    assert(index>=0 && uint64_t(index)<writes.size());
    assert(++writes[index]==1);writers[index]=currentCore;*p=value;
}
FUNCTION
int main() {
    std::vector<int32_t> topk(rows*top),slots(rows*top),output(2*rows*top+32,9191);
    std::vector<int64_t> logical((SIM_BATCH+1)*history),unpack(rows),valid(rows);
    std::vector<uint32_t> owners(SIM_BATCH),status(SIM_BATCH);
    outputBase=output.data();
    for(uint32_t b=0;b<SIM_BATCH;++b)owners[b]=SIM_BATCH-b;
    for(uint32_t owner=0;owner<=SIM_BATCH;++owner)
        for(uint32_t token=0;token<history;++token)logical[owner*history+token]=owner*100000+token;
    for(uint32_t row=0;row<rows;++row)for(uint32_t k=0;k<top;++k) {
        topk[row*top+k]=k==0?-1:k==1?history:int32_t((k*17+row)%history);
        slots[row*top+k]=(k*31+row*17)%16384;
    }
    uint64_t a[28]{};
    a[3]=uint64_t(logical.data());a[6]=uint64_t(status.data());a[9]=uint64_t(owners.data());
    a[18]=uint64_t(topk.data());a[19]=uint64_t(slots.data());
    a[20]=uint64_t(unpack.data());a[21]=uint64_t(valid.data());
    a[22]=uint64_t(output.data());a[23]=uint64_t(output.data()+rows*top);a[24]=history;
    for(uint32_t trial=0;trial<4;++trial) {
        for(uint32_t row=0;row<rows;++row) {
            unpack[row]=(row*3+5)%rows; // TND rows span/reorder packed batches.
            valid[row]=trial==2?0:trial==1?row%3!=0:1;
        }
        for(uint32_t b=0;b<SIM_BATCH;++b)status[b]=trial==1 && b%2==0?2:0;
        if(trial>=2) {unpack[0]=-1;unpack[rows-1]=rows;}
        std::fill(output.begin(),output.end(),9191);
        std::fill(writes.begin(),writes.end(),0);
        // Reverse block launch order to ensure no inter-block dependence.
        for(uint32_t i=0;i<rotating::kOutputCores;++i) {
            currentCore=rotating::kOutputCores-1-i;
            for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)PublishOutputs(a,currentCore);
        }
        for(uint32_t row=0;row<rows;++row)for(uint32_t k=0;k<top;++k) {
            bool active=valid[row] && unpack[row]>=0 && unpack[row]<rows;
            if(active)active=status[unpack[row]/4]==0;
            int32_t token=active?topk[unpack[row]*top+k]:-1;
            int32_t id=token>=0 && token<history?logical[owners[unpack[row]/4]*history+token]:-1;
            assert(output[row*top+k]==id);
            assert(output[rows*top+row*top+k]==(active?slots[unpack[row]*top+k]:-1));
            for(uint32_t plane=0;plane<2;++plane) {
                auto index=plane*rows*top+row*top+k;
                assert(writes[index]==1 && writers[index]==row%rotating::kOutputCores);
            }
        }
        assert(std::all_of(output.end()-32,output.end(),[](int32_t x){return x==9191;}));
    }
}
""".replace("FUNCTION", function)
    path = tmp_path / "outputs.cpp"
    path.write_text(source)
    binary = tmp_path / "outputs"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-DSIM_BATCH={batch}",
            f"-DV9_AIC_COUNT={aics}",
            str(path),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
