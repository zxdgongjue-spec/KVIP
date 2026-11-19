// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <latch>
#include <memory>
#include <mutex>
#include <type_traits>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#define __gm__
#define __ubuf__
#define __aicore__
#define __simt_callee__
#define __simt_vf__
#define LAUNCH_BOUND(n)
thread_local struct { unsigned x; } threadIdx;
template<class T> T asc_ldcg(T *p) { return *p; }
template<class T> void asc_stcg(T *p, T value) { std::atomic_ref<T>(*p).store(value,std::memory_order_relaxed); }
void asc_threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
thread_local std::barrier<> *activeVFBarrier=nullptr;
void asc_syncthreads() { if (activeVFBarrier) activeVFBarrier->arrive_and_wait(); }
void asc_atomic_or(unsigned *p, unsigned bits) { std::atomic_ref<unsigned>(*p).fetch_or(bits); }
enum class HardEvent { V_S };
enum class TPosition { VECCALC };
constexpr unsigned EVENT_ID0=0;
template<HardEvent> void SetFlag(unsigned) {}
template<HardEvent> void WaitFlag(unsigned) {}
uint64_t GetSystemCycle() { return 1; }
struct TensorView {
    void *p;
    void *GetPhyAddr() const { return p; }
};
template<TPosition> struct TBuf {
    std::vector<uint64_t> words;
    template<class> TensorView Get() { return {words.data()}; }
};
struct TPipe {
    template<class B> void InitBuffer(B &buf, size_t bytes) { buf.words.resize((bytes + 7) / 8); }
    void Destroy() {}
};
namespace pipeline_ab::high_post_index {
struct Edge;
inline void TransferOwnedFields(Edge*,uint32_t,uint32_t,uint32_t,uint32_t,
    uint64_t*,uint32_t*,uint64_t*,uint64_t*,uint64_t*,uint32_t*,uint32_t,uint32_t);
}
namespace Simt {
struct Dim3 { unsigned x, y, z; };
template<auto Function, class... A> void VF_CALL(Dim3 shape, A... args) {
    // The owned-field VF contains one real block barrier. Bound the host
    // resource requirement to a single 256-thread VF, while exercising the
    // production entry point and its barrier. Other VFs have independent lanes.
    if constexpr (std::is_same_v<decltype(Function),decltype(&pipeline_ab::high_post_index::TransferOwnedFields)>) {
        static_assert(Function==&pipeline_ab::high_post_index::TransferOwnedFields);
        static std::mutex runOneBlock;
        std::lock_guard guard(runOneBlock);
        std::barrier rendezvous(shape.x);
        std::vector<std::thread> workers;
        for (unsigned t=0;t<shape.x;++t) workers.emplace_back([&,t] {
            threadIdx.x=t; activeVFBarrier=&rendezvous; Function(args...);
        });
        for (auto &worker:workers) worker.join();
    } else {
        for (unsigned lane=0;lane<shape.x;++lane) { threadIdx.x=lane; Function(args...); }
    }
}
}
namespace kvaf_device { constexpr uint64_t kInvalidArgs=9; }
namespace pipeline_ab {
constexpr uint32_t kPrefetch=2048, kSlots=8192, THREADS=512, MAP=4096, EMPTY=UINT32_MAX,
    KEYS_POS=0, VALUES_POS=MAP, CONTROL=4*MAP;
uint32_t prefix(uint32_t, uint32_t*);
uint32_t get(uint32_t*, uint32_t*, uint32_t);
struct Args { uint64_t highIndex, keys, scales, addresses, logical; uint32_t resident, history; };
struct Context { uint64_t hotValid; };
}
#include "high_index_workspace.h"
#include "high_index_partition.h"
namespace pipeline_ab::high_post_index {
#include "transfer_segments.h"
#include "high_index_row.h"
}
using namespace pipeline_ab;
using namespace pipeline_ab::high_post_index;

constexpr unsigned batch = 16, lanes = 4, history = 6144, resident = 512, stride = history + 23;
using Index = std::vector<uint32_t>;
struct Team {
    std::barrier<> barrier{lanes};
    std::atomic<unsigned> arrivals{0};
    bool reject;
    uint64_t Gathered() {
        arrivals.fetch_add(1);
        barrier.arrive_and_wait();
        return reject ? 13 : 0;
    }
};
int main() {
    std::vector<uint64_t> storage(batch * kRowBytes / sizeof(uint64_t)), keys(batch * history * 16),
        addresses(batch * stride), logical(batch * history);
    Index scales(batch * history), valid(batch * resident, 1), wantedValid = valid;
    Args a{uint64_t(storage.data()),uint64_t(keys.data()),uint64_t(scales.data()),uint64_t(addresses.data()),
           uint64_t(logical.data()),resident,history};
    Context ctx{uint64_t(valid.data())};
    std::array<unsigned,batch> pairs{0,1,4,14,0,35,1,117,0,19,1700,1800,23,51,101,0};
    std::array<unsigned,batch> singles{0,0,3,64,4097,63,65,108,4096,0,1024,1025,3,17,0,21};
    std::array<unsigned,batch> invalid{};
    std::array<Team,batch> teams;
    std::vector<unsigned> wanted(batch * history);
    std::iota(wanted.begin(), wanted.end(), 0);
    std::iota(addresses.begin(), addresses.end(), UINT64_C(0x192000123));
    std::iota(logical.begin(), logical.end(), UINT64_C(0x6a6700456));
    std::iota(scales.begin(), scales.end(), 0x78a00000u);
    std::iota(keys.begin(), keys.end(), UINT64_C(0x2312312300));
    const auto oldAddress=addresses, oldLogical=logical, oldKey=keys;
    const auto oldScale=scales;
    std::mt19937 rng(17113);
    for (unsigned row = 0; row < batch; ++row) {
        unsigned owner = batch-1-row, offset=owner*history;
        // One row fails its gather fence. Its cycle has no legal scatter.
        teams[row].reject = row==15;
        Index ids(history);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        auto edges=reinterpret_cast<Edge*>(a.highIndex+row*uint64_t(kRowBytes));
        auto payload=reinterpret_cast<Tuple*>(edges+kCapacity);
        for (unsigned i=0;i<pairs[row];++i) {
            unsigned s=ids[2*i],d=ids[2*i+1];
            edges[i]={s,d};
            std::swap(wanted[offset+s],wanted[offset+d]);
        }
        for (unsigned i=0;i<singles[row];++i) {
            unsigned pos=2*pairs[row],s=ids[pos+i],d=ids[pos+(i+1)%singles[row]];
            edges[kCapacity-1-i]={s,d};
            if (!teams[row].reject) wanted[offset+d]=offset+s;
        }
        invalid[row]=row==15 ? 0 : row+1;
        for (unsigned i=0;i<invalid[row];++i) {
            unsigned slot=(83*(i==3 ? 2 : i)+row)%resident; // repeated clear is idempotent
            payload[i].padding=slot;
            wantedValid[owner*resident+slot]=0;
        }
    }
    std::latch independent(lanes), privatePeers(lanes-1);
    std::vector<std::thread> all;
    for (unsigned block=0;block<batch*lanes;++block) {
        all.emplace_back([&,block]{
            unsigned row=block%batch,lane=block/batch,owner=batch-1-row;
            // A slow row is allowed to arrive only after another independent
            // four-lane row has completed. A collective spanning rows cannot work.
            if (row==4) independent.wait();
            // Hold the only long-cycle owner until the three pure-swap
            // lanes have returned. A shared gather here would deadlock.
            if (row==12 && lane==lanes-1) privatePeers.wait();
            uint64_t ticks[32]{};
            auto error=ApplyRowIndex(&a,&ctx,row,owner,lane,lanes,pairs[row],singles[row],
                invalid[row],stride,ticks,true,teams[row]);
            assert(error==(teams[row].reject ? 13u : 0u));
            unsigned p=pairs[row]*(lane+1)/lanes-pairs[row]*lane/lanes;
            unsigned s=singles[row]*(lane+1)/lanes-singles[row]*lane/lanes;
            const unsigned moves=2*pairs[row]+singles[row];
            if(singles[row] && singles[row]<=kLocalTuples && singles[row]*lanes<=moves) {
                unsigned start=std::min(pairs[row],moves*lane/(2*lanes));
                unsigned end=std::min(pairs[row],moves*(lane+1)/(2*lanes));
                p=end-start;
                s=lane==lanes-1 ? singles[row] : 0;
                assert(2*p+s <= (moves+lanes-1)/lanes+1);
            }
            assert(ticks[9]==2*p+s && ticks[16]==p && ticks[17]==s);
            assert(ticks[18]==(s>kLocalTuples ? s : 0));
            const bool shared=singles[row] && (singles[row]>kLocalTuples || singles[row]*lanes>moves);
            const bool fields=shared && singles[row]>=64 && singles[row]<=4096;
            assert(ticks[13]==(fields ? singles[row]*(lane<2 ? 40u : lane==2 ? 36u : 32u) : 0));

            if (row==2) independent.count_down();
            if (row==12 && lane<lanes-1) privatePeers.count_down();
        });
    }
    for (auto &t:all)t.join();
    for (unsigned row=0;row<batch;++row) {
        const unsigned moves=2*pairs[row]+singles[row];
        bool shared=singles[row] && (singles[row]>kLocalTuples || singles[row]*lanes>moves);
        bool fields=shared && singles[row]>=64 && singles[row]<=4096;
        assert(teams[row].arrivals.load()==(shared && !fields ? lanes : 0u));
    }
    for (unsigned owner=0;owner<batch;++owner) {
        for (unsigned pos=0;pos<history;++pos) {
            unsigned dest=owner*history+pos,src=wanted[dest],raw=src-owner*history;
            assert(logical[dest]==oldLogical[src] && scales[dest]==oldScale[src]);
            assert(addresses[owner*stride+pos]==oldAddress[owner*stride+raw]);
            for (unsigned w=0;w<16;++w) assert(keys[16*dest+w]==oldKey[16*src+w]);
        }
        for (unsigned pos=history;pos<stride;++pos)
            assert(addresses[owner*stride+pos]==oldAddress[owner*stride+pos]);
    }
    assert(valid==wantedValid);
}
