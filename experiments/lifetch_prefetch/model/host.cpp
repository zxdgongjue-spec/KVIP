#include <acl/acl.h>
#include "context.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

extern "C" uint32_t aclrtlaunch_prefetch_service_model(uint32_t,void*,void*);

namespace {
void Check(uint32_t code) {
    if (code) throw std::runtime_error("ACL or launch error " + std::to_string(code));
}
void Require(bool condition, const char* error) {
    if (!condition) throw std::runtime_error(error);
}
class Allocations {
    std::vector<void*> device_;
public:
    template<class T> uint64_t Put(const std::vector<T>& values) {
        void* ptr = nullptr;
        const size_t bytes = values.size() * sizeof(T);
        const size_t capacity = (bytes + 127) / 128 * 128;
        Check(aclrtMalloc(&ptr, capacity, ACL_MEM_MALLOC_HUGE_FIRST));
        device_.push_back(ptr);
        Check(aclrtMemset(ptr, capacity, 0, capacity));
        Check(aclrtMemcpy(ptr, capacity, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        return reinterpret_cast<uint64_t>(ptr);
    }
    ~Allocations() {
        for (auto* ptr : device_) aclrtFree(ptr);
    }
};
template<class T> std::vector<T> Read(uint64_t ptr, size_t count) {
    std::vector<T> values(count);
    const size_t bytes = count * sizeof(T);
    Check(aclrtMemcpy(values.data(), bytes, reinterpret_cast<void*>(ptr), bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    return values;
}
template<class T> void Write(uint64_t ptr, const std::vector<T>& values) {
    const size_t bytes = values.size() * sizeof(T);
    Check(aclrtMemcpy(reinterpret_cast<void*>(ptr), bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}
void Run(aclrtStream stream, uint32_t batch, uint32_t history, uint32_t resident,
         bool expectedUb, bool expectedBatch, uint32_t epochCount) {
    constexpr uint32_t depth = 1024, width = 656, initial = depth - 32;
    constexpr uint32_t margin = 32;
    Allocations mem;
    rotating::Context ctx{};
    auto& shape = ctx.shape;
    shape.batch = batch;
    shape.history = history;
    shape.resident = resident;
    shape.indexHeads = 32;
    shape.aicCount = 2 * batch;
    shape.splitQ = 1;
    shape.rounds = 4;
    shape.tokenBytes = width;
    Check(rotating::PrepareShape(&shape));
    Require(shape.serviceLanes == batch, "expected one request per Service lane");
    ctx.windowLimit = 128;
    ctx.prefetchLimit = 512;
    ctx.outputStride = resident + 8192;
    ctx.mapping = mem.Put(std::vector<uint32_t>(batch * history, UINT32_MAX));
    ctx.incremental = mem.Put(std::vector<uint32_t>(batch * (512 + 4*2048*6 + 1024 + (history+255)/256*32)/4));
    ctx.locations = mem.Put(std::vector<uint32_t>(batch * 8192, UINT32_MAX));
    ctx.events = mem.Put(std::vector<rotating::Event>(batch * 4));
    ctx.mappingJournal = mem.Put(std::vector<rotating::MappingJournal>(batch));
    // Deliberately exercise WQE addresses without allocating simulated KV.
    constexpr uint64_t remoteBase = 0x123400000ULL, outputBase = 0x567800000ULL;
    std::vector<uint64_t> addresses(history);
    for (uint32_t i = 0; i < history; ++i) addresses[i] = remoteBase + uint64_t(i) * width;
    const auto addressData = mem.Put(addresses);
    std::vector<KvafBatchGetDirectUrmaEntity> entities(batch);
    std::vector<KvafExecuteBatchGetKernelArgs> args(2 * batch);
    std::vector<rotating::Row> rows(batch);
    for (uint32_t b = 0; b < batch; ++b) {
        auto& entity = entities[b];
        entity.sq_depth = depth;
        entity.cq_depth = 64;
        entity.sq_va = mem.Put(std::vector<uint32_t>(depth * 16));
        entity.sq_db_va = mem.Put(std::vector<uint64_t>(16));
        auto& arg = args[2*b];
        arg.channel_entity = mem.Put(std::vector<KvafBatchGetDirectUrmaEntity>{entity});
        KvafBatchGetQueueState queue{};
        queue.sq_producer_index = initial;
        queue.cq_consumer_index = 7;
        arg.queue_state = mem.Put(std::vector<KvafBatchGetQueueState>{queue});
        arg.output = outputBase;
        rows[b].addresses = addressData;
        rows[b].entity = arg.channel_entity;
        rows[b].queue = arg.queue_state;
    }
    ctx.rows = mem.Put(rows);
    ctx.args = mem.Put(args);
    const auto deviceContext = mem.Put(std::vector<rotating::Context>{ctx});
    const auto compact = mem.Put(std::vector<uint32_t>(batch * 128));
    const auto binding = mem.Put(std::vector<uint64_t>{deviceContext, 0, compact});
    std::vector<uint32_t> previous(batch, initial);
    std::vector<std::set<uint32_t>> admitted(batch);
    const std::vector<uint32_t> counts{31,24,12,28}; // logical pitch 32, 95 input IDs
    for (uint32_t epoch = 0; epoch < epochCount; ++epoch) {
        std::vector<uint32_t> metadata(batch * 128), locations(batch * 8192, UINT32_MAX),
            local(batch * 128, UINT32_MAX);
        std::vector<std::set<uint32_t>> wanted(batch);
        for (uint32_t b = 0; b < batch; ++b) {
            for (uint32_t q = 0; q < 4; ++q) {
                const uint32_t begin = resident + margin + b * 2 + q * 24 + (epoch >= 2 ? 32 : 0);
                const uint32_t n = epoch == 2 ? counts[q] - 10 : counts[q];
                metadata[b*128+q*32+6] = n;
                for (uint32_t j = 0; j < n; ++j) {
                    const auto id = begin + j;
                    locations[b*8192+q*2048+j] = id;
                    local[b*128+q*32+j] = id;
                    if (!admitted[b].count(id)) wanted[b].insert(id);
                }
            }
        }
        Write(ctx.incremental, metadata);
        Write(ctx.locations, locations);
        Write(compact, local);
        Write(binding, std::vector<uint64_t>{deviceContext, epoch, compact});
        std::cout << "SERVICE_BEGIN epoch=" << epoch << " batch=" << batch << " history=" << history << std::endl;
        Check(aclrtlaunch_prefetch_service_model(batch, stream, reinterpret_cast<void*>(binding)));
        Check(aclrtSynchronizeStream(stream));
        auto actualRows = Read<rotating::Row>(ctx.rows, batch);
        auto journal = Read<rotating::MappingJournal>(ctx.mappingJournal, batch);
        auto events = Read<rotating::Event>(ctx.events, batch*4);
        for (uint32_t b = 0; b < batch; ++b) {
            const auto& row = actualRows[b];
            const auto& event = events[epoch*batch+b];
            const auto n = wanted[b].size();
            const auto end = previous[b] + n;
            const auto countDoorbells = expectedBatch && n > 64 ? 2U : uint32_t(n != 0);
            Require(event.complete && row.epochCount[epoch] == n && event.postedCount == n,
                "wrong window selection or publication");
            Require((event.bitmapBytes >> 32 & 0xffff) == countDoorbells &&
                bool(event.bitmapBytes >> 48) == expectedUb, "wrong runtime route or doorbell count");
            auto head = Read<KvafBatchGetQueueState>(args[b*2].queue_state, 1)[0];
            auto bell = Read<uint64_t>(entities[b].sq_db_va, 1)[0];
            Require(head.sq_producer_index == end && head.cq_consumer_index == 7 &&
                bell == (end == initial ? 0 : end), "SQ head/bell or CQ moved incorrectly");
            auto sq = Read<uint32_t>(entities[b].sq_va, depth*16);
            std::set<uint32_t> got;
            for (uint32_t index = previous[b]; index < end; ++index) {
                auto wqe = &sq[(index & (depth-1))*16];
                const uint32_t word = (index + 1 == end ? 0x30360000U : 0x30150000U) |
                    (index & depth ? 0U : 0x80000000U);
                const uint64_t from = uint64_t(wqe[10]) | uint64_t(wqe[11]) << 32;
                const uint64_t to = uint64_t(wqe[14]) | uint64_t(wqe[15]) << 32;
                const uint32_t token = (from - remoteBase) / width;
                const uint32_t ordinal = index - initial;
                Require(word == wqe[0] && wqe[12] == width && addresses.at(token) == from,
                    "WQE completion, wrap or source mismatch");
                Require(to == outputBase + (uint64_t(b)*ctx.outputStride + resident + ordinal)*width,
                    "WQE physical slot mismatch");
                Require(journal[b].tokens[ordinal] == token && got.insert(token).second,
                    "duplicate WQE or mismatched mapping journal");
            }
            Require(got == wanted[b], "WQE token membership mismatch");
            admitted[b].insert(got.begin(), got.end());
            Require(row.requests == admitted[b].size() && row.pending == (epoch == 0 ? 1U : 1U + uint32_t(epoch >= 2)) &&
                journal[b].count == admitted[b].size(), "row pending or claim mismatch");
            const auto* stamps = reinterpret_cast<const uint64_t*>(event.pad);
            std::cout << "SERVICE_PASS epoch=" << epoch << " row=" << b << " posted=" << n
                << " duplicate=0 pending=" << row.pending << " bells=" << countDoorbells << " ub=" << expectedUb
                << " service_ticks=" << event.end-event.begin << " first_from_service_ticks="
                << (n ? stamps[0]-event.begin : 0) << " last_from_service_ticks="
                << (n ? stamps[1]-event.begin : 0) << std::endl;
            previous[b] = end;
        }
    }
}
}
int main(int argc, const char** argv) {
    try {
        Require(argc == 6, "service_model B H R base|ub|batch epochs");
        Check(aclInit(nullptr));
        Check(aclrtSetDevice(0));
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream));
        const std::string arm = argv[4];
        Run(stream, std::stoul(argv[1]), std::stoul(argv[2]), std::stoul(argv[3]), arm == "ub",
            arm == "batch", std::stoul(argv[5]));
        Check(aclrtDestroyStream(stream));
        Check(aclrtResetDevice(0));
        Check(aclFinalize());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
