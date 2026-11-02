#include <acl/acl.h>

#include <chrono>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>

#include "a5_lidu_fused_batch_get_prepare_mtp_tiling.h"

extern "C" uint32_t aclrtlaunch_a5_lidu_fused_batch_get_prepare_mtp(
    uint32_t, void *, void *, void *, void *, void *, void *, void *, void *,
    void *, void *, void *, void *, void *, void *, void *, void *, void *,
    void *, void *, void *, void *, void *, void *, void *, void *, void *);

namespace {
constexpr uint64_t kSystemWorkspaceBytes = 16ULL * 1024ULL * 1024ULL;
constexpr uint32_t kSparseCount = 2048;
constexpr uint32_t kMtpMissCapacity = 8192;
constexpr uint32_t kProfileLanes = 8;
constexpr uint32_t kProfileFields = 32;
constexpr uint32_t kGlobalControlWords = 16;
constexpr uint32_t kWorkspaceWordsPerRequest = 4 * kMtpMissCapacity + 16;

struct Config {
    uint32_t batch;
    uint32_t packedQueries;
    uint32_t poolSize;
    uint32_t sourceCapacity;
    uint32_t residentCapacity;
    uint32_t residentBlockCapacity;
    uint32_t usedCoreNum;
    uint32_t coresPerRequest;
    uint32_t layerId;
    uint32_t numLayers;
    uint32_t maxNumReqs;
    uint32_t qcap;
    uint32_t warmup;
    uint32_t active;
    uint32_t profileEnabled;
};

void Check(uint32_t status, const char *operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with ACL status " +
            std::to_string(status));
    }
}

std::vector<char> Read(const std::string &path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot read " + path);
    }
    return {
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()};
}

void Write(const std::string &path, const void *data, size_t bytes)
{
    std::ofstream stream(path, std::ios::binary);
    stream.write(static_cast<const char *>(data), bytes);
    if (!stream) {
        throw std::runtime_error("cannot write " + path);
    }
}

Config ReadConfig(const std::string &path)
{
    std::ifstream stream(path);
    Config config{};
    stream >> config.batch >> config.packedQueries >> config.poolSize >>
        config.sourceCapacity >> config.residentCapacity >>
        config.residentBlockCapacity >> config.usedCoreNum >>
        config.coresPerRequest >> config.layerId >> config.numLayers >>
        config.maxNumReqs >> config.qcap >> config.warmup >> config.active;
    if (!stream || config.batch == 0 || config.packedQueries == 0 ||
        config.usedCoreNum == 0 || config.coresPerRequest == 0 ||
        config.active == 0) {
        throw std::runtime_error("invalid config " + path);
    }
    config.profileEnabled = 1;
    uint32_t profileEnabled = 1;
    if (stream >> profileEnabled) {
        if (profileEnabled > 1) {
            throw std::runtime_error("profile_enabled must be 0 or 1");
        }
        config.profileEnabled = profileEnabled;
    }
    return config;
}

class DeviceAllocations {
public:
    ~DeviceAllocations()
    {
        for (void *pointer : pointers_) {
            if (pointer != nullptr) {
                aclrtFree(pointer);
            }
        }
    }

    void *Allocate(size_t bytes)
    {
        if (bytes == 0) {
            throw std::runtime_error("zero-byte device allocation");
        }
        void *pointer = nullptr;
        Check(
            aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc");
        pointers_.push_back(pointer);
        return pointer;
    }

    void FreeAll()
    {
        for (void *pointer : pointers_) {
            Check(aclrtFree(pointer), "aclrtFree");
        }
        pointers_.clear();
    }

private:
    std::vector<void *> pointers_;
};

void HostToDevice(void *device, const std::vector<char> &host)
{
    Check(
        aclrtMemcpy(
            device, host.size(), host.data(), host.size(),
            ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy H2D");
}

void DeviceToHost(std::vector<char> &host, void *device)
{
    Check(
        aclrtMemcpy(
            host.data(), host.size(), device, host.size(),
            ACL_MEMCPY_DEVICE_TO_HOST),
        "aclrtMemcpy D2H");
}

std::vector<char> FilledInt32(size_t count, int32_t value)
{
    std::vector<char> result(count * sizeof(int32_t));
    auto *words = reinterpret_cast<int32_t *>(result.data());
    for (size_t index = 0; index < count; ++index) {
        words[index] = value;
    }
    return result;
}
}  // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: mtp_prepare_sim CASE_DIR");
        }
        const std::string root = argv[1];
        const Config config = ReadConfig(root + "/config.txt");

        auto topk = Read(root + "/topk.bin");
        auto queryLengths = Read(root + "/actual_seq_lengths_query.bin");
        auto requestEntries = Read(root + "/req_pool_entries.bin");
        auto tokenToSlot = Read(root + "/token_to_slot_pool.bin");
        auto slotToToken = Read(root + "/slot_to_token_pool.bin");
        auto cursors = Read(root + "/eviction_cursors.bin");
        auto cacheTokens = Read(root + "/cache_tokens.bin");
        auto candidateLens = Read(root + "/candidate_lens.bin");
        auto missSlots = Read(root + "/miss_slots.bin");
        auto missBitmap = Read(root + "/miss_slot_bitmap.bin");
        auto residentBlocks = Read(root + "/resident_block_table.bin");
        auto logicalLengths = Read(root + "/logical_seq_lengths.bin");
        std::vector<char> profileEnabled(sizeof(int32_t));
        *reinterpret_cast<int32_t *>(profileEnabled.data()) = config.profileEnabled;

        const auto requireBytes = [](const std::vector<char> &buffer,
                                     uint64_t expected,
                                     const char *name) {
            if (buffer.size() != expected) {
                throw std::runtime_error(
                    std::string(name) + " size mismatch: got " +
                    std::to_string(buffer.size()) + ", expected " +
                    std::to_string(expected));
            }
        };
        requireBytes(
            topk,
            uint64_t(config.packedQueries) * kSparseCount * sizeof(int32_t),
            "topk");
        requireBytes(
            tokenToSlot,
            uint64_t(config.poolSize) * config.sourceCapacity * sizeof(int32_t),
            "token_to_slot_pool");
        requireBytes(
            slotToToken,
            uint64_t(config.poolSize) * config.residentCapacity * sizeof(int32_t),
            "slot_to_token_pool");

        auto destination = FilledInt32(
            uint64_t(config.packedQueries) * kSparseCount, -777777);
        auto missCounts = FilledInt32(config.batch + 1, -777777);
        auto indices = FilledInt32(
            uint64_t(config.batch) * kMtpMissCapacity, -7001);
        auto offsets = FilledInt32(
            uint64_t(config.batch) * kMtpMissCapacity, -7002);
        auto outputIndices = FilledInt32(
            uint64_t(config.batch) * kMtpMissCapacity, -7003);
        auto residentLengths = FilledInt32(config.batch, -777777);
        std::vector<char> profiling(
            uint64_t(config.batch) * kProfileLanes * kProfileFields *
            sizeof(int64_t));

        const uint64_t userWorkspaceWords =
            kGlobalControlWords +
            uint64_t(kWorkspaceWordsPerRequest) * config.batch;
        const uint64_t userWorkspaceBytes =
            ((userWorkspaceWords * sizeof(int32_t) + 511ULL) / 512ULL) *
            512ULL;
        std::vector<char> workspace(
            kSystemWorkspaceBytes + userWorkspaceBytes, 0);

        A5LiduFusedBatchGetPrepareMtpTilingData tiling{
            config.usedCoreNum,
            config.coresPerRequest,
            config.batch,
            config.packedQueries,
            config.poolSize,
            config.sourceCapacity,
            config.residentCapacity,
            config.residentBlockCapacity,
            config.layerId,
            config.numLayers,
            config.maxNumReqs,
            config.qcap};
        std::vector<char> tilingBytes(sizeof(tiling));
        std::memcpy(tilingBytes.data(), &tiling, sizeof(tiling));

        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        DeviceAllocations allocations;
        const auto put = [&](const std::vector<char> &host) {
            void *device = allocations.Allocate(host.size());
            HostToDevice(device, host);
            return device;
        };

        void *topkDevice = put(topk);
        void *queryLengthsDevice = put(queryLengths);
        void *requestEntriesDevice = put(requestEntries);
        void *tokenToSlotDevice = put(tokenToSlot);
        void *slotToTokenDevice = put(slotToToken);
        void *cursorsDevice = put(cursors);
        void *cacheTokensDevice = put(cacheTokens);
        void *candidateLensDevice = put(candidateLens);
        void *missSlotsDevice = put(missSlots);
        void *missBitmapDevice = put(missBitmap);
        void *residentBlocksDevice = put(residentBlocks);
        void *logicalLengthsDevice = put(logicalLengths);
        void *profileEnabledDevice = put(profileEnabled);
        void *destinationDevice = put(destination);
        void *missCountsDevice = put(missCounts);
        void *indicesDevice = put(indices);
        void *offsetsDevice = put(offsets);
        void *outputIndicesDevice = put(outputIndices);
        void *residentLengthsDevice = put(residentLengths);
        void *profilingDevice = put(profiling);
        void *workspaceDevice = put(workspace);
        void *tilingDevice = put(tilingBytes);

        std::vector<int64_t> allProfiling(
            uint64_t(config.active) * config.batch * kProfileLanes *
            kProfileFields);
        std::vector<double> hostMicroseconds(config.active);
        const uint32_t repeats = config.warmup + config.active;
        const auto createDirectory = [](const std::string &path) {
            if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
                throw std::runtime_error("cannot create " + path);
            }
        };
        createDirectory(root + "/samples");
        const auto captureOutputs = [&](const std::string &directory) {
            createDirectory(directory);
            const auto save = [&](const char *name,
                                  const std::vector<char> &initial,
                                  void *device) {
                // Never replace the host reset image with an active output.
                std::vector<char> snapshot(initial.size());
                DeviceToHost(snapshot, device);
                Write(directory + "/" + name, snapshot.data(), snapshot.size());
            };
            save("actual_destination_slots.bin", destination, destinationDevice);
            save("actual_miss_counts.bin", missCounts, missCountsDevice);
            save("actual_indices.bin", indices, indicesDevice);
            save("actual_offsets.bin", offsets, offsetsDevice);
            save("actual_output_indices.bin", outputIndices, outputIndicesDevice);
            save("actual_resident_seq_lengths.bin", residentLengths, residentLengthsDevice);
            save("actual_token_to_slot_pool.bin", tokenToSlot, tokenToSlotDevice);
            save("actual_slot_to_token_pool.bin", slotToToken, slotToTokenDevice);
            save("actual_eviction_cursors.bin", cursors, cursorsDevice);
        };
        for (uint32_t repetition = 0; repetition < repeats; ++repetition) {
            HostToDevice(tokenToSlotDevice, tokenToSlot);
            HostToDevice(slotToTokenDevice, slotToToken);
            HostToDevice(cursorsDevice, cursors);
            HostToDevice(destinationDevice, destination);
            HostToDevice(missCountsDevice, missCounts);
            HostToDevice(indicesDevice, indices);
            HostToDevice(offsetsDevice, offsets);
            HostToDevice(outputIndicesDevice, outputIndices);
            HostToDevice(residentLengthsDevice, residentLengths);
            std::fill(profiling.begin(), profiling.end(), 0);
            HostToDevice(profilingDevice, profiling);
            HostToDevice(workspaceDevice, workspace);

            const auto begin = std::chrono::steady_clock::now();
            Check(
                aclrtlaunch_a5_lidu_fused_batch_get_prepare_mtp(
                    config.usedCoreNum, stream, topkDevice,
                    queryLengthsDevice, requestEntriesDevice,
                    tokenToSlotDevice, slotToTokenDevice, cursorsDevice,
                    cacheTokensDevice, candidateLensDevice, missSlotsDevice,
                    missBitmapDevice, residentBlocksDevice,
                    logicalLengthsDevice, profileEnabledDevice,
                    destinationDevice, missCountsDevice, indicesDevice,
                    offsetsDevice, outputIndicesDevice, residentLengthsDevice,
                    tokenToSlotDevice, slotToTokenDevice, cursorsDevice,
                    profilingDevice, workspaceDevice, tilingDevice),
                "kernel launch");
            Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
            const auto end = std::chrono::steady_clock::now();

            if (repetition >= config.warmup) {
                const uint32_t activeIndex = repetition - config.warmup;
                DeviceToHost(profiling, profilingDevice);
                std::memcpy(
                    allProfiling.data() +
                        uint64_t(activeIndex) * config.batch * kProfileLanes *
                            kProfileFields,
                    profiling.data(), profiling.size());
                hostMicroseconds[activeIndex] =
                    std::chrono::duration<double, std::micro>(end - begin)
                        .count();
                captureOutputs(root + "/samples/" + std::to_string(activeIndex));
            }
        }

        DeviceToHost(destination, destinationDevice);
        DeviceToHost(missCounts, missCountsDevice);
        DeviceToHost(indices, indicesDevice);
        DeviceToHost(offsets, offsetsDevice);
        DeviceToHost(outputIndices, outputIndicesDevice);
        DeviceToHost(residentLengths, residentLengthsDevice);
        DeviceToHost(tokenToSlot, tokenToSlotDevice);
        DeviceToHost(slotToToken, slotToTokenDevice);
        DeviceToHost(cursors, cursorsDevice);

        Write(root + "/actual_destination_slots.bin", destination.data(), destination.size());
        Write(root + "/actual_miss_counts.bin", missCounts.data(), missCounts.size());
        Write(root + "/actual_indices.bin", indices.data(), indices.size());
        Write(root + "/actual_offsets.bin", offsets.data(), offsets.size());
        Write(root + "/actual_output_indices.bin", outputIndices.data(), outputIndices.size());
        Write(root + "/actual_resident_seq_lengths.bin", residentLengths.data(), residentLengths.size());
        Write(root + "/actual_token_to_slot_pool.bin", tokenToSlot.data(), tokenToSlot.size());
        Write(root + "/actual_slot_to_token_pool.bin", slotToToken.data(), slotToToken.size());
        Write(root + "/actual_eviction_cursors.bin", cursors.data(), cursors.size());
        Write(
            root + "/profiling.bin", allProfiling.data(),
            allProfiling.size() * sizeof(int64_t));
        Write(
            root + "/host_microseconds.bin", hostMicroseconds.data(),
            hostMicroseconds.size() * sizeof(double));

        allocations.FreeAll();
        Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
        Check(aclrtResetDevice(0), "aclrtResetDevice");
        Check(aclFinalize(), "aclFinalize");
        std::cout << "PASS active=" << config.active
                  << " used_cores=" << config.usedCoreNum
                  << " simulator_only=true\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
