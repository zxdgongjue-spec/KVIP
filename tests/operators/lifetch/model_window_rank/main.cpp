// One stream, alternate variants after warming both. Every returned ID is checked.
#include <acl/acl.h>
#include "aclrtlaunch_window_rank_probe.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(int rc) {
    if (rc) throw std::runtime_error("ACL status " + std::to_string(rc));
}

static std::vector<uint32_t> load(const std::string& file) {
    std::ifstream stream(file, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open " + file);
    const auto size = static_cast<size_t>(stream.tellg());
    if (size % 4) throw std::runtime_error("Truncated fixture");
    std::vector<uint32_t> data(size / 4);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(data.data()), size);
    if (!stream) throw std::runtime_error("Incomplete fixture");
    return data;
}

int main(int argc, char** argv) {
    if (argc != 2 && (argc != 3 || std::string(argv[2]) != "--once")) return 2;
    try {
        auto inputs = load(std::string(argv[1]) + ".bin");
        auto expected = load(std::string(argv[1]) + ".want");
        const size_t calls = inputs.size() / 7184;
        if (!calls || inputs.size() != calls * 7184 || expected.size() != calls * 72)
            throw std::runtime_error("Invalid input dimensions");
        check(aclInit(nullptr));
        check(aclrtSetDevice(0));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        void* deviceInputs; void* deviceOutputs;
        check(aclrtMalloc(&deviceInputs, inputs.size()*4, ACL_MEM_MALLOC_HUGE_FIRST));
        check(aclrtMalloc(&deviceOutputs, expected.size()*4, ACL_MEM_MALLOC_HUGE_FIRST));
        check(aclrtMemcpy(deviceInputs, inputs.size()*4, inputs.data(), inputs.size()*4, ACL_MEMCPY_HOST_TO_DEVICE));
        std::vector<uint32_t> actual(expected.size());
        const int order[] = {0,1,0,1,1,0};
        const unsigned passes = argc == 3 ? 2 : 6;
        for (unsigned pass=0; pass<passes; ++pass) {
            const int variant=order[pass];
            check(ACLRT_LAUNCH_KERNEL(window_rank_probe)(1U,stream,deviceInputs,deviceOutputs,calls,variant));
            check(aclrtSynchronizeStream(stream));
            check(aclrtMemcpy(actual.data(),actual.size()*4,deviceOutputs,actual.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
            for(size_t i=0; i<calls; ++i) {
                const auto got=actual.data()+i*72, want=expected.data()+i*72;
                if (got[0] != want[0] || got[0] > 64 || !std::equal(got+8,got+8+got[0],want+8))
                    throw std::runtime_error("Mismatch for input "+std::to_string(i)+" count "+std::to_string(got[0])+" expected "+std::to_string(want[0])+" pass "+std::to_string(pass));
                const uint64_t ticks=(uint64_t(got[2])<<32)|got[1];
                std::printf("PASS row=%zu pass=%u variant=%d count=%u ticks=%llu\n",i,pass,variant,got[0],static_cast<unsigned long long>(ticks));
            }
            std::fflush(stdout);
        }
        check(aclrtFree(deviceOutputs)); check(aclrtFree(deviceInputs));
        check(aclrtDestroyStream(stream)); check(aclrtResetDevice(0)); check(aclFinalize());
    } catch (const std::exception& error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
}
