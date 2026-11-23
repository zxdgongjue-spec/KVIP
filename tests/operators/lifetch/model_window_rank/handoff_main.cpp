// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include "aclrtlaunch_handoff_probe.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

static void check(int rc) {
    if (rc) throw std::runtime_error("ACL status " + std::to_string(rc));
}

int main(int argc, char** argv) {
    if (argc!=2) return 2;
    try {
        constexpr uint32_t stride=528;
        std::ifstream file(argv[1], std::ios::binary|std::ios::ate);
        if (!file) throw std::runtime_error("Missing fixture");
        auto bytes=static_cast<size_t>(file.tellg());
        if (!bytes || bytes%(stride*4)) throw std::runtime_error("Bad fixture size");
        size_t rows=bytes/(stride*4);
        std::vector<uint32_t> input(rows*stride), actual(rows*stride);
        file.seekg(0); file.read(reinterpret_cast<char*>(input.data()),bytes);
        check(aclInit(nullptr)); check(aclrtSetDevice(0)); aclrtStream stream;
        check(aclrtCreateStream(&stream)); void *in,*ex,*out;
        check(aclrtMalloc(&in,bytes,ACL_MEM_MALLOC_HUGE_FIRST));
        check(aclrtMalloc(&out,bytes,ACL_MEM_MALLOC_HUGE_FIRST));
        check(aclrtMalloc(&ex,(256+8*2048)*4,ACL_MEM_MALLOC_HUGE_FIRST));
        check(aclrtMemcpy(in,bytes,input.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE));
        const int order[]={0,1,0,1,1,0};
        for (unsigned pass=0;pass<6;++pass) {
            const int v=order[pass];
            check(ACLRT_LAUNCH_KERNEL(handoff_probe)(1U,stream,in,ex,out,rows,v));
            check(aclrtSynchronizeStream(stream));
            check(aclrtMemcpy(actual.data(),bytes,out,bytes,ACL_MEMCPY_DEVICE_TO_HOST));
            for(size_t i=0;i<rows;++i) {
                auto a=actual.data()+i*stride,b=input.data()+i*stride;
                if(b[0]!=1 && b[0]!=2) throw std::runtime_error("Invalid row count");
                for(uint32_t q=0;q<b[0]*4;++q) {
                    uint32_t n=b[8+q];
                    if(n>64 || a[8+q]!=n || !std::equal(b+16+q*64,b+16+q*64+n,a+16+q*64))
                        throw std::runtime_error("Mismatch at row="+std::to_string(i)+" Q="+std::to_string(q));
                }
                uint64_t ticks=(uint64_t(a[1])<<32)|a[0];
                if(!ticks) throw std::runtime_error("Missing timestamps");
                std::printf("PASS row=%zu pass=%u variant=%d rows=%u ticks=%llu\n",i,pass,v,b[0],
                            static_cast<unsigned long long>(ticks));
            }
            std::fflush(stdout);
        }
        check(aclrtFree(out));check(aclrtFree(ex));check(aclrtFree(in));
        check(aclrtDestroyStream(stream));check(aclrtResetDevice(0));check(aclFinalize());
    } catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
