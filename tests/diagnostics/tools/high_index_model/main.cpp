// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include "context.h"
#include "pipeline_types.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
extern "C" uint32_t aclrtlaunch_v9_streaming_resolve_update(uint32_t, void *, void *);
static void Check(uint32_t r) {
    if (r)
        throw std::runtime_error("ACL " + std::to_string(r));
}
struct D {
    void *p{};
    size_t n;
    template <class T> explicit D(const std::vector<T> &v) : n(sizeof(T) * v.size()) {
        Check(aclrtMalloc(&p, n, ACL_MEM_MALLOC_HUGE_FIRST));
        Check(aclrtMemcpy(p, n, v.data(), n, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    ~D() { aclrtFree(p); }
    uint64_t addr() const { return uint64_t(p); }
    template <class T> std::vector<T> read() {
        std::vector<T> x(n / sizeof(T));
        Check(aclrtMemcpy(x.data(), n, p, n, ACL_MEMCPY_DEVICE_TO_HOST));
        return x;
    }
};
int main(int argc, char **argv) {
    try {
        using namespace rotating;
        using namespace pipeline_ab;
        constexpr uint32_t H = 32768, R = 16384, E = 10240;
        const uint32_t B = 16, C = 64, A = H + 128;
        const std::string layout = argc == 2 ? argv[1] : "";
        if (layout != "even" && layout != "skew" && layout != "even-mixed" &&
            layout != "skew-mixed")
            throw std::runtime_error("expected even, skew, even-mixed or skew-mixed");
        const bool exact = layout == "even" || layout == "skew";
        const bool skew = layout == "skew" || layout == "skew-mixed";
        RuntimeShape shape{};
        shape.batch = B;
        shape.history = H;
        shape.resident = R;
        shape.indexHeads = 32;
        shape.aicCount = C / 2;
        shape.splitQ = 1;
        shape.rounds = 8;
        shape.finalWorkers = 4;
        shape.tokenBytes = 656;
        if (PrepareShape(&shape))
            throw std::runtime_error("shape");
        std::vector<uint32_t> src(B * E), dst(B * E), own(B), hot(uint64_t(B) * R, 1),
            pos(uint64_t(B) * H);
        std::vector<BatchFlag> flags(B * kFlagCount);
        uint64_t events = 0, moves = 0;
        std::vector<uint32_t> counts(B);
        for (uint32_t b = 0; b < B; ++b) {
            own[b] = B - 1 - b;
            std::vector<uint32_t> p(H), inv(H);
            std::iota(p.begin(), p.end(), 0U);
            std::iota(inv.begin(), inv.end(), 0U);
            // even/skew contain exactly 5,000 disjoint swaps. Mixed fixtures have 5,440 events
            // and exercise repeat victims across phase boundaries in half the requests.
            const uint32_t n = exact ? (skew ? (b < 4 ? 966 : 94 + (b < 12)) : 312 + (b < 8))
                                     : (skew ? (b < 4 ? 880 : 160) : 340);
            uint32_t start = 0, off = 0;
            for (uint32_t ph = 0; ph < 5; ++ph) {
                uint32_t num = ph == 0 ? (n + 1) / 2 : (n - n / 2) / 4;
                if (ph == 4)
                    num = n - start;
                auto &f = flags[b * kFlagCount + (ph ? 12 + ph - 1 : 0)];
                f.ready = 1;
                f.count = num;
                for (uint32_t j = 0; j < num; ++j) {
                    uint32_t i = ph ? 2048 + off + j : j;
                    // Deterministic sparse permutation; re-use a prior destination on odd rows to
                    // force real closures across phases.
                    uint32_t s = R + ((start + j) * 7) % R, cell = start + j;
                    if (!exact && (b & 1) && ph > 0 && j == 0)
                        cell = ph - 1;
                    uint32_t d = (cell * 13) % R;
                    src[b * E + i] = s;
                    dst[b * E + i] = d;
                }
                for (uint32_t at = 0; at < num; at += 512) {
                    std::vector<uint32_t> ss, dd;
                    std::unordered_set<uint32_t> sset, dset;
                    for (uint32_t j = at; j < std::min(at + 512, num); ++j) {
                        uint32_t i = b * E + (ph ? 2048 + off : 0) + j;
                        auto sx = inv[src[i]], dx = dst[i];
                        ss.push_back(sx);
                        dd.push_back(dx);
                        sset.insert(sx);
                        dset.insert(dx);
                    }
                    std::vector<uint32_t> os, od;
                    for (auto d : dd)
                        if (!sset.count(d))
                            os.push_back(d);
                    for (auto s : ss)
                        if (!dset.count(s))
                            od.push_back(s);
                    ss.insert(ss.end(), os.begin(), os.end());
                    dd.insert(dd.end(), od.begin(), od.end());
                    std::vector<uint32_t> tmp;
                    for (auto s : ss)
                        tmp.push_back(p[s]);
                    for (size_t j = 0; j < ss.size(); ++j) {
                        p[dd[j]] = tmp[j];
                        inv[tmp[j]] = dd[j];
                    }
                }
                start += num;
                if (ph)
                    off += num;
            }
            events += start;
            for (uint32_t i = 0; i < H; ++i) {
                counts[b] += p[i] != i;
                pos[own[b] * H + i] = p[i];
            }
            moves += counts[b];
        }
        if (exact && (events != 5000 || moves != 10000))
            throw std::runtime_error("exact 10000 fixture");
        printf("INDEX_MODEL_START B=%u cores=%u history=%u resident=%u layout=%s events=%llu "
               "moves=%llu\n",
               B, C, H, R, layout.c_str(), (unsigned long long)events, (unsigned long long)moves);
        fflush(stdout);
        Check(aclInit(nullptr));
        Check(aclrtSetDevice(0));
        aclrtStream stream;
        Check(aclrtCreateStream(&stream));
        {
            std::vector<uint64_t> keys(uint64_t(B) * H * 16), addrs(uint64_t(B) * A),
                logicals(uint64_t(B) * H);
            std::vector<uint32_t> scales(uint64_t(B) * H);
            std::iota(keys.begin(), keys.end(), uint64_t(0x101010101010ULL));
            std::iota(addrs.begin(), addrs.end(), uint64_t(0x22222222ULL));
            std::iota(logicals.begin(), logicals.end(), uint64_t(0x3333333ULL));
            std::iota(scales.begin(), scales.end(), 17U);
            D dk(keys), df(scales), da(addrs), dl(logicals), dsrc(src), ddst(dst), dhot(hot),
                dflags(flags), do_(own);
            D high(std::vector<uint8_t>(uint64_t(B) * 2 * E * 160 + uint64_t(C) * 1024));
            D dense(std::vector<uint32_t>(uint64_t(B) * 6 * H)),
                worker(std::vector<uint32_t>(B * 32));
            D times(std::vector<uint64_t>(B * 256 + C * 32)),
                records(std::vector<uint64_t>(C * 16));
            std::vector<uint64_t> update(28);
            update[7] = A;
            D bindings(update);
            Context ctx{};
            ctx.hotValid = dhot.addr();
            ctx.rowOwners = do_.addr();
            ctx.shape = shape;
            D dctx(std::vector<Context>{ctx});
            Args a{};
            a.highIndex = high.addr();
            a.context = dctx.addr();
            a.keys = dk.addr();
            a.scales = df.addr();
            a.addresses = da.addr();
            a.logical = dl.addr();
            a.valid = dhot.addr();
            a.flags = dflags.addr();
            a.eventsSrc = dsrc.addr();
            a.eventsDst = ddst.addr();
            a.workerStatus = worker.addr();
            a.times = times.addr();
            a.history = H;
            a.resident = R;
            a.maxEvents = E;
            a.prefetchCount = 0x81040000U;
            a.updateBindings = bindings.addr();
            a.denseShadow = dense.addr();
            a.postIndex = records.addr();
            D args(std::vector<Args>{a});
#if INDEX_TARGET_NPU
            // Reset input tuples outside every measured bracket. Events include the complete
            // generated device entry, including its private plan and shared joins.
            Args uninstrumented = a;
            uninstrumented.prefetchCount &= ~0x80000000U;
            D headlineArgs(std::vector<Args>{uninstrumented});
            std::vector<double> elapsedUs;
            aclrtEvent begin, end;
            Check(aclrtCreateEventExWithFlag(&begin, ACL_EVENT_TIME_LINE));
            Check(aclrtCreateEventExWithFlag(&end, ACL_EVENT_TIME_LINE));
            auto restore = [&] {
                Check(aclrtMemcpy(dk.p, dk.n, keys.data(), dk.n, ACL_MEMCPY_HOST_TO_DEVICE));
                Check(aclrtMemcpy(df.p, df.n, scales.data(), df.n, ACL_MEMCPY_HOST_TO_DEVICE));
                Check(aclrtMemcpy(da.p, da.n, addrs.data(), da.n, ACL_MEMCPY_HOST_TO_DEVICE));
                Check(aclrtMemcpy(dl.p, dl.n, logicals.data(), dl.n, ACL_MEMCPY_HOST_TO_DEVICE));
                Check(aclrtMemcpy(dhot.p, dhot.n, hot.data(), dhot.n, ACL_MEMCPY_HOST_TO_DEVICE));
                Check(aclrtMemset(worker.p, worker.n, 0, worker.n));
                Check(aclrtMemset(times.p, times.n, 0, times.n));
                Check(aclrtMemset(records.p, records.n, 0, records.n));
            };
            for (unsigned repeat = 0; repeat != 7; ++repeat) {
                restore();
                Check(aclrtRecordEvent(begin, stream));
                Check(aclrtlaunch_v9_streaming_resolve_update(C, stream, headlineArgs.p));
                Check(aclrtRecordEvent(end, stream));
                Check(aclrtSynchronizeStream(stream));
                float ms;
                Check(aclrtEventElapsedTime(&ms, begin, end));
                if (repeat)
                    elapsedUs.push_back(1000.0 * ms);
            }
            Check(aclrtDestroyEvent(begin));
            Check(aclrtDestroyEvent(end));
            std::sort(elapsedUs.begin(), elapsedUs.end());
            const double mean =
                std::accumulate(elapsedUs.begin(), elapsedUs.end(), 0.0) / elapsedUs.size();
            printf("DEVICE_INDEX_US mean=%.3f p50=%.3f samples=%zu warmup=1 "
                   "diagnostics_excluded=true\n",
                   mean, (elapsedUs[2] + elapsedUs[3]) / 2, elapsedUs.size());
            const auto headlineKeys = dk.read<uint64_t>();
            const auto headlineAddr = da.read<uint64_t>();
            const auto headlineLogical = dl.read<uint64_t>();
            const auto headlineScales = df.read<uint32_t>();
            const auto headlineValid = dhot.read<uint32_t>();
            restore();
#endif
            Check(aclrtlaunch_v9_streaming_resolve_update(C, stream, args.p));
            Check(aclrtSynchronizeStream(stream));
            auto actualk = dk.read<uint64_t>(), actuala = da.read<uint64_t>(),
                 actuall = dl.read<uint64_t>();
            auto actualf = df.read<uint32_t>(), work = worker.read<uint32_t>();
            auto actualh = dhot.read<uint32_t>();
#if INDEX_TARGET_NPU
            if (actualk != headlineKeys || actuala != headlineAddr || actuall != headlineLogical ||
                actualf != headlineScales || actualh != headlineValid)
                throw std::runtime_error("diagnostic and timed outputs differ");
#endif
            auto ticks = times.read<uint64_t>();
            bool privatePlan = false;
            for (unsigned c = 0; c < C; ++c)
                privatePlan |= ticks[B * 256 + c * 32 + 15] != 0;
            for (uint32_t b = 0; b < B; ++b) {
                uint32_t count =
                    privatePlan ? 2 * work[b * 32 + 3] + work[b * 32 + 4] : work[b * 32 + 3];
                if (work[b * 32] != 0 || count != counts[b])
                    throw std::runtime_error("plan error or move count row=" + std::to_string(b) +
                                             " got=" + std::to_string(count) +
                                             " expected=" + std::to_string(counts[b]));
                uint32_t o = own[b];
                for (uint32_t i = 0; i < H; ++i) {
                    uint32_t idx = o * H + i, q = pos[idx];
                    if (std::memcmp(&actualk[uint64_t(idx) * 16], &keys[uint64_t(o * H + q) * 16],
                                    128) ||
                        actualf[idx] != scales[o * H + q] ||
                        actuala[o * A + i] != addrs[o * A + q] ||
                        actuall[idx] != logicals[o * H + q])
                        throw std::runtime_error("tuple wrong row " + std::to_string(b) + " i " +
                                                 std::to_string(i));
                }
            }
            if (actualh != hot)
                throw std::runtime_error("hot valid altered");
            uint64_t start = UINT64_MAX, plan = 0, gather = 0, finish = 0;
            uint64_t earlyStart = UINT64_MAX, earlyEnd = 0, movedCount = 0;
            for (uint32_t c = 0; c < C; ++c) {
                auto t = ticks.data() + B * 256 + c * 32;
                if (t[15]) {
                    earlyEnd = std::max(earlyEnd, t[15]);
                    earlyStart = std::min(earlyStart, t[14]);
                }
                movedCount += t[10];
                start = std::min(start, t[4]);
                plan = std::max(plan, t[11]);
                gather = std::max(gather, t[12]);
                finish = std::max(finish, t[6]);
            }
            if (movedCount != moves)
                throw std::runtime_error("per-core moved counters differ from CPU oracle");
            const uint64_t first = earlyEnd ? std::min(earlyStart, start) : start;
            printf("INDEX_WORK early_plan_ticks=%llu post_to_index_join_ticks=%llu "
                   "complete_index_bracket_ticks=%llu\n",
                   (unsigned long long)(earlyEnd ? earlyEnd - earlyStart : 0),
                   (unsigned long long)(finish - start), (unsigned long long)(finish - first));
            printf("MODEL_EXACT_PASS events=%llu moved=%llu "
                   "full_keys_scales_logical_addresses_and_hotvalid=true "
                   "ticks_plan_gather_scatter=%llu/%llu/%llu ticks_total=%llu scope=index_only "
                   "synthetic\n",
                   (unsigned long long)events, (unsigned long long)moves,
                   (unsigned long long)(plan - start), (unsigned long long)(gather - plan),
                   (unsigned long long)(finish - gather), (unsigned long long)(finish - start));
            fflush(stdout);
        }
        Check(aclrtDestroyStream(stream));
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
