// SPDX-License-Identifier: Apache-2.0
#include "context.h"
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace rotating;
void check(uint32_t code, const char *what) {
  if (code) {
    std::cerr << what << " returned " << code << std::endl;
    throw std::runtime_error(std::string(what) + ": " + std::to_string(code));
  }
}
struct Memory {
  std::vector<void *> alloc;
  uint64_t put(const void *p, size_t n) {
    void *d = nullptr;
    check(aclrtMalloc(&d, n, ACL_MEM_MALLOC_HUGE_FIRST), "allocate");
    alloc.push_back(d);
    check(aclrtMemcpy(d, n, p, n, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    return reinterpret_cast<uint64_t>(d);
  }
  template <class T> uint64_t vec(const std::vector<T> &v) {
    return put(v.data(), v.size() * sizeof(T));
  }
  uint64_t zero(size_t n) { return vec(std::vector<uint8_t>(n)); }
  ~Memory() {
    for (auto d : alloc)
      aclrtFree(d);
  }
};
int main(int argc, char **argv) {
  if (argc != 4)
    return 2;
  const unsigned replays = std::stoul(argv[3]);
  if (!replays || replays > 2)
    return 2;
  bool passed = true;
  check(aclInit(nullptr), "init");
  check(aclrtSetDevice(0), "set device");
  aclrtStream stream;
  check(aclrtCreateStream(&stream), "stream");
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    std::cerr << "dlopen failure " << dlerror() << std::endl;
    return 2;
  }
  using launch_fn = uint32_t (*)(uint32_t, void *, void *);
  auto launch =
      reinterpret_cast<launch_fn>(dlsym(library, "aclrtlaunch_v9_li_kernel"));
  if (!launch)
    throw std::runtime_error(dlerror());
  constexpr uint32_t B = 2, R = 2048, H = 5120, D = 128, Heads = 32, Q = B * 4,
                     Cores = B * 2;
  {
    Memory m;
    std::vector<uint8_t> q(Q * Heads * D), k(B * H * D);
    std::vector<uint16_t> w(Q * Heads);
    std::vector<float> qs(Q * Heads);
    const char *fixtureOption = std::getenv("KVIP_MODEL_SCORE_FIXTURE");
    const std::string fixture = fixtureOption ? fixtureOption : "default";
    if (fixture != "default" && fixture != "adjacent-cutoff" && fixture != "sparse-first-window")
      throw std::runtime_error("unknown score fixture: " + fixture);
    const bool closeScores = fixture == "adjacent-cutoff";
    const bool sparseFirst = fixture == "sparse-first-window";
    auto scaledRow = [closeScores, sparseFirst](unsigned row) {
      return closeScores || (sparseFirst && row == 0);
    };
    auto querySign = [scaledRow](unsigned query, unsigned h) {
      if (scaledRow(query / 4))
        return h == query % Heads ? 1 : -1;
      return (h + query) % 3 == 0 ? -1 : 1;
    };
    auto keySign = [scaledRow](unsigned row, unsigned token) {
      return !scaledRow(row) && token % 3 == 0 ? -1 : 1;
    };
    auto weightSign = [scaledRow](unsigned query, unsigned h) {
      if (scaledRow(query / 4) && h == query % Heads)
        return 1;
      return (query + 2 * h) % 5 < 2 ? -1 : 1;
    };
    auto magnitude = [scaledRow](unsigned row, unsigned token) {
      return 1 + unsigned(!scaledRow(row) && token >= R);
    };
    for (unsigned query = 0; query < Q; query++)
      for (unsigned h = 0; h < Heads; h++) {
        std::fill_n(&q[(query * Heads + h) * D], D,
                    querySign(query, h) < 0 ? 0xb8 : 0x38);
        w[query * Heads + h] = weightSign(query, h) < 0 ? 0xbf80 : 0x3f80;
        qs[query * Heads + h] = scaledRow(query / 4) ? 1 : 1 + (h % 3);
      }
    std::vector<float> keyScales(B * H, 1.0f);
    for (unsigned row = 0; row < B; row++)
      for (unsigned t = 0; t < H; t++) {
        if (sparseFirst && row == 0) {
          // Seed with 1536 equal minimum scores. Across the entire cold
          // history only 384 tokens exceed that score, so any first
          // nonempty strict pack preserves the seed cutoff for all four Qs.
          unsigned target = 128;
          if (t < R)
            target += t % 4 == 0;
          else if (t % 8 == 0)
            target += 1 + ((t - R) / 512) % 2;
          keyScales[row * H + t] = float(target) / 128.0f;
        } else if (closeScores) {
          // The sole active head gives an exact dot product of 128. Each
          // integer in [128, 136] is one BF16 step from the next integer.
          unsigned high = row == 0 ? 129 : 131;
          unsigned target = t % 8 == 0 ? 128 : high;
          if (t >= R) {
            unsigned slab = (t - R) / 512;
            target = high + (slab == 0 ? 0 : slab < 3 ? 1 : 3);
          }
          keyScales[row * H + t] = float(target) / 128.0f;
        }
        uint8_t value = magnitude(row, t) == 1 ? 0x38 : 0x40;
        if (keySign(row, t) < 0)
          value |= 0x80;
        std::fill_n(&k[(row * H + t) * D], D, value);
      }
    std::vector<int32_t> blocks(B * H / 128);
    for (unsigned i = 0; i < blocks.size(); i++)
      blocks[i] = i;
    Context c{};
    auto &s = c.shape;
    s.batch = B;
    s.history = H;
    s.resident = R;
    s.indexHeads = Heads;
    s.aicCount = Cores;
    s.splitQ = 1;
    s.rounds = 4;
    s.tokenBytes = 656;
    s.serviceAics = 1;
    if (PrepareShape(&s) || !UseSequenceSplit(s) || s.rounds != 4 ||
        s.serviceLanes != B)
      throw std::runtime_error("wrong shape");
    std::array<uint64_t, 16> a{};
    a[0] = m.vec(q);
    a[1] = m.vec(k);
    a[2] = m.vec(w);
    a[3] = m.vec(qs);
    a[4] = m.vec(keyScales);
    a[5] = m.vec(std::vector<int32_t>(B, 4));
    a[6] = m.vec(std::vector<int32_t>(B, H));
    a[7] = m.vec(blocks);
    a[8] = a[10] = a[11] = m.zero(64);
    a[9] = m.zero(Q * 2048 * 4);
    a[12] = m.zero(Q * H * 4 + 1024 * 1024);
    constexpr unsigned seeds = 4 * 2048 * 6 + 1024;
    a[13] = m.zero(B * (seeds + 4 * H * 2 + 8192 * 64 + H * 8 + 512));
    c.hits = a[13] + B * seeds;
    c.progress = m.zero(Q * sizeof(Progress));
    c.outputStride = R;
    c.incremental =
        m.zero(B * (512 + seeds + (H + 255) / 256 * 32 + 4 * 32 * 8));
    c.locations = m.zero(B * 8192 * 4);
    c.windowLimit = 256;
    c.prefetchLimit = 1024;
    c.prefetchRankStart = 1024;
    c.events = m.zero(4 * B * sizeof(Event));
    c.rowStatus = m.zero(B * 128);
    std::vector<uint32_t> hot(B * R, 1), lru(B * 2048), mapping(B * H);
    std::vector<uint64_t> addresses(B * H);
    for (unsigned b = 0; b < B; b++) {
      // Four and 103 are median per-Q eligible counts in the archived G2 and G7
      // window fixtures.
      for (unsigned t = 0; t < R; t++)
        hot[b * R + t] = ((t * 31 + 7) % 2048 >= (b == 0 ? 4U : 103U));
      for (unsigned t = 0; t < 2048; t++)
        lru[b * 2048 + t] = t;
      for (unsigned t = 0; t < H; t++) {
        mapping[b * H + t] = (t < R && hot[b * R + t]) ? t : UINT32_MAX;
        addresses[b * H + t] = 0x1800000000ULL + (b * H + t) * 656ULL;
      }
    }
    c.hotValid = m.vec(hot);
    c.lruSlots = m.vec(lru);
    c.reservedHot = m.vec(std::vector<uint32_t>(B * R, UINT32_MAX));
    c.mapping = m.vec(mapping);
    c.tokenAddresses = m.vec(addresses);
    c.addressStride = H;
    c.outputStride = R;
    constexpr uint32_t depth = 4096;
    std::array<KvafBatchGetDirectUrmaEntity, B> entities{};
    std::array<KvafBatchGetQueueState, B> queues{};
    std::vector<KvafExecuteBatchGetKernelArgs> request(2 * B);
    std::vector<Row> rows(B);
    for (unsigned b = 0; b < B; b++) {
      auto &e = entities[b];
      auto &req = request[2 * b];
      auto &row = rows[b];
      e.sq_va = m.zero(depth * 64);
      e.sq_db_va = m.zero(128);
      e.sq_depth = e.cq_depth = depth;
      e.tp_id = 0x1234;
      e.remote_token_id = 0x54321;
      e.remote_token_value = 0x2468ace0;
      e.local_token_id = 0x12345;
      req.channel_entity = m.put(&e, sizeof(e));
      req.queue_state = m.put(&queues[b], sizeof(queues[b]));
      req.core_status = m.zero(sizeof(KvafBatchGetCoreStatus));
      req.output = 0x1200000000ULL;
      row.entity = req.channel_entity;
      row.queue = req.queue_state;
      row.output = req.output + b * R * 656ULL;
      row.addresses = c.tokenAddresses + b * H * 8;
      row.limit = row.prefetchLimit = 1024;
      row.queueLimit = depth;
    }
    c.args = m.vec(request);
    std::vector<uint8_t> rowmem(B * (sizeof(Row) + 512));
    std::memcpy(rowmem.data(), rows.data(), B * sizeof(Row));
    c.rows = m.vec(rowmem);
    a[14] = m.put(&c, sizeof(c));
    a[15] = m.zero(65536);
    void *bindings = reinterpret_cast<void *>(m.put(a.data(), sizeof(a)));
    auto seed = (launch_fn)dlsym(library, "aclrtlaunch_v9_li_kernel_1");
    auto cold = (launch_fn)dlsym(library, "aclrtlaunch_v9_li_kernel_2");
    if (!seed || !cold)
      throw std::runtime_error(dlerror());
    for (unsigned replay = 0; replay < replays; ++replay) {
      if (replay) {
        auto restore = [&](uint64_t address, const void *src, size_t n) {
          check(aclrtMemcpy(reinterpret_cast<void *>(address), n, src, n,
                            ACL_MEMCPY_HOST_TO_DEVICE),
                "restore replay state");
        };
        auto clear = [&](uint64_t address, size_t n) {
          std::vector<uint8_t> zeros(n);
          restore(address, zeros.data(), n);
        };
        clear(c.events, 4 * B * sizeof(Event));
        clear(c.progress, Q * sizeof(Progress));
        // Keep the previous phase value in words [8..15] of every Q header,
        // while reinitializing the ordinary query and WQE business state.
        std::vector<uint64_t> hdr(Q * 16);
        check(aclrtMemcpy(hdr.data(), hdr.size() * 8, (void *)c.incremental,
                          hdr.size() * 8, ACL_MEMCPY_DEVICE_TO_HOST),
              "read old phase");
        for (unsigned query = 0; query < Q; query++) {
          if (std::getenv("EXPECT_FINAL_MAIL") && hdr[query * 16 + 8] != 2)
            throw std::runtime_error("missing old completed phase");
          std::fill_n(hdr.data() + query * 16, 8, 0);
        }
        restore(c.incremental, hdr.data(), hdr.size() * 8);
        // Clear seeds and coverage but retain the old helper offer/completion
        // and assignment in words [8,24) of every Q's trailing rank row.
        clear(c.incremental + B * 512, B * (seeds + (H + 255) / 256 * 32));
        auto rankOffset = B * (512 + seeds + (H + 255) / 256 * 32);
        std::array<uint64_t, Q * 32> previousRank{};
        check(aclrtMemcpy(previousRank.data(), sizeof(previousRank),
                          reinterpret_cast<void *>(c.incremental + rankOffset),
                          sizeof(previousRank), ACL_MEMCPY_DEVICE_TO_HOST),
              "read old helper state");
        unsigned completedHelpers = 0;
        for (unsigned query = 0; query < Q; ++query) {
          auto *words = previousRank.data() + query * 32;
          completedHelpers += words[8] == 1 && words[9] == 1 && words[16] == 1;
          std::fill_n(words, 8, 0);
          std::fill_n(words + 24, 8, 0);
        }
        if (std::getenv("EXPECT_IDLE_HELPERS") && completedHelpers == 0)
          throw std::runtime_error("second replay has no stale helper "
                                   "assignment and completion to exercise");
        restore(c.incremental + rankOffset, previousRank.data(),
                sizeof(previousRank));
        std::cout << "STALE_FINISHED_HELPERS=" << completedHelpers << std::endl;
        restore(c.rows, rowmem.data(), rowmem.size());
        restore(c.mapping, mapping.data(), mapping.size() * 4);
        std::vector<uint32_t> reserved(B * R, UINT32_MAX);
        restore(c.reservedHot, reserved.data(), reserved.size() * 4);
        clear(c.locations, B * 8192 * 4);
        clear(c.rowStatus, B * 128);
        clear(a[12], Q * H * 4 + 1024 * 1024);
        clear(a[13], B * (seeds + 4 * H * 2 + 8192 * 64 + H * 8 + 512));
        for (unsigned b = 0; b < B; b++) {
          clear(request[2 * b].queue_state, sizeof(KvafBatchGetQueueState));
          clear(entities[b].sq_db_va, 128);
          clear(entities[b].sq_va, depth * 64);
        }
      }
      check(launch(Cores, stream, bindings), "launch resident");
      check(seed(Cores, stream, bindings), "launch seed");
      check(cold(Cores, stream, bindings), "launch cold");
      check(aclrtSynchronizeStream(stream), "sync scoring and posting");
      auto get = [&](uint64_t addr, void *out, size_t n) {
        check(aclrtMemcpy(out, n, (void *)addr, n, ACL_MEMCPY_DEVICE_TO_HOST),
              "D2H");
      };
      auto dump = [&](const std::string suffix, const void *out, size_t n) {
        std::ofstream f(std::string(argv[2]) +
                            (replay == 0 ? "" : ".r" + std::to_string(replay)) +
                            suffix,
                        std::ios::binary);
        f.write(static_cast<const char *>(out), n);
      };
      std::vector<uint16_t> score(Q * H);
      get(a[12], score.data(), score.size() * 2);
      dump("", score.data(), score.size() * 2);
      std::vector<uint8_t> seedOut(B * seeds);
      get(a[13], seedOut.data(), seedOut.size());
      dump(".seeds", seedOut.data(), seedOut.size());
      std::vector<uint8_t> finalWindow(B * seeds);
      get(c.incremental + B * 512, finalWindow.data(), finalWindow.size());
      dump(".window", finalWindow.data(), finalWindow.size());
      std::vector<uint64_t> meta(Q * 16);
      get(c.incremental, meta.data(), meta.size() * 8);
      dump(".meta", meta.data(), meta.size() * 8);
      std::array<uint64_t, Q * 32> rankWords{};
      auto rankOffset = B * (512 + seeds + (H + 255) / 256 * 32);
      get(c.incremental + rankOffset, rankWords.data(), sizeof(rankWords));
      dump(".rank", rankWords.data(), sizeof(rankWords));
      std::vector<Event> events(4 * B);
      get(c.events, events.data(), events.size() * sizeof(Event));
      dump(".events", events.data(), events.size() * sizeof(Event));
      auto mapped = mapping;
      get(c.mapping, mapped.data(), mapped.size() * 4);
      dump(".map", mapped.data(), mapped.size() * 4);
      get(c.rows, rows.data(), B * sizeof(Row));
      unsigned correct = 0;
      for (unsigned qi = 0; qi < Q; qi++)
        for (unsigned t = 0; t < H; t++) {
          float val = 0;
          for (unsigned h = 0; h < Heads; h++)
            val += weightSign(qi, h) * qs[qi * Heads + h] *
                   std::max(0, querySign(qi, h) * keySign(qi / 4, t) *
                                   int(D * magnitude(qi / 4, t)));
          val *= keyScales[(qi / 4) * H + t];
          uint32_t u;
          std::memcpy(&u, &val, 4);
          uint16_t bits = u >> 16;
          uint16_t sortable =
              (bits & 0x8000) ? uint16_t(~bits) : uint16_t(bits ^ 0x8000);
          correct += score[qi * H + t] == sortable;
        }
      std::cout << "SCORE_MATCH=" << correct << "/" << Q * H << std::endl;
      bool ok = correct == Q * H;
      for (unsigned b = 0; b < B; b++) {
        get(request[b * 2].queue_state, &queues[b], sizeof(queues[b]));
        auto &row = rows[b];
        auto count = row.requests;
        std::vector<uint32_t> sq(count * 16);
        if (count)
          get(entities[b].sq_va, sq.data(), sq.size() * 4);
        dump(".sq" + std::to_string(b), sq.data(), sq.size() * 4);
        std::cout << "ROW=" << b << " QUEUE=" << queues[b].sq_producer_index
                  << " POSTS=" << count << " EPOCHS=";
        for (unsigned e = 0; e < 4; e++)
          std::cout << (e ? "," : "") << row.epochCount[e];
        std::cout << " LAST_Q=";
        for (unsigned q = 0; q < 4; q++)
          std::cout << (q ? "," : "") << meta[(b * 4 + q) * 16 + 3];
        std::cout << std::endl;
        ok = ok && count > 0 && count == queues[b].sq_producer_index &&
             row.epochCount[3] > 0;
      }
      passed &= ok;
      std::cout << "REPLAY=" << replay
                << " VALIDATION=" << (ok ? "PASS" : "FAIL") << std::endl;
    }
  }
  check(aclrtDestroyStream(stream), "destroy");
  check(aclrtResetDevice(0), "reset");
  check(aclFinalize(), "finalize");
  return passed ? 0 : 1;
}
