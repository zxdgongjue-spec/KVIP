// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
extern "C" uint32_t kvop_direct_update_plan_words() { return 8 + 2 * 20480; }
extern "C" uint32_t kvop_direct_update_copy_workers() { return 4; }
extern "C" uint32_t aclrtlaunch_kvop_direct_update_kernel(uint32_t, void*, void*);
extern "C" uint32_t aclrtlaunch_kvop_direct_update_owned_kernel(uint32_t, void*, void*);
extern "C" uint32_t aclrtlaunch_kvop_direct_apply_kernel(uint32_t, void*, void*);
extern "C" uint32_t aclrtlaunch_kvop_direct_apply_owned_kernel(uint32_t, void*, void*);
extern "C" uint32_t kvop_direct_plan(uint32_t batch, void* bindings, void* stream) {
  if (!batch || batch > 32 || !bindings || !stream) return 100000;
  return aclrtlaunch_kvop_direct_update_kernel(batch, stream, bindings);
}
extern "C" uint32_t kvop_direct_apply(uint32_t batch, void* bindings, void* stream) {
  if (!batch || batch > 32 || !bindings || !stream) return 100000;
  return aclrtlaunch_kvop_direct_apply_kernel(batch * 4, stream, bindings);
}
extern "C" uint32_t kvop_direct_update(uint32_t batch, void* bindings, void* stream) {
  auto result = kvop_direct_plan(batch, bindings, stream);
  return result ? result : kvop_direct_apply(batch, bindings, stream);
}
extern "C" uint32_t kvop_direct_update_owned(uint32_t batch, void* bindings, void* stream) {
  if (!batch || batch > 32 || !bindings || !stream) return 100000;
  auto result = aclrtlaunch_kvop_direct_update_owned_kernel(batch, stream, bindings);
  return result ? result : aclrtlaunch_kvop_direct_apply_owned_kernel(batch * 4, stream, bindings);
}

// Extended service ABI; keep the original owned/test ABI unchanged.
// Four TND rows per request are independent elements, so one request can be
// mapped by several AIVs while one of them still plans the Index update.
#include <cstdlib>
constexpr uint32_t kPublishAivBudget = 64, kPublishMaxSplit = 8;
extern "C" uint32_t kvop_direct_publish_split(uint32_t batch) {
  if (!batch || batch > 32) return 0;
  static const uint32_t limit = [] {
    const char* value = std::getenv("KVIP_LIFETCH_PUBLISH_SPLIT");
    if (!value) return kPublishMaxSplit;
    char* end = nullptr;
    const auto split = std::strtoul(value, &end, 10);
    return end != value && *end == '\0' && split && split <= kPublishMaxSplit ? uint32_t(split) : 0U;
  }();
  if (!limit) return 0;
  uint32_t split = limit;
  while (split > 1 && uint64_t(batch) * split > kPublishAivBudget) split >>= 1;
  return split;
}

extern "C" uint32_t aclrtlaunch_kvop_direct_update_outputs_kernel(uint32_t, void*, void*, uint32_t);
extern "C" uint32_t kvop_direct_update_outputs(uint32_t batch, void* bindings, void* stream) {
  if (!batch || batch > 32 || !bindings || !stream) return 100000;
  const uint32_t split = kvop_direct_publish_split(batch);
  if (!split) return 100000;
  auto result = aclrtlaunch_kvop_direct_update_outputs_kernel(batch * split, stream, bindings, split);
  return result ? result : aclrtlaunch_kvop_direct_apply_owned_kernel(batch * 4, stream, bindings);
}

extern "C" uint32_t aclrtlaunch_kvop_direct_update_outputs_early_kernel(uint32_t, void*, void*, uint32_t);
extern "C" uint32_t kvop_direct_update_outputs_early(uint32_t batch, void* bindings, void* stream) {
  if (!batch || batch > 32 || !bindings || !stream) return 100000;
  const uint32_t split = kvop_direct_publish_split(batch);
  if (!split) return 100000;
  auto result = aclrtlaunch_kvop_direct_update_outputs_early_kernel(batch * split, stream, bindings, split);
  return result ? result : aclrtlaunch_kvop_direct_apply_owned_kernel(batch * 4, stream, bindings);
}

extern "C" uint32_t kvop_direct_plan_outputs(uint32_t batch, void* bindings, void* stream) {
  if (!batch || batch > 32 || !bindings || !stream) return 100000;
  const uint32_t split = kvop_direct_publish_split(batch);
  if (!split) return 100000;
  return aclrtlaunch_kvop_direct_update_outputs_kernel(batch * split, stream, bindings, split);
}
extern "C" uint32_t kvop_direct_apply_owned(uint32_t batch, void* bindings, void* stream) {
  if (!batch || batch > 32 || !bindings || !stream) return 100000;
  return aclrtlaunch_kvop_direct_apply_owned_kernel(batch * 4, stream, bindings);
}

extern "C" uint32_t kvop_direct_update_abi_version() { return 3; }

extern "C" uint32_t kvop_direct_update_work_words() { return 2 * 20480; }
