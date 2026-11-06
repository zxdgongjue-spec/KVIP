// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
extern "C" uint32_t kvop_direct_update_plan_words() { return 8 + 2 * 4096; }
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
extern "C" uint32_t aclrtlaunch_kvop_direct_update_outputs_kernel(uint32_t, void*, void*);
extern "C" uint32_t kvop_direct_update_outputs(uint32_t batch, void* bindings, void* stream) {
  if (!batch || batch > 32 || !bindings || !stream) return 100000;
  auto result = aclrtlaunch_kvop_direct_update_outputs_kernel(batch, stream, bindings);
  return result ? result : aclrtlaunch_kvop_direct_apply_owned_kernel(batch * 4, stream, bindings);
}

extern "C" uint32_t aclrtlaunch_kvop_direct_update_outputs_early_kernel(uint32_t, void*, void*);
extern "C" uint32_t kvop_direct_update_outputs_early(uint32_t batch, void* bindings, void* stream) {
  if (!batch || batch > 32 || !bindings || !stream) return 100000;
  auto result = aclrtlaunch_kvop_direct_update_outputs_early_kernel(batch, stream, bindings);
  return result ? result : aclrtlaunch_kvop_direct_apply_owned_kernel(batch * 4, stream, bindings);
}

extern "C" uint32_t kvop_direct_plan_outputs(uint32_t batch, void* bindings, void* stream) {
  if (!batch || batch > 32 || !bindings || !stream) return 100000;
  return aclrtlaunch_kvop_direct_update_outputs_kernel(batch, stream, bindings);
}
extern "C" uint32_t kvop_direct_apply_owned(uint32_t batch, void* bindings, void* stream) {
  if (!batch || batch > 32 || !bindings || !stream) return 100000;
  return aclrtlaunch_kvop_direct_apply_owned_kernel(batch * 4, stream, bindings);
}

extern "C" uint32_t kvop_direct_update_abi_version() { return 2; }
