# NextGen V4-A entry audit (2026-11-28)

## Freeze scope
F6 E/G1/G2/R1/R2/R3 real-device PASS on physical 7. F6 is an isolated patched SO, not a merge into main. The tracked experiment base 5462f9cb6a2e3cede7ec291120074d9285234fd8 does not itself contain the applied F2/F6 production-source diff. Use the freeze script to verify exact source, candidate and 608 evidence entries and preserve the cumulative diff. Do not replace original production SO or change KVAF. Q4 noncausal independent first-round, H=128K, BS4/16, fixed seed. No stateful, causal, timing, or 512K fused correctness claim.

## Source audit
- `include/runtime_shape.h`: 80-byte ABI; PrepareShape requires history<=210944 and derives liCores=batch*(1+splitQ). Do not relax this limit without workspace/address/transport capacity audit.
- `op_host/launch.cpp`: v9_abi_version=12; shape-controlled launch. Preserve ABI-12 default; a new explicit experimental entry is preferred.
- `op_kernel/indexer/indexer.h` FinishBaseBlock: resident TopK and PrepareRotationProgress at resident boundary; EpochAt maps physical core and S2 loop to service; AIC waits event 8, AIV signals event 8; ProcessTopK at last S2. One physical core can no longer stand for one request-group in a task queue.
- `op_kernel/prefetch.h` FastService/Service: epoch/lane events, predecessor checks, row traversal b=lane+serviceLanes*g. ServiceLane is tied to static geometry. Keep row admission and predecessor closure independent from physical Score ownership.
- `experiments/nextgen_v2_score_queue/b16_score_queue.cpp`: 15 pointers, Task{batch,group,begin,end} half-open 128-token tiles; offsets[33], 32 AIC; physical qlen=4, explicit per-query visible length; history<=524288. No production Context/Rotation/Fetch/TopK publication.
- `experiments/nextgen_v3_local_topk/b16_local_topk.cpp`: 8 pointers, per-task/per-lane 2048 candidates at (task*2+sub)*K, clipped visible end.
- `experiments/nextgen_v3_global_topk/b16_global_topk.cpp`: 9 pointers, one row per AIV, row offsets/IDs and exact contiguous shard cover; inactive row emits -1, single shard copies local output. V3b is independent three-stage Graph, not a fused LI runtime.
- `tests/operators/lifetch/experiments/nextgen_v3b_topk_test.py`: Packed planner offsets/tasks/visible/row tables; 256 tasks, <=8 shards per group, deterministic cutoff tie membership. Count direct baseline must remain measured. V0 documents inherited planner restrictions Q0/Q1 and total Q<56 unsupported.
- `tests/operators/lifetch/support/execution.py`: explicit StreamingLIFetchForward, terminal=None, F6 test reset restores physical cache identity; query_ends repeat length for Q4. Final TopK is arguments[9], Score arguments[12], remote address fixture arguments[13]. No proof of causal physical-prefix correctness after index permutations.

## Proposed V4 sequence
1. V4-A1 host-only ABI/ownership audit: map task ranges and logical row IDs; define ready contiguous prefix per query, independent epoch/lane ownership, final publication and Graph lifetime. Verify no overlapping writes, no holes, no wait on empty tasks. Preserve existing geometry as Count reference.
2. V4-B bridge with explicit alternate route; Score+Local+Global then existing Final Fetch/Index Update. Prefetch disabled explicitly only in this staged bridge, closure must still be satisfied by a defined protocol; no stale metadata or waiting on native event 8. Compare exact TopK, payload, CQ and ownership with F6. No overlap/performance claim.
3. V4-C integrate incremental ready-prefix / resident threshold and Rotation, one writer per metadata row or explicit merge, exactly-once epoch completion, no lost updates. Compare matched work with Count route.
4. V4-D causal and stateful correctness; define logical visibility vs physical index permutations, inactive outputs, row reset, replay ownership. Then 512K resource contract, bounds and real URMA capacity; profile only after correctness.

Do not change production entry, do not silently fall back, do not claim standalone V3b gain for whole-chain. No NPU execution or source compilation in this audit. Pending: detailed runtime binding/Context layout, workspace and ready-prefix protocol, exact event-8 branch analysis on patched source, actual planner upload and launch overhead.
