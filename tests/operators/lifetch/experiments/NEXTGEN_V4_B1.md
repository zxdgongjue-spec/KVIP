# NextGen V4-B1: external TopK final-publication bridge

## Scope
This is the first additive production-code migration slice. It does not replace or modify the existing Generic LI or SequenceSplit launch path. The new finalizer kernel is opt-in (KVIP_ENABLE_NEXTGEN_DYNAMIC=OFF by default).

## Execution contract
The staged caller must already have completed:
1. V2 Score Queue into the logical Score workspace.
2. V3 Local TopK.
3. V3 Global TopK into the existing LIFetch output buffer (bindings[9]).

Before each replay, initialize the existing Context through the existing v9_init path, as the current LIFetch lifecycle requires. Then enqueue Score, Local TopK and Global TopK, and call kvip_nextgen_finalize_and_resolve on the same NPU stream. It:
1. launches one AIV-only finalizer per logical Query;
2. reuses QLIVector::BeginFinalMetadata and EmitFinalClassification to publish queryBitmaps, queryErrors and Context::locations;
3. launches the existing v9_streaming_resolve_update for final Fetch and Index Update.

The finalizer uses streamingTopkOnly_, so it does not wait on native Rotation event 8 and does not launch legacy LI. Intermediate Prefetch must be configured to zero for this bridge; final Fetch remains enabled. Because the current Local/Global kernels do not publish device-side completion status, the caller must synchronize and validate Score, Local TopK and Global TopK outputs before passing upstream_validated=1. This is a correctness-only gate and must not be included in performance timing. V4-B2 must replace this host validation gate with device-visible status before graph/runtime use.

## Admission
Initial gate is intentionally narrow: BS16, Q4, H=131072, R=16384, indexHeads=32, AIC=32, liCores=32, splitQ=1, finalWorkers=4, positive rounds/service lanes, Fetch workers 1/2/4, prefetch_limit=0. This does not relax main's history<=210944 admission and makes no 512K claim.

## Important limitations
- This commit only migrates final publication and the existing final Fetch/Index Update entry. V2/V3 Score/TopK kernels remain in their validated isolated experiment libraries; unified build and Python route wiring are a later V4-B2 step.
- The bridge relies on the staged caller to ensure Global TopK output is exactly the pointer supplied as bindings[9].
- The staged caller must use the same logical query order for Score, Local TopK, Global TopK and finalizer.
- No Prefetch/Rotation overlap or full-chain performance claim is made.
- Compile gate is isolated; no NPU launch is performed by the build script.
