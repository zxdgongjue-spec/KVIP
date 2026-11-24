# LIFetch three-slot batch pipeline

One A5 workload compares native full-batch P2, grouped serial, and grouped pipeline. All three use the same BS16 synthetic fixture with dense Q4, 32 index heads, H128K, R16K, miss 10%, requested W4, prefetch budget 512, and initial rank 1024. The grouped arms split the same requests into four groups of four. They share the same grouped kernels.

## Scheduling

All MIX Score kernels and resident Threshold queue on one compute stream. Each scored window publishes an event after its scores and predicates have been written. The group owner stream waits for that event and launches the AIV-only Service kernel. After the last Service call, the same owner stream launches the native Terminal with the group's KVAF resources.

The pipeline rotates three owner stream slots across the four groups. Slots use exclusive physical KVAF queue ranges `[0,24)`, `[24,48)`, and `[48,72)`. Reusing a slot waits for its previous Terminal completion. Request storage and output buffers stay distinct for the timed invocation and remain alive through CQ completion and correctness checks. The grouped serial arm waits for the previous group's Terminal before the next group's Score.

At C32/V64 a BS4 group uses 8 MIX groups for Score, 4 AIVs for Service, and 16 AIVs for Terminal. Score, Service, and Terminal from three distinct groups have a nominal combined budget of 36 AIVs. Device kernel intervals are required to establish actual overlap. The scheduler also preserves each request's window and Service order.

## Run on A5

Use the existing checkout shared with the A5 container. On the host, prepare the pinned unified KVAF sources and UMDK once if needed.

```bash
bash experiments/lifetch_batch_pipeline/run_a5.sh --prepare-deps
```

In the container, load the matching serving Python environment. Run the repository's dependency build if the pinned KVAF build is missing, then run the comparison.

```bash
bash experiments/lifetch_batch_pipeline/run_a5.sh --build-deps  # only if needed
bash experiments/lifetch_batch_pipeline/run_a5.sh --trace
```

The dependency default is `$KVIP_A5_ROOT/KVAF-unified-813f1750` and the service default is `127.0.0.1:28986`. Set `KVIP_KVAF_ROOT` to reuse an already-built SDK at the pinned revision. Use `KVIP_A5_DEVICE` for the container-visible device ordinal and `KVIP_AB_OUTPUT` to select the results location.

`--comparison-dry-run` prints the fixture and schedule on a CPU-only host. The defaults are two warmups and six measured rounds per arm. `--warmup` and `--iterations` adjust these counts. `--trace` records a separate CPU/NPU profiler invocation for each arm.

## Interpreting the comparison

Compare grouped pipeline with grouped serial to measure the effect of scheduling with the same grouping, Score kernels, window Service, and native Terminal. Compare both with native BS16 P2 to measure the net effects of grouping, extra launches, window timing, and overlap. The whole-batch endpoint is the last group's KV and Index completion. The timing uses device events around eager submission and includes enqueue gaps. Each measured round validates Top2048, KV bytes, and the resulting Index.

`summary.csv` and `summary.json` report p50/p90, relative latency, and the actual per-request prefetch and final Get comparisons. Each arm saves its measured samples, per-window event intervals, and Get counts in `pipeline.json`. Use the optional `npu_trace.json` kernel intervals to verify whether Score, Service, and Terminal overlap on the hardware. The event envelope can include dispatch gaps.
