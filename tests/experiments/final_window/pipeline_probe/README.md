# Per-query final-window feasibility probes

These are isolated simulator modules, not a production implementation or a hardware end-to-end benchmark. Production files are copied to a new output directory and remain unchanged.

## Fixture

- History 131328, resident 16384, four Top2048 queries, one active request.
- Candidate counts 11 / 0 / 94 / 87, with 146 unique new tokens and 84 preceding admissions.
- LRU slots start at 10000, with sufficient spare capacity.
- Counts and cross-query overlap follow the supplied trace. Score values and token identities are constructed, not the server's captured tensors.
- The output probe also covers tagged hot hits, ABC slots and nonidentity logical IDs.
- Index sources and destinations are disjoint; exhausted-LRU/general permutation fallback is outside this fixture.

## Modules

- `group`: unchanged window TopK/filter/union/post algorithms, four Qs then one service.
- `split`: one persistent TPipe, four complete window invocations, one Q per invocation. Deliberately measures repeated-wrapper costs.
- `hoisted`: loads window progress/metadata once; runs TopK then unchanged service per Q. Next-Q preloading is disabled because service shares its scratch buffers. Selection and WQE algorithms are unchanged.
- `output`: directly produces logical IDs and slots in one pass, avoiding the separate bitmap/compact-list and second output-worker traversal.
- `index`: original Index swap helpers, prefix and each Q separately, address/logical metadata once at the end. Metadata must remain stable until all Q outputs and address reads finish.

The `8191` window-limit sentinel exists only in generated probe code to select scheduling. Its per-Q cap is 2047 rather than 2048; fixture candidates never exceed 94, so this does not affect selection.

## Run

Inside a configured CANN 9.1 simulator environment, with simulator libraries on `LD_LIBRARY_PATH`:

```bash
bash tests/experiments/final_window/pipeline_probe/run.sh hoisted /tmp/new-hoisted-probe /tmp/window-sync-perf/kvaf-include
bash tests/experiments/final_window/pipeline_probe/run.sh output /tmp/new-output-probe /tmp/window-sync-perf/kvaf-include
bash tests/experiments/final_window/pipeline_probe/run.sh index /tmp/new-index-probe /tmp/window-sync-perf/kvaf-include
```

Times come from `msprof op simulator` and its instruction timeline, not an assumed hardware clock. `extract.py` matches device stamps to `SYS_CNT` instruction timestamps. Simulator times must not be presented as measured NPU times.

Validation checks TopK sets, mapping uniqueness, WQE addresses/destinations and doorbell count; output checks all 8192 logical IDs/slots; Index checks both sides of every key/scale/address/identity swap.
