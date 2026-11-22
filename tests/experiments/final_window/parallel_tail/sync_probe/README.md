# Count rendezvous isolation

The kernel publishes counts 1/2/3/4 from four cores, waits on the per-core generation flags, and copies all four counts from UB to GM on every core. `verified.log` records all four rows as `1 2 3 4`, `sync_bad=0`.

This isolates the rendezvous only. It does not prove correctness of the calling tail's bitmap initialization, record lifetime, or output/Index ordering. The successful run uses explicit `pipe.Destroy()`; without it the simulator application terminated before the host verifier, so simulator profiler exit status alone must not be treated as PASS.
