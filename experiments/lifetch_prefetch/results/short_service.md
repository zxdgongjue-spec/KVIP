# Native P2 Service prefetch smoke

CANN 9.1 Ascend950 model with 4 AIC and 8 AIV, BS2, Q4, history 4096,
resident 2048, W4, and pf512. The test injects compact Q4 candidates into the
native Service path. The first window admits 88 unique tokens per request
and exercises SQ wrap. The second window repeats the candidate IDs.

| Variant | First doorbell | Last doorbell | Service span | Doorbells per request |
| --- | ---: | ---: | ---: | ---: |
| Original P2 | 19,372.5 | 19,406.5 | 20,895.5 | 1 |
| Same-AIV UB candidates | 18,338.5 | 18,761.5 | 19,175.5 | 1 |
| 64-entry asynchronous posting | 18,285.0 | 22,496.0 | 23,170.5 | 2 |

Times are in model ticks from the request's Service entry, averaged over the
two requests in the first window. UB cut the first-doorbell time by 5.34% and
the Service span by 8.23%. Separating the original 88 entries into 64 and 24
advanced the first doorbell by 5.61%, delayed the last by 15.92%, and lengthened
Service by 10.89%.

All three paths passed the WQE source, destination slot, SQ owner bit, and
mapping journal checks. The first window had zero duplicate WQEs. The repeat
window posted zero entries. Each nonempty window had one signaled WQE and one
pending CQE.

The original P2 observations combine two bounded runs. The UB and split runs
completed both checked windows and exited 134 when the CANN model tore down.
An ACL-only control with no device kernel launch reproduced the same
`double free or corruption (fasttop)` on exit. The CSV retains the process
exit status. The model writes simulated doorbells to GM. It covers Service
and its descriptors, with seeded TopK candidates and no URMA transfer,
Final Get, CQ latency, or end-to-end KV-ready measurement.

Source at measurement used the Service changes on MR28 base `443e88a1`
with source tracing off. The integrated version retains the subsequent
source instrumentation from `5db7ef34`. CANN container image
`sha256:556cd79602b2bd97b1d19652432601b9a18e04133a70731c395d44929fc21b8b`.
The matching [fixture and checker](../model/host.cpp) and [run instructions](../README.md)
are included. A5 measurements at BS16, Q4, history 131072 and miss 10% are pending.
