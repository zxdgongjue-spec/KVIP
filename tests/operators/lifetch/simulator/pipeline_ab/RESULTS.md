# Resolver / Index-update experiment status

The experiment is isolated from production code. It models one request, Q4,
131072 history, 16384 resident, 512 prefetch changes, and 346 final misses.
LI, real KV Fetch/CQ, SFA, graph overhead, and per-round mailbox initialization
are outside the measurement. Simulator results are not hardware measurements.

Normal fixture, separate simulator runs using device trace spans:

| Design | AIVs | Span | Breakdown |
| --- | --- | --- | --- |
| A: sequential | resolver 1, planner 1, apply 4 | 308.139 us | resolver 44.571 us; planner 225.583 us; apply 36.857 us; launch gaps included |
| B: concurrent | resolver 1 + updater 1 | 76.319 us | resolver 52.386 us; updater 76.319 us, overlapping |

The B simulated span is 75.23% lower for this isolated fixture; it is not an
end-to-end claim. Serial normal passed data/ownership checks. B normal passed
all those checks plus an independent CPU partial-permutation oracle comparing
every resulting row; no claim of byte-identical entire cold layouts is made.

B device simulator results:

| Fixture | Distinct final misses | Q1..Q4 counts | Consumer trace | Status |
| --- | --- | --- | --- | --- |
| normal | 346 | 141,205,0,0 | 76.319 us | PASS |
| collision: prefetch victim recovered | 347 | 142,205,0,0 | 75.01 us | PASS |
| cross-query | 346 | 141,204,0,1 | 80.82 us | PASS |

All B fixtures check complete row consistency, selected hot ownership, and an
independent per-batch permutation oracle; CPU reference has 15 passing tests.
Producer and consumer each use one AIV. The B consumer statically reserves
221696 bytes of UB under the configured 223232-byte reservation; physical
hardware support and occupancy have not been validated. The current mailbox
must be initialized before use, and this initialization was outside timing.
Performance under repeated graph replays, model workload, and true transport
must still be validated. `run.sh` builds the experimental implementation and
sets a 300-second wall-clock limit for each simulated run.

Logs: `evidence/serial-normal.log`, `evidence/parallel-normal.log`,
`evidence/parallel-collision.log`, `evidence/parallel-cross-query.log`.
