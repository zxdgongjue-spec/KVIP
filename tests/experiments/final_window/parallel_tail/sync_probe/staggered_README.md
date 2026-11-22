# Staggered ready publication regression

Four producers delay publication by 9000 × Q SYS_CNT ticks. Each publishes count Q+1. Every consumer must return all four counts [1,2,3,4].

The former per-lane divergent polling returned partially initialized counts (sync_bad=6). The warp-wide ballot loop returns all counts (sync_bad=0). See staggered_before.log and staggered_after.log. This is a prototype helper regression, not evidence of a production bug.

The native simulator reports ~17.7 us including the deliberate producer delays. This is a correctness stress case, not a posting latency benchmark.
