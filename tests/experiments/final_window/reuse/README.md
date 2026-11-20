# Unmodified intermediate-window reuse

This harness directly calls the production `QLIVector::RotateFetch` entry point.
It does not patch `AdvanceWindow`, `SelectBitmap`, `Service`, `FastService`, or
any other production kernel. `run.py` links the checked-out `csrc` directory.

Both cases use one AIV and identical Q4 scores, prior TopK, 192 miss candidates,
146 distinct requests, and 84 occupied LRU slots. All score producers are ready
through the last window. The final case changes only input configuration:
`windowLimit=8192` gives each Q a 2048-candidate ceiling, and rank start 2048
keeps all misses selected by TopK. The existing 2048-slot capacity still applies.
The third case checks ABC hits and nonaligned per-Q ends.

```bash
python tests/experiments/final_window/reuse/run.py \
  --kvaf-include /path/to/KVAF/access_operator/demo/aiv_batch_get/include \
  --output /tmp/lifetch-window-reuse-new
```

Run inside the CANN environment with the Ascend950PR_9599 simulator library in
`LD_LIBRARY_PATH`. The report uses vendor-profiler kernel microseconds and raw
internal counter ticks, without assuming a counter frequency.

Validation checks the exact saved TopK, mapping assignments, all WQE source and
destination addresses, request count and doorbell count. It measures TopK and
submission only. It does not implement public output conversion, run the old
resolver, or simulate NIC/CQ completion. The older `compare.py` end-to-end B is
unchanged; this harness must not be presented as an end-to-end replacement.

## Local result (2026-11-20)

Ascend950PR_9599 device model, identical candidate fixture:

| Phase | Intermediate | Final |
| --- | ---: | ---: |
| Whole kernel, vendor profiler us | 52.71 | 52.71 |
| TopK, ticks | 53899 | 53899 |
| Service, ticks | 33428 | 33428 |
| Scan, ticks | 16697 | 16697 |
| Candidate processing, ticks | 27 | 27 |
| Posting, ticks | 11927 | 11927 |
| Distinct requests posted | 146 | 146 |

Both exact TopK and WQE/mapping checks pass. The separate ABC/nonaligned-end
fixture also passes (84.21 us); it has different inputs and is not a performance
comparison against the first fixture. These are simulator results, not NIC or
end-to-end NPU measurements.
