# Final window using the existing prefetch implementation

Run `compare.py SNAPSHOT --kvaf-root KVAF --output NEW_DIRECTORY`.
The snapshot is a single-group `lifetch_failure_v1` dump. Defaults are four
windows, budget 1024, rank start 1024, two warmups and three profiled replays.

A uses the checked-out operator unchanged. B builds an isolated source copy:

- The tail calls the original `RotateFetch` once at the completed history
  endpoint. Its TopK, candidate filtering, union, slot allocation and posting
  code are unchanged. One service AIV processes its assigned requests and Qs.
- The terminal call uses rank 2048 and a per-Q limit of 2048, with the existing
  2048-slot LRU capacity. Intermediate windows retain the command-line settings.
- Existing output-location helpers convert saved TopK IDs into public outputs.
  Existing workers update Index, finish output publication, drain CQ and retire
  state. Two phase barriers connect the window and output adapter to these workers.
- The original resolver and separate final TopK launch are removed from B.
  There is no second TopK, second Fetch or resolver fallback. Residual misses
  after the bounded terminal window fail completion; this is an experiment,
  not an unrestricted-capacity replacement.

The production prefetch, incremental TopK and ranked-candidate files are copied
unchanged. Only the isolated B build is changed; serving libraries are untouched.

## Hardware AB

```bash
python tests/experiments/final_window/compare.py SNAPSHOT \
  --kvaf-root /path/to/KVAF --kvaf-server 127.0.0.1:28885 --device 4 \
  --windows 4 --prefetch-tokens 1024 --prefetch-rank-start 1024 \
  --warmup 2 --iterations 3 --output /tmp/lifetch-final-window-new
```

The runner restores identical captured inputs outside measurement. It checks
completion, bounds, sorted logical TopK and selected KV equality between A/B.
DDR payload is synthetic with captured address aliases. Performance is not
accepted when correctness fails. `ab_summary.json` reports device kernel sums,
not graph-event latency. The historical snapshot runner is loaded from c239791.

Separate diagnostic replays produce `level3.json`, `window_ticks.json` and
`final_detail.log`. The terminal event is epoch `windows`; its TopK, service,
scan, candidate and posting intervals use the existing window stamps. The output
adapter and Index/CQ handoff are separate intervals, not charged to window TopK.
Inactive rows are excluded from aggregate terminal timings.

## Local validation

`reuse/run.py` compares the unmodified window alone on identical intermediate
and terminal candidates. `model/run.py` checks the integrated window, output
adapter and publication with the same 192 per-Q misses (146 unique), then an
ABC/nonaligned-end case. It checks exact TopK, mappings, WQE addresses and output
slots. Both require the CANN Ascend950PR_9599 simulator environment:

```bash
python tests/experiments/final_window/model/run.py \
  --kvaf-include /path/to/KVAF/access_operator/demo/aiv_batch_get/include \
  --output /tmp/lifetch-final-model-new
```

Local model timing excludes real NIC/CQ and downstream Index workers. Vendor
profiler microseconds are not a prediction of hardware end-to-end performance.

The integrated local normal case passes (`mode=5 posted=146 bad=0`). Its
vendor-profiled maximum core duration is 84.93 us, including the output adapter
and handoff. This is a wider scope than the unchanged window-only measurement
(52.71 us); it does not establish a faster complete tail on hardware.
