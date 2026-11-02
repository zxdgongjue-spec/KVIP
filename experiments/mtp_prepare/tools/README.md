# Kernel matrix runner

Run commands from this package root. The runner reads local snapshots and never
loads vLLM or Git history. Python requires NumPy and PyTorch for the vendored CPU
reference. CANN simulation runs in the specified Linux container or an explicitly
selected native Linux CANN installation.

Inspect the full Cartesian matrix without running:

```bash
python3 tools/compare_kernels.py --plan
```

Compile and execute a smoke comparison before starting the complete matrix:

```bash
python3 tools/compare_kernels.py --docker \
  --variants v0_base v2_low --cases b1_q4_m300 \
  --warmup 0 --active 1 --output .artifacts/smoke
```

Execute every packaged version on every case, using one isolated build and
process per version. Each active output sample is checked against the CPU oracle.
Build, runtime and oracle failures remain in the report and produce a nonzero
exit; the other matrix cells continue running.

```bash
python3 tools/compare_kernels.py --docker \
  --warmup 1 --active 3 --jobs 1 --timeout 14400 \
  --output .artifacts/full
```

The default image is
`quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12`. Pull it before use. The
Docker image digest, source hashes, input hashes, harness hashes and oracle hash
are recorded. Only the disposable container's simulator configuration may be
changed. Native CANN is checked read-only and must already use `flush_level=2`.

```bash
python3 tools/compare_kernels.py --native --cann /absolute/CANN/path \
  --variants v0_base v2_low --cases b1_q4_m300 --output .artifacts/native-smoke
```

Generate input files and source snapshots without compiling, or recheck an old
matrix without rerunning simulation:

```bash
python3 tools/compare_kernels.py --prepare-only --output .artifacts/prepared
python3 tools/compare_kernels.py --verify-existing /absolute/old/matrix \
  --output .artifacts/reverified
```

Import maps historical names only when the SHA256 tuple of all three kernel
source files matches a packaged variant. Archived source files must also match
their original manifest; missing source snapshots or mismatched hashes are
rejected. Unknown source tuples remain unranked. An old matrix with only
the last output is accepted only when `active=1`; otherwise all active outputs
cannot be verified and the cell fails. The importer does not invent missing
version/case results. New runs write every active output under `samples/N/`.

Use `--profile-enabled 0` for correctness-only execution; those cells have no
performance ranking. Use a separate empty output directory for every command.
`--variants NAME...`, `--cases NAME...` and `--case-file FILE` select focused
matrices. Custom case files can specify heterogeneous `query_counts` and
`miss_counts`; unsupported layout mutations are explicitly rejected.

`results.csv`, `results.json` and `report.md` contain all matrix cells. A failed
kernel cannot win a case. Low-miss percentages use `v0_base`; a case with any
request above 2048 unique misses uses `p0_extended_reference`. Missing or failed
references produce no percentage. Cross-version percentages are positive for
fewer cycles. Different case shapes must not be compared as if they had equal
workloads.

`traces/*.json` are Perfetto-compatible stage views reconstructed from device
duration fields. Their phase positions are not instruction timestamps. The
metric is simulator cycles, with `network=false` and
`device_CQ_executed=false`; simulator host wall time is diagnostic only.

Run the script contract tests without CANN:

```bash
python3 -m unittest discover -s tests -v
```
