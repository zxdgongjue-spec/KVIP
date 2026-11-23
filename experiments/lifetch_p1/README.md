# LIFetch P1 overlap experiment

P1 runs one Score MIX block per request and publishes each window with a stream event. A second stream advances each query's own Top2048, then runs request-owned Post for cross-query deduplication and prefetch. The next Score window can start as soon as its score stream is ready. Both streams join before Final.

At BS16 with 32 AIC and 64 AIV, each Score window uses 16 AIC and 32 AIV. TopK uses 32 AIV with one pair of queries per worker. Post uses 16 AIV with one request per worker in the four-window configuration. P1 trades Score parallelism for the ability to overlap its next window with TopK or Post. The last Post is exposed on the critical path.

## Scope and source

This directory delivers the executable P1 source as an isolated patch, its host checks, a CANN smoke target, and archived measurements. The patch targets `7b4e3d52d7a85451d1982a6e04823e4e18ecabfb` from `codex/lifetch-score-t2`. It contains the paired-query kernel, host stream submission and tests. The delivery branch uses main `609fa26f` as its review base.

The legacy experiment has a separate Final and Resolve path. Main uses the MR25 Terminal for final Top2048, Resolve/Get and completion. A main-native P1 comparison would need to connect window results to that Terminal with the same state and completion contract.

The submitter validates the prepared host shape, window coverage, AIV budget and event ownership before issuing work. Callers pass a shape that matches the bound device context. They own the resources for both streams and must drain outstanding work before releasing them after a launch error. The host check includes 56 AIV for the front stages. The complete BS16 smoke requires 64 AIV for Threshold and Final.

## Reproduce

From the KVIP source root, obtain the pinned Git object and run the C++17 host contract check. The preparer writes a new standalone tree.

```sh
git fetch origin refs/heads/codex/lifetch-score-t2
python3 experiments/lifetch_p1/validate.py
work=$(mktemp -d)
python3 experiments/lifetch_p1/validate.py --prepare "$work/legacy"
```

Build in Linux with CANN 9.1 and the compatible patched KVAF headers. The KVAF baseline is `1c1310387ebfb6776a07316874e924265b209ada`. Set `cann` to the CANN install root and `kvaf_include` to the patched KVAF `access_operator/demo/aiv_batch_get/include` directory.

```sh
source "$cann/set_env.sh"
soc=Ascend950PR_9599
cmake -S "$work/legacy/experiments/lifetch_score_t2/p1_overlap_model" \
  -B "$work/build" -DRUN_MODE=sim -DSOC_VERSION="$soc" \
  -DASCEND_CANN_PACKAGE_PATH="$cann" -DKVAF_INCLUDE="$kvaf_include"
cmake --build "$work/build" --target p1_overlap_smoke --parallel 2
```

For the BS2 device smoke, select 4 AIC and 8 AIV in an isolated Model configuration. Use `pem.parsim=1` and `pem.parsim_thd_limit=4` in `config_stars.json`, and `sysconf.esl_top_thread_cnt=8` in `camodel_v100.json`. Use separate output directories for the two runs.

```sh
sim="$cann/tools/simulator/$soc"
for mode in serial dual; do
  mkdir -p "$work/$mode/model"
  (
    cd "$work/$mode"
    ln -s /dev/null instr.bin
    export CAMODEL_LOG_PATH="$PWD/model"
    export LD_LIBRARY_PATH="$cann/lib64:$sim/camodel:$sim/lib:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="$sim/lib/libruntime_camodel.so"
    timeout --kill-after=10s 1200 "$work/build/p1_overlap_smoke" \
      0 2 "$mode" 8 4096 4 > run.log 2>&1
  ) || exit $?
  grep '^P1_OVERLAP_PASS ' "$work/$mode/run.log" || exit 1
done
```

The smoke uses a deterministic Score and exact per-query Top2048 oracle, with Post order and buffer guards checked. The four queries select different key components and have different expected Top2048 sets. This compute fixture uses `pf=0`. Positive-prefetch results below come from the separate archived local-GM transport harness with `pf=512`.

## Archived CANN results

Timings are CANN scheduler model ticks. The BS2 paired endpoint is the later of Post0 and Score1. Both arms use the same input and executable.

| Positive-prefetch case | Observation | Completion |
| --- | --- | --- |
| BS2, H4096, miss 5% | P1 single stream 70,187 → dual stream 66,330 ticks, **5.50% shorter**. Both arms verified 74 copied KV tokens. | Compared prefix closed |
| BS16, H3584, miss 5% | Score1 and TopK0 overlap by **7,604 ticks**. All 163,840 Score values match the CPU reference. | Post0 open at timeout |
| BS16, H131072, miss 5% | Score overlaps TopK by **55,378 ticks** and Post by **180,441 ticks**. All **4,256** early copied tokens occur in the final selection. | Final closed, Resolve/Get open at timeout |

[Archived result data](model-results.json) includes the measured intervals and source log fingerprints. The BS16 runs establish overlap and prefetch behavior in the model. The standard-case latency comparison with SerialAfter and MR25 requires matched runs through final KV readiness.
