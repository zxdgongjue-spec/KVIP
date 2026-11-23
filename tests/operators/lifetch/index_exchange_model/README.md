# Overlapping Index exchange regression

This CANN simulator test calls the production Index exchange implementation on
four AIVs. It checks every key word, scale, address and logical ID against an
independent host permutation after each of two launches. Inputs are restored
between launches; publication generations are retained. It needs no KVAF server.

```bash
source /usr/local/Ascend/cann/set_env.sh
cmake -S tests/operators/lifetch/index_exchange_model -B /tmp/index-exchange-build
cmake --build /tmp/index-exchange-build -j4
msprof op simulator --soc-version=Ascend950PR_9599 \
  --output=/tmp/index-exchange-2498 \
  --application='/tmp/index-exchange-build/index_exchange_model 2498 84'
```

Arguments are total changes and the number of hot rows forming a cycle; the
remaining changes are independent cold-to-hot swaps. Useful cases are `230 84`,
`2498 84`, `4000 3000`, and `10240 8000`. The latter two include cycles crossing
worker partitions. Each launch must report four zero statuses and `audit_bad=0`.
The fixed geometry is resident=16384 and history=131328.

These timings cover Index update only, not the complete LIFetch tail or URMA.
Use the existing synthetic/real bench for hardware performance acceptance.

## Validation of the capacity fix

The original UB-only implementation rejects `2498 84` with Index error 2
(terminal status 32), although the host permutation is valid. The GM snapshot
implementation passes both launches, including every payload field.
`4000 3000` (cross-worker hot cycle) and `4097 84` (bitmap membership path)
also pass both launches with four zero statuses and `audit_bad=0`.

The separate complete-tail 230-change simulator fixture, with detailed stamps
disabled and software CQ completion, measured 37.53 us before and 38.08 us after
the fix (+0.55 us, 1.47%). Both updated launches passed its output checks.
These are simulator results, not hardware URMA measurements; only one launch
per run was profiled. Small updates retain the original UB payload path.

Index errors now distinguish invalid records (1), inconsistent closure (2),
and missing/insufficient snapshot space (4); the terminal shifts these by four
bits into its completion status.
