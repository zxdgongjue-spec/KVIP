# Resident-source Index update regression

The cold-only swap rejected source 5732 -> destination 4820 because the source
was below resident=16384. Legal resident sources now use the disjoint swap when
no assigned LRU destination aliases them; overlapping plans reuse the existing
Index permutation planner. Displaced resident KV is invalidated.

Dependency detection uses the existing inverse LRU input and is folded into
Index validation and the existing request-local output-reader join. The packed
arrival/vote word is reset only after the later CQ/Index completion join.
Normal execution has no new global barrier or host-device copy.

Generate a complete native-body software-CQ model:

```bash
python tests/experiments/final_window/parallel_tail/runtime/resident_model/prepare.py /tmp/resident-model --pattern isolated
```

`--pattern chain` and `--pattern cycle` exercise overlapping resident updates.
The driver audits two launches, retaining generation state between launches.
The software CQ models 5 us; this is not an actual URMA bandwidth measurement.
`validation.json` records measured simulator results and the remaining overhead.
The unmodified comparison baseline is commit `32823c0`.

Host permutation and join tests:

```bash
python -m unittest discover -s tests/experiments/final_window/parallel_tail/runtime/resident_model -p 'test_*.py'
```
