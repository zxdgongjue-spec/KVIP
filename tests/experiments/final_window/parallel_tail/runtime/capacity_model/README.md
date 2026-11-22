# Capacity and exchange regression evidence

`validation.json` contains simulator results for the retained implementation.
It records the old terminal separately from intermediate and retained versions.
No number in that file is a real URMA hardware measurement.

Run the host correctness/report checks:

```bash
python -m unittest discover -s tests/experiments/final_window/parallel_tail/runtime/capacity_model -p 'test_*.py'
```

Generate a complete native-body model with the same adapters as the live AB
build:

```bash
python tests/experiments/final_window/parallel_tail/runtime/capacity_model/prepare.py /tmp/capacity-230 --changes 230 --candidate
```

Build the generated CMake project in a configured CANN simulator environment,
with `kvaf-include` pointing to patched KVAF device headers. Execute the
`native_tail_sim 9` application using `msprof op simulator` and the matching
SoC. The strict report parser requires both restored-input launches, every
state audit, and a completed four-core profile.

The 230 case has 84 prior assignments and 146 unique final assignments. Cases
2048 through 2100 keep those 146 final assignments and increase prior work.
The 4000 case keeps 1902 prior assignments and uses 2098 final assignments, so
it measures additional payload work as well as capacity handling. It is not a
same-byte test of allocator overhead alone.

The separate resident-source fixture covers cycles crossing worker partitions.
It is not evidence that arbitrarily large overlapping permutations fit the
bounded general-update workspace. Such a request still reports a capacity
error rather than overwriting scratch memory.
