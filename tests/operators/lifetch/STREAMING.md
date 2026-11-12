# Streaming LIFetch validation

The retained production pipeline is selected automatically.
Use [the unified test design](../../../docs/performance/lifetch-tests.md) and
`python tests/operators/lifetch/test_lifetch.py --help`.

The former comparison scripts and optimization switches have been retired.
Bench compares native serial LI/LIDU/Get with the current fused implementation;
diagnose retains separately profiled stage and core records.
