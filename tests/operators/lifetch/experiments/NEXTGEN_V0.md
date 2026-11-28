# NextGen V0: host-only entry gate

Base: 95e1aa38556b2a90ba6f413e0fde64a5877fc750 (M0-C basic PASS). Frozen operator: 79c6a22557ed63a53da687e58f1f5bc1401fd8ed. P6 source: a233563cc11071be12f9bcd0ffaebd98fca714a7.

Two planner files copied byte-for-byte from P6: b16_mtp_cpp_planner.cpp 04ebbb34097f0a5d2806cbc1ed447b91a4a4f892; b16_cpp_planner_fast.cpp 9f7539671035038ccee8301bf3136ea7311a3ef1.

This test compiles CPU-only code, validates packed task/row ownership and physical coverage, records historical model predictions and CPU planner timing. Shape admission uses the actual frozen headers. Host/Device route predicates are mirrors from launch.cpp and indexer.cpp, not device execution proof. Q0/Q1 and total Q<56 are explicitly unsupported by the inherited planner. History >210944 is outside the new fused runtime contract. Do not treat predicted time or CPU planner timing as NPU latency. No operator source, production runtime or service changes. Next: real varlen correctness (initially <=192K) before attaching the planner to device execution. The old planner model requires recalibration on the new device pipeline.
