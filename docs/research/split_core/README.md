# LIFetch score 分核接入

当前 MR 保留生产 history/BSQ score 分核及直接验证。默认 `score_route="legacy"`，URMA 显式 `auto/history/bsq` 启用；UBmem 保留原入口。

| 文件范围 | 用途 |
|---|---|
| `csrc/operators/common/lifetch/{independent,score,staged}_schedule.h` | 物理预算、独立 score 验证计划、生产 staged 计划 |
| `csrc/operators/lightning_indexer_with_fetch/` | score 与纯 AIV 分离编译、host 启动与完整 tail |
| `python/kvip/operators/lifetch/`、`python/kvip/adapters/vllm/` | 显式路由、live 输入、共享 scratch、预算与 completion |
| `tests/operators/lifetch/` | main 的统一 correctness/bench/diagnose 入口 |
| `tests/runtime/test_lifetch_*.py`、`test_independent_schedule.py` | host binding、实际 C++ 契约、预算与受控 FP8 输入回归 |
| `experiments/split_core/` | 当前 score/staged 的独立 oracle、CPU 检查与 Model 重现工具 |

- [score 分核策略](score_routes.md)
- [调用、资源和完整处理顺序](staged_integration.md)
- [验证命令](../../../experiments/split_core/README.md)
- [精简历史验证摘要](results/score-integration-validation.json)

历史回放、独立 TopK、native 矩阵、trace/adaptive 研究及原始产物已移出当前 MR 文件树，可在提交 `5b006c6651485ab3d9acba83764982c50252ac2e` 恢复。它们包含不同源码、binary 和计量口径，不用于证明当前生产路径提速。当前日志、fixture 和构建结果写入被忽略的 `tests/results/` 或 `build/`，不提交生成产物。

历史 r5 Model 结果仍绑定 `ecf4e8e`，精简摘要保留其版本、哈希和限制。清理及 main 测试重构合入后的检查另行执行；物理 NPU/KVAF、非零 prefetch 和真机性能仍待验收。
