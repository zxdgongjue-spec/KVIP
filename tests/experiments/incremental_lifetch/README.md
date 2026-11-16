# 增量位图 TopK：URMA 端到端 A/B

基线保留当前融合 LIFetch。实验将每个预取窗口的新增位图候选与已有 Top2048
合并，保存分数、原 token ID 和每个序列分片的处理边界。最终 TopK 接续处理
尚未纳入的区间，不重新扫描已处理历史。中间预取按更新后的每 Q 阈值过滤候选；
等于阈值的候选保留，最终 TopK 的并列分数按现有正确性检查验证。

压缩和 ID 还原直接复用 `incremental_topk` demo 抽出的
`csrc/operators/common/lifetch/incremental_compact.h`；TopK 仍调用
`topk::LITopk<uint16_t>`。LI 位图编码、核分配、Resolver、Index 更新和 KVAF
提交/完成路径保持现有实现。原始分数和位图只读取已发布区间，不改写生产者位图。

跨窗口保存在原 hot TopK 的 GM 区域，原 hot 阈值另存且不修改。每 Q 另有
128 字节的来源指针、双分片游标及当前阈值，随 round init 清零。窗口服务核
沿用当前请求分配；前序服务未完成时立即跳过，不推进消费游标。后续窗口跨过
已跳过的事件，从最近实际完成的游标接续；最终 TopK 等待仍在途的状态并补齐。
原始分数仍按现有 LI 的 8192/16384 分块容量加载，但压缩候选跨分片累积，
在候选缓冲装满、相对 ID 跨度超过65536或窗口结束时才做 TopK。压缩前过滤低于
已有 TopK 阈值的候选。核心 tiling 不变。

额外 UB 只保存 token ID 映射，原 LI 缓冲区偏移不变。最终 TopK 阶段复用尚未使用
的分类输出缓冲作为旧 ID 缓冲；heads32 的该阶段总分配为 221216 字节，低于
223232 字节预算。没有逐 bit 标量展开，也没有逐候选 GM 原子操作。

## 运行

在 KVIP 根目录，初始化 CANN 和现有依赖环境后：

```bash
bash tests/experiments/incremental_lifetch/compare.sh \
  /vllm-workspace/vllm-ascend/third_party/KVAF \
  4 127.0.0.1:28885 1,2,4,8,16 10
```

参数依次为 KVAF 根目录、device、server、BS 列表、单个 miss 百分比。
脚本为实验复制独立源码目录；不会覆盖原算子。两边均运行统一入口的
`diagnose --level 3`，包含无打点完整图计时、正确性检查、独立打点 replay
和阶段日志。每版编译缓存跨 BS 复用，结果位于 `tests/results/incremental-lifetch/`。

本实验入口用于 URMA、Q4、heads32、H131072/R16384 的合成负载对比；
不是 UBmem 或模型服务适配。未运行本地仿真，真机正确性和性能由上述 A/B 验证。

## 提炼已有日志（无需 NPU）

```bash
# 默认最近一次 A/B，也可在末尾指定某次结果目录。
python tests/experiments/incremental_lifetch/summarize.py
```

输出正确性、完整图 mean/p50/p99、阶段耗时 A/B 差值、预取/补取/无用读取和
冲突数量，保存为该次结果目录的 `key_results.log`。阶段使用全部有效打点轮次
的平均值（包括 round 0）；独立于无打点图计时。中途失败时保留已有一侧结果。

新版本汇总还显示每窗口 active/skipped lane、TopK 调用总数和最慢 lane 的增量处理时间，来自独立打点轮次。
