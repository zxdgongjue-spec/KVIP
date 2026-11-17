# 全量真实 trace 解析与测试

在 KVIP 根目录运行。解析/导出只需 Python 3.11+、NumPy，不启动 NPU。
默认扫描所有轮次、所有组和所有 DP，不需要选择 round/group/layer。

## 全量扫描与转换

```bash
python scripts/lifetch_score_trace.py \
  --root /home/c00982601/CacheOS/ubmem-validation \
  --output /tmp/lifetch-real-suite
```

输出目录必须是新目录，并位于输入目录树之外。省略 `--output` 只列出清单。

- **READY**：真实 Q/Index 完整，转换为可执行的一轮前态快照。
- **SCORE_ONLY**：分数和状态可分析，缺少 Q/Index，不能执行 LI；保留 CPU 分析结果。
- **FAILED**：记录缺失、格式/校验不通过或无法回放，报告具体原因。
- **EMPTY**：采集计划存在，但没有已登记记录或 manifest，不代表有分数数据。

`status=capturing`、`completed_rounds=0`、`rounds=[]` 和
`awaiting a real scheduled decode batch` 表示尚未登记实际采集轮。
`real_input_rounds=[26,51,76]` 是计划，不能据此宣称这些输入已经保存。
未登记到 series 的 manifest 会列入 `unindexed`，不静默忽略或伪造完整系列。

转换输出：

- `suite.json`：全部来源、记录状态、逐请求 miss 统计、可执行测试清单。
- `case*-r*-g*.log`：每个采集轮/组的详细 CPU 解析或失败原因。
- `case*-r*-g*/`：READY 输入；所有 DP 在运行时合并，最大 BS16。

默认将四层组的合成 KV 投影为单层 656B；真实 Q、Index、权重、分数、初始 hot、
ABC 和页槽编号保持原样。共享组只测试一次，不把共享同一 LI 输入的四层重复计为四份负载。
归档历史搬运记录仍是原组口径；回放搬运量由运行时重新统计。
每个采集轮独立恢复前态，不把跳跃采样误作连续 decode 状态演进。

## 一次运行全部真实输入

```bash
python scripts/lifetch_score_trace.py \
  --run-suite /tmp/lifetch-real-suite \
  --output /tmp/lifetch-real-results \
  -- \
  --transport urma --device 4 \
  --kvaf-root /vllm-workspace/vllm-ascend/third_party/KVAF \
  --kvaf-server 127.0.0.1:28885 \
  --intermediate-fetches 4 --prefetch-tokens 1024 --prefetch-rank-start 1024 \
  --warmup 4 --iterations 20 --timeout-seconds 600
```

自动调用统一入口的 `bench --input real`，保留真实回放正确性校验，比较串行和融合。
单个 case 失败后继续执行其余 case。没有 READY 输入时明确报错，不生成空 PASS。
所有设备任务仍逐 case 启动，当前不跨真实输入复用 DDR 会话。

结果保存在 `suite_results.json` 和汇总 `results.csv`，包括采集轮、组、执行/正确性状态、
串行与融合 mean/p50/p99 和收益。每个 case 的完整终端输出保存在同名 `.log`。

需要详细打点时，在 `--run-suite` 后加 `--mode diagnose`，末尾的测试参数加 `--level 2`
或 `--level 3`。UBmem 使用同一入口，把运行参数换成相应 transport、KVAF 路径和设备即可。
`--` 后只传运行参数；输入选择、group、round-range 和输出路径由清单控制。

## 统计口径与验证

- Demand：captured Q1～Q4 TopK 并集，跨 Q 去重。
- Hot：调用前有效 hot 槽中的需求；ABC-only：仅 ABC 尾部命中的需求。
- Cold：Demand − Hot − ABC-only；总 miss 率按总 cold / 总 demand 计算。
- Candidate hot：有效 hot 命中中位于 LRU 候选槽的数量，不等于实际补取次数。
- Cold w/o DDR：cold 需求中没有有效 DDR 地址的数量，非零不能导出成功。

复用现有 schema-4 的归档哈希、数组形状、TopK 与分数一致性、逻辑映射和页共享校验。
不同 DP 的页空间隔离，同一 DP 的页别名保留。未保存真实 KV，搬运验证使用确定性 C8 KV。
CPU 校验不代替真机正确性验证。

可选单样本兼容接口仍保留：`--root TRACE --round 2 --group 2 --layer-index 0 --output NEW_DIR`。
正常全量测试无需这些选择参数。
