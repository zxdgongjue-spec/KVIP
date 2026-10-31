# 生产合入前的硬件验收

当前主工作区没有激活本次框架适配。以下测试在匹配 SDK/SoC 的 A5 环境完成后才允许
声明真实功能/性能无影响。CPU oracle、source diff、memory 桩均不能替代此门槛。

1. 按根 README 构建基线与候选，固定相同 CANN/KVAF/SoC、AIC、B/Q/C/S、窗口、
   环境和模型；记录 source revision、library SHA256、SDK 版本与完整命令。
2. 先跑 NPU memory correctness，覆盖非连续页、tail、victim hot、阈值相等、重复
   miss、extra、跨步/层/换序/源改写。不得报告网络加速比。
3. 跑真实 KVAF eager 与 Graph，确认 SQ/CQ、资源释放、错误隔离、未知完成、跨 rank
   可见性；frame 失败不得交付输出。
4. 按相同 warmup、并发与输入样本跑 V9 基线/候选及原生 fallback，确认模型 accuracy、
   selection/feedback 与对应 profile 一致，再采 TPOT 和 selection latency。
5. 重复交错运行排除热身/温度/时钟偏差。默认要求无回归；若需放宽测量容差，应明确
   记录依据，不自动选择有利样本。脚本仅比较输入证据，不证明日志真实性或统计显著性。

将两组真实证据分别保存为 JSON，字段如下。此工具只验证提交证据，不采集硬件数据。

| 字段 | 要求 |
| --- | --- |
| transport / execution_device | kvaf / npu，禁止 memory 或 CPU 桩 |
| functional_pass | 字面 true |
| soc / cann / kvaf / model / workload | 明确实际硬件、SDK、模型和完整 workload；两组一致 |
| source_revision / library_sha256 / command | 实测源码版本、二进制 SHA256 和完整采集命令 |
| measurement_config | warmup_iterations≥1、measurement_iterations≥100、repetitions≥3、sampling_policy=all-measured-samples、明确 timing_method；两组完全相同 |
| tpot_ms / selection_latency_us | 按采集顺序保留的有限正样本；各自数量等于 measurement_iterations |
| evidence_source.kind | real-npu-kvaf |
| evidence_source.raw_log / functional_report | 各自包含 path、sha256；相对路径以对应证据 JSON 所在目录解析 |

measurement_iterations 是本次比较的全部样本总数，至少覆盖三次交错重复运行；采集者
须在原始日志中记录每个 repetition 和 warmup 边界，不能只提交各轮中位数或筛选快样本。
workload 应包括并发、B/Q/C/S、窗口、页布局、head 数及测量模式。首次准备、page metadata
快照、锁与 completion 的 host 成本也应另行报告，不与重叠设备阶段直接相加。

functional_report 是 JSON，包含 functional_pass=true，与该记录一致的 transport、
execution_device、硬件/SDK/model/workload/source_revision/library_sha256，以及 checks
中的 accuracy、selection、feedback、lifecycle、eager、graph 全部为字面 true。
原始日志与报告的 SHA256 必须匹配实际文件。合成测试 fixture 只验证门槛逻辑。

```bash
python tools/qualify_performance.py --baseline baseline.json --candidate candidate.json
```

脚本对 TPOT 和 selection latency 各自的 median、p95、p99 分别比较，所有指标都需
通过；p95/p99 使用 nearest-rank，median 为偶数样本中间两项均值。默认允许回归 0%，
明确拒绝 CPU/memory 桩、缺字段、缺日志、错误 hash、未通过的功能报告及配置不一致。
样本下限是工程门槛，不是统计显著性证明；仍应根据抖动和服务 SLO 增加样本。
性能未通过则保持
manifest 中 production_promotion 的 pending 状态，不自动修改认证或部署配置。
回滚使用原 wheel 和原框架适配并重启 worker，不能在请求/Graph 运行中热替换。
