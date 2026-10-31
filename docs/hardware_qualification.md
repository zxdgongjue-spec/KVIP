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

将两组真实证据分别保存为 JSON，字段为 transport=`kvaf`、execution_device=`npu`、
functional_pass=true、soc/cann/kvaf/model/workload/source_revision/library_sha256，
tpot_ms 为至少五个有限正样本。workload 可用结构化 JSON 描述完整运行配置。

```bash
python tools/qualify_performance.py --baseline baseline.json --candidate candidate.json
```

脚本默认允许回归 0%，明确拒绝 CPU/memory 桩及缺字段数据；性能未通过则保持
manifest 中 production_promotion 的 pending 状态，不自动修改认证或部署配置。
回滚使用原 wheel 和原框架适配并重启 worker，不能在请求/Graph 运行中热替换。
