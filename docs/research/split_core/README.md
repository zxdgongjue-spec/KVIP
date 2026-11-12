# 分核研究与KVIP主仓集成状态

本分支基于KVIP主仓main **13d2898**，同步独立预算核心、60场景脚本、实验源码与结果。使用干净派生分支，提交供MR审查，保持Draft、不合入。

## 已接入核心的内容

- 新流式链路使用独立C/V预算。每阶段分别检查LI当前mixed路径的AIC/AIV占用与AIV-only尾部的`3B+4W`，按实际设备查询验证请求上限；物理资源没有1:2任务配对约束。
- 四query publication records与LI工作核数解耦，补齐C28/BS16的F4工作区与发布布局。F4表示记录数，不能按四个额外驻留LI核理解。
- 两transport新增显式V预算的`kvip_lifetch_streaming_forward_v2`，保留旧六参符号。URMA按`2B+8W`检查实际queue views；超预算或不足会在启动前失败。
- 精确local/merge、history分段读取算法已整理为共享核心`csrc/operators/common/lifetch/split_topk.cpp`，并有独立CANN target。

**接入边界：**共享TopK模块是显式研究入口，尚未替换新main的Final classification/publication。当前生产入口保留新main条件化P2、每query完整选择T1；旧P32/T8 score-only原型没有被伪装成完整生产预取路径。今后接入T4/T8须保留locations、query bitmap/error发布和general fallback；恢复P32的resident阈值、Service及graph数据准备仍待实现。

## 用例与验证

[执行脚本与完整命令](../../../experiments/split_core/README.md)覆盖60个基础场景：BS1/2/4/8/16 × miss3/5/10/15% × sequence30/50/100%。small/medium/large分别24/24/12组。两档预算、G32/G64、算法和重复次数是额外维度。

生成器修改真实FP8特征，逐query精确miss为61/102/205/307；当前受控重叠U=2m。30%有效长度39322与存储39424分开，native与fused使用同一可见上界。CPU60组已验证，G32/G64另有24组独立真实FP8解码、BF16与padding回归。

当前核心新增的Model组件观测如下；均有正常footer、local/global oracle、输入未改及guard检查，单次、无预热：

| BS | 有效H / 存储H | miss/q | TopK分片 | V预算 | Model ticks |
|---:|---:|---:|---:|---:|---:|
| 1 | 39322 / 39424 | 61 | 8 | 56 | 22439 |
| 4 | 65536 / 65536 | 102 | 4 | 56 | 30120 |
| 16 | 39322 / 39424 | 307 | 4 | 56 | 49909 |

这些是AIV-only TopK组件，不含QK、完整LI、Service、真实transport或vLLM。没有相同shape的新基线，故不计算加速百分比。`results/current-model-matrix.json`保留58个NOT_RUN行；完整60场景Model扫点及当前流式链路的真机验收尚未执行。

CPU/构建验证包括当前预算与F4布局、graph启动期分配/错误传播、共享LI dispatch、实际FP8数据、适用的runtime/adapter/cache测试；两transport新host源码通过C++语法编译。共享TopK模块和历史两个mock driver均已从源码经CANN编译，不依赖旧.a。精确命令、源码绑定和最终测试数见`results/integration-validation.json`。

本分支完整CPU测试为205项通过、79项子测试通过。主仓原有测试保持完整；本分支新增当前streaming预算、F4布局与fixture功能回归。验证仅针对本分支源码，精确结果见`results/integration-validation.json`。

## 串行、融合与分核统一比较

[统一比较表、图与可比范围](pipeline-comparison.md)补齐旧六kernel串行、Fusion3与history分核：Fusion3同组对照1,340,113→1,226,673（−8.46%）；分核同binary对照1,340,108→693,586（P32/T8，−48.24%）。固定P32的T32→T8降低3.47%。两族driver不同，不能将跨driver差值归因于分核。新main streaming保留NOT_RUN，原始clock/performance限制标识不变。

## 历史实验与结论

[完整分析、计算模型、论文与创新假设](split_core_independent_analysis.md)、[Prompt](split_core_v2_prompt.md)、[15项参考](split_core_references.csv)、[结果图](split_core_v2_results.png)和[35次日志/摘要](results/model-records.json)已纳入。

这些结果固定在**81b1472＋Service0/GM-copy mock overlay**，不是本MR新流式main的性能：原链1340108 ticks；P32/T8为693586（−48.24%），P28/T8为698318（−47.89%）。与原链比较包含Q4及删除resident/classify/progress等组合变化。固定P的干净控制为T32→T8降低3.47%、T28→T8降低1.85%，支持生产P与选择T分别规划。

研究优先级是P/T联合规划、精确阈值的有界候选压缩、非均匀尾部、低桥接局部通路与range-ready/Service deadline联合计划。独立核池、队列、双缓冲已有先例，不作为首创。结构模型覆盖60场景，不等于60场景实测。

导出包保留日志与数据hash、独立审查范围及旧计时撤回标识；不提交driver、运行库、权重或大型原始tensor。离线verifier只核摘要/日志/表一致性，完整数值复核需要按复现入口重放。重建的历史标准输入20个共享文件均与原输入hash一致，binary和Model计数仍需重新建立同binary控制。
