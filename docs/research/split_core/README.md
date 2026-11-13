# 分核研究与KVIP主仓集成状态

本分支基于KVIP主仓main **13d2898**，同步独立预算核心、60场景脚本、实验源码与结果。使用干净派生分支，提交供MR审查，保持Draft、不合入。

## 已接入核心的内容

- 新流式链路使用独立C/V预算。每阶段分别检查LI当前mixed路径的AIC/AIV占用与AIV-only尾部的`3B+4W`，按实际设备查询验证请求上限；物理资源没有1:2任务配对约束。
- 四query publication records与LI工作核数解耦，补齐C28/BS16的F4工作区与发布布局。F4表示记录数，不能按四个额外驻留LI核理解。
- 两transport新增显式V预算的`kvip_lifetch_streaming_forward_v2`，保留旧六参符号。URMA按`2B+8W`检查实际queue views；超预算或不足会在启动前失败。
- 精确local/merge、history分段读取算法已整理为共享核心`csrc/operators/common/lifetch/split_topk.cpp`，并有独立CANN target。

**接入边界：**共享TopK模块是显式研究入口，尚未替换新main的Final classification/publication。当前生产入口保留新main条件化P2、每query完整选择T1；旧P32/T8 score-only原型没有被伪装成完整生产预取路径。今后接入T4/T8须保留locations、query bitmap/error发布和general fallback；恢复P32的resident阈值、Service及graph数据准备仍待实现。

## 当前42组用例（2026-11-13更新）

固定**sequence128K（131072）、Q4**；BS=1/2/4/8/16/24/32 × miss目标3/5/10/15/20/50%，共42组。small/medium/large为12/12/18组。冒烟与客户入口均遍历完整42组，不自动删减。执行方式见[仿真脚本与命令](../../../experiments/split_core/README.md)，清单见[cases.json](../../../scripts/split_core_sim/cases.json)。

`scripts/generate_split_core_sim.py`生成`smoke.sh`、`customer.sh`与可选`compare.sh`；默认T1/V56/G32，策略/预算对照为42×3×2=252条件。TopK host上限扩到BS32并提供capability检查，BS24/32不代表当前native完整streaming支持。

名义miss3/5/10/15/20/50%对应每query整数61/102/205/307/410/1024。20%实际20.01953125%，50%准确50%；当前受控重叠U=2m，50%时U2048触及LRU容量边界。新矩阵42组CPU已通过，CANN新driver已编译并核对BS32/Q4能力；完整42组Model及新增BS24/32的kernel执行尚未运行。见`results/customer-v2-validation.json`与42行NOT_RUN的`customer-v2-model-plan.json`；旧结果不迁移为新矩阵通过。

## 旧矩阵验证归档

以下组件结果及`results/current-model-matrix.json`、`results/cpu-matrix.json`来自先前60组矩阵（BS≤16、history30/50/100%、miss≤15%），保留原版本与状态。它们不是新42组的完成记录。

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

## 全通路分核审计与trace

[源码审计、前后差异与依赖图](end_to_end_schedule.md)、[交互式trace对比](scheduling-traces.html)、[当前TopK msprof PIPE图](topk-msprof-pipes.png)和[历史serial6/Fusion3包络](historical-fusion3-native-envelope.png)已补充。当前生产是条件P2/T1＋固定3B角色与按batch条带Fetch；P32/T8尚未整链接入，也没有按miss字节动态均衡。新抓2轮TopK Model得到3份官方trace，选中core0 V0/V1；完整当前streaming硬件trace仍未采集。见`results/scheduling-trace-evidence.json`及capture recipe。
