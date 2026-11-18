# 分片 radix Top2048：B1 / B16 控制实验

本实验仅比较 **score 已完成后的数值 Top2048**。输入是同一份不可修改的 GM uint16 sortable key；Q4、H131072、K2048 固定。没有 QK、反量化、resident θ、Cold predicate、Service、Fetch 或 Index 更新。不能把结果解释成完整 LI 或 LIFetch 提速。

## 对照与实现

`original/P1` 调用 `post.cpp::li_qk_topk`，沿原 `QLIVector::ProcessTopK` / `LITopk<uint16_t>` 路径，每个真实 Q 一个 AIV。它将 GM score 分块搬入 UB，执行高/低 8 位累计直方图、精确阈值选择和 GT/EQ gather；长 H 时将上一轮2048候选与下一块继续选择。不是整个 batch 只用一个 AIV，也不是在 GM 内执行全排序。

`sharded` 用五个 AIV kernel：

1. **high**：每 Q/history 段读取 score，生成256桶高8位累计直方图。
2. **high_reduce**：每 Q 批量搬入所有分片直方图，在 UB 中相加，复用原生 `FindHighTargetBinVFImpl` 选择全局高位桶及桶内 rank。
3. **low**：分片读取 score，仅统计命中全局高位桶的低8位累计直方图。
4. **low_reduce**：每 Q 合并低位直方图，复用 `FindKthVFImpl` 得到唯一全局阈值。由各片累计计数直接计算 GT/EQ，再分配互不重叠的输出偏移和 EQ 配额。
5. **emit**：分片读取 score，以128个uint16 key为一组用原生 SIMD Compare/Squeeze压紧块内uint16索引；只将实际选中的索引扩展成真实uint32 token ID，再DMA发布。

累计直方图必须按累计计数处理：`GT_high = high[255] - high[high_bin]`，`GT_low = low[255] - low[low_bin]`，`EQ = low[low_bin] - low[low_bin-1]`。最后一项在 low_bin=0 时减0。原生 `ACCUMULATE` 不是单桶频数，不能直接按频数求阈值。

每个块最多16384 key，原生 uint16 histogram 的块内计数不会溢出；跨块/分片累计使用 uint32。DMA 的32B尾部显式补零，更长的256-key对齐尾部先清零；高位累计直方图所有桶均扣除padding，低位仅在全局高位桶为0时扣除。emit 的有效mask排除padding。

**精确性**：同一 Q 的所有分片共用一个全局 τ，全部 `score>τ` 必须输出，`score==τ` 按分片前缀配额补足2048。输出可无序；输入 token 区间互不重叠，不需要数值选择中的跨片去重。四Q共享 miss/dedup/slot/victim 的 owner 约束属于后续 Service，不能由此取消。

## B1 与 B16 的不同策略

| 配置 | 扫描任务 | 活跃 AIV 上限 | 每 AIV 最多段数 | 单段 history |
|---|---:|---:|---:|---:|
| B1/P1 | 4 | 4 | 1 | 128K |
| B1/P4 | 16 | 16 | 1 | 32K |
| B1/P16 | 64 | 64 | 1 | 8K |
| B1/P32 | 128 | 64 | 2 | 4K |
| B16/P1 | 64 | 64 | 1 | 128K |
| B16/P2 | 128 | 64 | 2 | 64K |
| B16/P4 | 256 | 64 | 4 | 32K |

B1 增加 P 会提供更多实际扫描并行，但增加直方图通信和归约。P32 已超过64个 AIV task槽，并不会比 P16 多一倍同时执行的 AIV。

B16/Q4 的原路径已提供64个 Q 任务。增加 P 不增加扫描 AIV 数；首版采用静态常驻 task 循环：`Q = task % (B*4)`、`part = task / (B*4)`，每 AIV 连续处理同一真实 Q 的各段。它不是原子任务抢占，也不为每段另发 kernel；每次调用固定五个 kernel。应优先检查新 P1 的算法收益，再判断 P2/P4 粒度是否值得。score 的30/32 AIC预算与这里的64 AIV尾部预算是独立参数。

新路径读取 GM score 三遍：B1 为约3MiB，B16为约48MiB；原路径每个 score块通常只从GM搬入UB一次，再在UB中完成本块选择。跨片高/低histogram总大小为 `2*B*4*P*256*4B`；没有 `P*2048` 个候选 score/token 的 GM gather/map 中转。更多计算并行仍可能被额外GM流量、五次launch及两个归约抵消。

## 验证与数据

CPU测试验证真实 host 的阈值/唯一性/GT完整性/guard检查，覆盖uniform、ties、zeros、skew、P30不整除边界；CPU launch stub 不构成硬件验证或性能数据。

设备 host 使用独立 `nth_element` 计算精确τ，检查所有Q的2048个有效唯一token、强制GT完整性、每片高/低累计直方图、偏移/配额、guard及输入不可修改。Model日志由实际 ELF entry + 每个物理AIV launch序列校验后统计，指令数是所有活跃AIV的issued总数，指令占比不是时间占比。

2026-11-18，CANN9.1.0 / Ascend950PR_9599 Model，32AIC/64AIV拓扑，数值TopK全部只使用AIV。以下为**单次完整Model执行**，每case warmup0/iteration1；全部九组正确性PASS。没有A5微秒结果，不能将ticks换算成真机时间。L2模拟配置cache_enable=0；模型内存/launch成本不能等价到A5。

| Batch | 路径/P | Model ticks | 相对原路径 | 全AIV issued指令数 | 指令数/原路径 |
|---|---|---:|---:|---:|---:|
| 1 | original/1 | 50,060 | 1.000 | 92,800 | 1.000 |
| 1 | sharded/1 | 99,117 | 1.980 | 95,307 | 1.027 |
| 1 | sharded/4 | 51,914 | 1.037 | 107,463 | 1.158 |
| 1 | sharded/16 | **41,829** | **0.836** | 170,103 | 1.833 |
| 1 | sharded/32 | 52,814 | 1.055 | 218,663 | 2.356 |
| 16 | original/1 | **49,534** | **1.000** | 1,484,800 | 1.000 |
| 16 | sharded/1 | 102,463 | 2.069 | 1,515,948 | 1.021 |
| 16 | sharded/2 | 104,518 | 2.110 | 1,536,940 | 1.035 |
| 16 | sharded/4 | 107,264 | 2.165 | 1,578,924 | 1.063 |

**结论：B1/P16在已测点中最好，ticks下降16.4%，但总指令增加83.3%。这是并行执行缩短关键路径，不是减少总工作量。B16的分片新算法全部劣化约2.1倍，首版不能作为B16性能候选。** 不同Batch的ticks不能直接解释为跨请求实机吞吐。

`candidate/new-sharded-P1`隔离同一种算法内的分片影响：B1/P4=0.524、P16=0.422、P32=0.533；B16/P2=1.020、P4=1.047。B1新P1已比原算法慢98%，不能只拿新P1作基线宣称整体优化。

### 指令与阶段窗口

issue包络是此kernel在全部活跃AIV中最早/最晚issued tick之差；不是硬件kernel持续时间，也不是耗时占比，各kernel包络之和不严格等于whole ticks。

| Kernel | B1/P16 指令数/占比 | B1/P16 issue包络 | B16/P2 指令数/占比 | B16/P2 issue包络 |
|---|---:|---:|---:|---:|
| high | 34,788 / 20.45% | 5,751 | 188,672 / 12.28% | 26,391 |
| high_reduce | 3,652 / 2.15% | 7,737 | 34,240 / 2.23% | 6,394 |
| low | 37,963 / 22.32% | 6,130 | 222,960 / 14.51% | 25,486 |
| low_reduce | 6,044 / 3.55% | 11,667 | 48,320 / 3.14% | 7,870 |
| emit | 87,656 / 51.53% | 6,958 | 1,042,748 / 67.85% | 34,634 |

B1/P16 两个归约包络合计19,404，超过单个并行扫描阶段。P32没有增加活跃AIV，增加任务循环和histogram归约，因而整体反而慢。B16/P2每AIV仍扫描合计128K；三次GM score读与分阶段串行连接不能由更多AIV隐藏。低issued占比的归约也可能占较大窗口，所以不能用“指令占比低”判定不存在关键路径瓶颈。

最终版本前的过滤实现用64个uint32比较一组；B1新P1总指令167,789。改为128个uint16一组、仅扩展选中索引后降为95,307，减少43.2%。这与分片本身的收益分开统计；旧版本未完成所有九组，不能作为完整候选表。

可核验的提交证据见[evidence/radix_model_20261118](evidence/radix_model_20261118/README.md)：`summary.csv`、`phases.csv`、`categories.csv`、`opcodes.csv`和`instruction-proof.json`，包含每个活跃物理AIV的entry/分母与raw日志hash。完整raw issued日志与fixture保存在本地结果目录。部分B16 CCU/popped辅助日志因磁盘压力丢弃，pruning记录明确标记；它们不参与issued计数。A5 npu构建链接通过，实际msprof需用户在A5运行。

## A5 仓库入口

在能访问 GitCode 的宿主机更新共享仓库后，在原 A5 容器中执行：

```bash
cd /home/w00568585/wt-mr14
bash experiments/li_qk/run_radix_a5.sh --device 5 --profile
```

脚本直接使用仓库文件，不下载bundle，不调用docker。可用 `--suite b1` 或 `--suite b16` 分开运行；`--plan-only` 仅打印计划。CANN/SOC与环境不同时用 `--cann-root` / `--soc`，KVAF头文件在其他位置时用 `--kvaf-include`。

默认warmup4/iterations20，丢弃r0，再做三轮交错测量。`results/summary.csv` 同时给出 candidate/original 与 candidate/new-P1，比值小于1表示该测量指标下降。前者比较原算法，后者隔离同一种新算法内的分片影响。msprof是独立profile运行，每case保存trace及导出结果；不能与bench event样本混用。

构建、oracle或采样失败会非零退出，终端打印失败阶段和日志尾部。结果目录保留source commit/status、构建与输入hash、全部轮次report、bench日志、Model指令分解或msprof trace。

## 接入生产路径的条件

本提交只增加独立实验入口，未改变MR14默认路由。只有数值TopK真机收益确认后，才考虑让score/Cold producer顺带维护高位直方图，消除第一遍GM读；低位统计仍依赖全局高位桶。resident θ 和Final τ必须独立生成，不能用局部阈值。最终classification仍需等待Service mapping及scratch生命周期安全；Fetch/CQ与Index commit的依赖另行保持。

参考：[RadiK，ICS2024](https://doi.org/10.1145/3650200.3656596)，[论文](https://arxiv.org/html/2501.14336v1)；[SC2023 Parallel Top-K](https://doi.org/10.1145/3581784.3607062)。分片histogram归约是已有算法思想，本实验贡献边界是Ascend独立UB、实际Q/history任务分配及LI尾部适配，尚未证明论文创新性。
