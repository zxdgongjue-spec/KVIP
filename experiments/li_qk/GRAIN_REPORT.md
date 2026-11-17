# G32 / B16 / 30 核：执行粒度与均衡分配的实测拆解

日期：2026-11-17。范围：独立数字 LI 控制实验，不修改生产 LIFetch 默认通路。

## 结论

本模板的 history 计算 tile 是 **128 token**，原来的 **2048 token 是任务分配量子**，不是 AscendC 规定的最小高效任务。分配后，同 owner/half 的连续块合并为较长片段；一个片段内部连续循环执行多个 128-token tile。

固定 B16/Q4/G32/D128/H8192，使用 30 AIC + 60 AIV、一次 balanced score launch：分配量子从 2048 改为 512，全核指令 issue 窗口包络缩短 **14.12%**；改为 128，缩短 **15.60%**。总初始化次数与总指令数增加，收益来自最长核工作量减少。128 相对 512 只再缩短约 1.7%，不能据此确定 A5 最优参数。

对于实际 H128K，相同静态任务计划中，2048→128 只能把最长核工作量从 H71680 降到 H70016，约 **2.32%**。该粒度微调本身没有大幅整链加速的依据。优先收益点仍是：消除 30+2 两波尾部、减少片段重新初始化、producer 直写最终 score 布局。

## 三种粒度不要混淆

| 层级 | 当前取值/行为 | 含义 |
|---|---|---|
| 计算 tile | H128，Q4×G32 为 M128 | 一次 QK、ReLU、加权 head 归约的内部工作块 |
| 分配量子 `work_grain` | 128 / 512 / 2048 等 | 把总 history 工作均分到 30 核时的取整单位 |
| 连续执行片段 | 同 owner/half 的连续区间 | 一次 `RunScoreRow` 初始化后循环执行的长度 |

源码依据：`indexer/indexer.h` 的 `M_BASE_SIZE=128`、`S1_BASE_SIZE=4`、`S2_BASE_SIZE=128`，以及 `ProcessMain` 的 `s2LoopIdx` 循环；`indexer_score.h` 的 head 归约沿 G32 累加。它们证明当前模板的循环粒度，不能推广为全部 AscendC 算子的硬件下限。

本轮允许 Model 的 **score-only** 入口测试每 owner H256，分成 H128 half。完整 LI 仍要求 H≥2048，满足 TopK2048 的当前实验约束。没有给生产接口增加短 history 许可。

## 实验与证据边界

- CANN 9.1.0，SoC `Ascend950PR_9599`，950 Model；模拟拓扑为 32 AIC / 64 AIV，实际 score 使用 30 / 60。
- 七次 capture 均为独立 `T2_balanced_score`，一次 score launch，无 scatter、Final TopK、Fetch、Service 或 Index。
- 四个短 history 探针：H256/512/1024/2048，grain128。
- 固定工作量对照：H8192，grain2048/512/128；同一编译 binary、同一 fixture，输入 SHA 一致。
- 七次全部 `MODEL_PASSED`，exit0；score oracle、guard、输入只读核验通过。日志 mask `0xffffffff`，覆盖全部 30 个活跃 AIC 和 60 个活跃 AIV。
- 原始 Model issue 日志经过 captured ELF / symbol / runtime entry PC 校验；汇总保留少量 unmapped epilogue，未把它归入算法子过程。
- 每点一次探索采样，未重复测量。Model issue tick **不能换算为 A5 微秒**；issue 窗口含等待空档，既不是活跃周期，也不是最后一条指令的完成时间。

原始证据保存在 `tests/results/li-granularity-20261117/`；最终表为 `summary-final/cases.csv` 与 `cores.csv`。本报告不把旧 H128K score 的超时结果补写为成功，也没有真实 A5 msprof trace。

## 固定 H8192：均衡改善最长核，增加少量总工作

| 分配量子 | 每核最少/最多 H | 有效片段/Init 总数 | 最长 AIC issue span | 最长 AIV issue span | 全核 issue 包络 | 相对 2048 |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 4096 / 6144 | 46 | 40161 | 40589 | 40759 | 1.0000 |
| 512 | 4096 / 4608 | 58 | 34347 | 34875 | 35002 | 0.8588 |
| 128 | 4352 / 4480 | 60 | 33588 | 34230 | 34402 | 0.8440 |

全核包络定义为所有已记录活跃核的 `max(last_issue_tick) − min(first_issue_tick)`。各核 span 定义为该核最后和第一次 issue 的 tick 差。

| 分配量子 | 全部 AIC issued 指令 | 全部 AIV issued 指令 | whole Model ticks |
|---:|---:|---:|---:|
| 2048 | 413982 | 1560452 | 41808 |
| 512 | 420582 | 1580980 | 36025 |
| 128 | 422116 | 1584840 | 35422 |

这不是删指令带来的加速。细粒度切分增加了 owner/half 片段边界和初始化工作，但减少了最忙核的负担。执行窗口与总指令数必须分别看；AIC 与 AIV 的分母也不能相加后解读为耗时占比。

## 短片段：128 可执行，但初始化成本很重

以下不同总 H 的试验仅用于观察短任务特征，不能当固定工作量速度比。

| 每 owner 总 H | 最大核 H 工作量 | 最大 AIC span | 最大 AIV span | 全核 issue 包络 |
|---:|---:|---:|---:|---:|
| 256 | 256 | 10906 | 11577 | 11724 |
| 512 | 384 | 12067 | 12662 | 12775 |
| 1024 | 640 | 13597 | 13994 | 14216 |
| 2048 | 1152 | 17092 | 17530 | 17638 |

选取每次只有一个有效片段、一次初始化的 core0，观察归一后的执行窗口：

| core0 连续 H | AIC span | AIV0 span | AIV0 span / 每128 token | AIV0 issued 指令 |
|---:|---:|---:|---:|---:|
| 128 | 6902 | 7552 | 7552.0 | 1747 |
| 256 | 8081 | 8463 | 4231.5 | 2465 |
| 512 | 9353 | 9856 | 2464.0 | 3901 |
| 1024 | 11678 | 12197 | 1524.6 | 6773 |
| 4096 | 26347 | 26888 | 840.3 | 24099 |

前四行来自短 H 探针，末行来自固定 H8192/grain2048。Model 全局竞争负载不同，所以这张表不能确定精确的硬件拐点；但短片段摊销明显较差。可以从 **H4K 连续片段**开始进一步比较 H8K/H16K，不能把 H4K 宣称为已经证实的最高效最小长度。

## H128K：静态均衡收益上限较小

| 分配量子 | 每核最少/最多 H | 有效片段总数 |
|---:|---:|---:|
| 2048 | 69632 / 71680 | 60 |
| 512 | 69632 / 70144 | 60 |
| 128 | 69888 / 70016 | 60 |

这里仅计算任务覆盖，不是新一轮完整 H128K 设备仿真。由于每核已有约 70K 工作，2048 取整误差占比很小。512 是合理的候选起点；最终保留 2048/512/128 三点在 A5 上同场比较。

## canonical 与 Final 分别是什么

`canonical[owner, q, history]` 是 **最终分数数组布局**。每个元素已经完成 D128 点积、ReLU、G32 加权归约和 sortable key 转换。当前 score 先写虚拟 task-row scratch，scatter 再搬到 canonical；scatter 不做第二次打分。

“直写 canonical”是把输出寻址并入已有 score 写出，使不同 producer 写互不重叠的真实 owner/Q/history 区间。它不是融合额外 QK，也不是把输入虚拟 row 改成真实 owner。输入地址、θ 地址和真实输出地址必须独立表达；直接关闭 `scoreOnly` 会引入其它流程。

本实验的 Final 是每个真实 Q 对完整有效 H 做 TopK2048。不同 Q 选中同一 token 是合法结果；跨 Q 的 Fetch 去重属于 LIFetch 后处理。沿 history 分核不切 D/G 归约维度，当前也不做局部 TopK，所以 Final 没有额外的局部候选 merge。

直写 canonical **本轮尚未实现**。在 B16/H128K 下，完整有效 score 为 16 MiB；可消除至少 16 MiB scratch 读取 + 16 MiB canonical 写入，以及普通 T2 的两次 scatter launch。收益微秒数需要 A5 测量，不能由字节数直接推导。

## 计算与规约怎样一起执行

当前一个 MIX1:2 核组已经是流水：AIC 算 QK tile，两个独立 UB 的 AIV 各处理两个 Q 的 ReLU/加权 head 归约，Fixpipe 与 CV/VC、双缓冲协调生产消费。不是把 30 个同质核拆为 15 个计算核、15 个规约核。

动态规划可分成三层：

1. host 根据 B/Q/H 与核预算生成每核片段表，一次 score kernel 启动后各核独立循环。当前是确定性的静态均衡，没有设备原子抢任务。
2. 同 shape 片段尽量复用每核的 QLI/TPipe 初始化，只更新 owner/input/output/run 信息。当前 `RunScoreRow` 每片段重新 Init/Destroy，仍须保留 CV/VC drain；常驻 kernel 不等于已复用 QLI 上下文。
3. 完整 owner/Q score 就绪后，使用独立 AIV 预算执行 Final；若要与其它 owner 的 score 重叠，需要 ready/consumed、独立 scratch 和资源规划，并验证 CANN/A5 的真实并发调度。不能假设两个 stream 必然重叠。

30 核计算阶段结束后，同一批 AIV 可以进入 Final 阶段。也可探索让预留核上的 AIV 提前处理已完成 owner，但只有两组预留核意味着四个 AIV，不能当作 64 个 Final AIV 同时可用。是否优于全核阶段切换要实测。

## 进一步优化的优先级

| 改动 | 解决的问题 | 当前证据/状态 |
|---|---|---|
| 30+2 两波改为一次均衡循环 | 第二波仅2/30核忙，关键路径重新串行 | 已实现；本轮验证了不同 grain 的均衡 score |
| 调整分配量子 | 最长核多承担取整工作 | H8K Model 有收益；H128K 静态改善仅约2.3% |
| 每核复用初始化，剪掉纯 score 无用资源 | 短片段与 owner 切换固定成本 | 源码发现，尚未实现/测量收益 |
| producer 直写 canonical | scratch 中转与 scatter launch/读写 | 待实现，先独立验证 |
| per-owner Final 提前执行 | 完成较早的 owner 被全局等待挡住 | 需依赖、资源与 A5 并发证据 |

无用资源的具体源码依据：`QLIVector::InitBuffers` 在 scoreOnly 通路仍分配 TopK 的 mrgValue、indices、scoreOut、sharedTmp，并初始化 TopK buffer 绑定；`AllocEventID/FreeEventID` 仍设置/等待 TopK 事件，prefix/classify buffer 也保留。应给纯 score 明确的资源生命周期，逐项排除隐藏依赖，不能直接删 fence。

理想纯计算模型下，普通 T2/30核两波的最迟完成工作量约 H64K+H64K；均衡后约 H70K。忽略带宽、初始化和后处理时，有较大潜在改善空间。但 score 提速不能直接宣称 LI/Fetch 整链等比例提速；必须同时看 score p50、数字 LI p50 和实际 LIFetch ConsumerReady。

## 复现与 A5 后续

完整七次 Model 脚本为 `run_granularity_model.sh`：

```bash
bash experiments/li_qk/run_granularity_model.sh \
  /work/model-build /work/grain-results-new \
  /usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_3510/lib \
  /usr/local/Ascend/cann-9.1.0/aarch64-linux/ccec_compiler/bin/llvm-objdump
```

它要求新输出目录，自动抓全核日志、核验 PC、产生 cases/cores 表。短 H 只用于 Model score；A5 用固定 H128K 完整 LI。

宿主机更新后，容器可分别运行：

```bash
bash /home/w00568585/li_qk_shared/run_in_container.sh --suite batch16 --work-grain 2048 --profile
bash /home/w00568585/li_qk_shared/run_in_container.sh --suite batch16 --work-grain 512 --profile
bash /home/w00568585/li_qk_shared/run_in_container.sh --suite batch16 --work-grain 128 --profile
```

每次包括 B16 T1、T2普通waves、T2balanced；仅 balanced 使用指定 grain，score 固定30/60，Final独立64 AIV。三次各建新目录；r0丢弃，r1/r2/r3计入性能。每次的 `results/summary.csv` 都按同轮 B16 T1 归一，msprof另采，不混入事件计时。

需要更严格地复用同一 binary 时，先构建一次，使用 README 中 `run.py run --build SAME_BUILD` 的循环。物理 A5 trace 仍需用户在服务器执行 `--profile` 后获取。

## 验证记录

- 七项 CPU 控制检查、七项 launcher 编排检查通过，含 C++/Python粒度计划一致、唯一覆盖、短H入口限制、非法grain拒绝、参数转交与结果篡改拒绝。
- 当前 C++/header 与所有共享 device source SHA 均匹配 Model 和全新 NPU 构建 receipt；binary/library SHA也核验一致。
- NPU构建的所有wrapper SHA与最终代码一致。Model构建记录中的部分Python wrapper在后续增补统计/参数检查后变化；compiled C++/headers未变化，每个capture保留实际argv、geometry、Model配置与fixture SHA。
- 本地没有A5设备性能或真实msprof。七次Model已完整PASS，不能替代完整H128K/A5性能验收。
