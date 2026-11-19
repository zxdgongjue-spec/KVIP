# 小 batch 的下一步优化

执行需求见 [SMALL_BATCH_PROMPT.md](SMALL_BATCH_PROMPT.md)。优化前为0f846ec的window联合路线；MR23仍作外部控制。用户提供的B1/P16 r0中，联合LI相对原融合省约95–101µs，完整链省24.58%–42.30%，但ResolveUpdate没有降低。这是B1初步证据，不代表B2/B4或正式轮，也不代表超越MR14 direct优化串行。

## 可核验的硬件与论文依据

Ascend C官方[3510架构说明](https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_3510.html)给出AIC:AIV=1:2、每AIV UB 256KB、L1 512KB、L0A/B各64KB、L0C 256KB；实际可用UB受编译保留与SIMT DCache影响。当前代码按223232字节预算，不能将256KB全算成算法scratch。32AIC/64AIV是用户设备规格及本项目运行时预算，不是所有950 SKU的统一核数。各AIV的UB独立；跨核中间状态仍要明确搬运/发布依赖。

| 来源 | 平台/日期 | 可借鉴机制 | 本项目适配边界 |
| --- | --- | --- | --- |
| [FlashInfer](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html)，MLSys2025 | GPU | 按工作量规划、确定性队列、图兼容persistent调度 | B/P/W分开选择；不增加无必要的partial输出；CUDA CTA不是Ascend AIV |
| [FlashAttention-3](https://proceedings.neurips.cc/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html)，NeurIPS2024 | Hopper GPU | 矩阵与搬运异步、角色分工和流水 | 借鉴生产/消费分工；本库GM发布、事件和独立UB不可省略，不能照搬TMA/WGMMA |
| [Sample-Guided Exact Top-K](https://arxiv.org/html/2609.08450v1)，2026-09预印本 | GPU FP32 score | 提议边界→全行认证/候选形成→精确修正，按shape选择row-local/KV-split | 采样只能影响工作量，不能直接决定结果；本库residentθ已有安全筛选，BF16及resident语义需单独适配 |
| [RadiK](https://arxiv.org/abs/2501.14336)，ICS2024，2025上传预印本 | GPU radix TopK | 输入/批量规模、带宽与片上状态决定分配方式 | 不推出所有shape都应跨history多轮radix；旧P16分片radix的额外launch已在A5劣化 |
| [FlashOffloading](https://arxiv.org/html/2611.07542v1)，2026-11-05预印本 | Ascend910C | reverse sparse index；在CopySFA的KVgather中融合miss搬运，并与hit计算重叠 | 主收益边界在LI之后的SFA consumer；本轮不改SFA/vLLM，也不将预印本页内会议模板文字当成正式录用证明 |

这些机制支持按实际关键路径选择方案，不支持移植GPU加速比。资料核验日期2026-11-19。

## 本轮实现：每Q并行增量TopK，按request提交

旧window每窗口：

```text
分核score/predicate → 一个request lane循环Q0/Q1/Q2/Q3的增量TopK/rank
                   → 跨Q union/admission/URMA → 下一窗口
```

新 `window_query` 每窗口：

```text
分核score/predicate → 4B个AIV分别更新各Q的Top2048、rank和候选
                   → 原request owner跨Q union/admission/URMA → 下一窗口
```

原数学算法、唯一residentθ、canonical/bitmap、TopK prefix、rank policy和末段保持同路径。每Q只DMA回写自己的128字节metadata，禁止整request的512字节metadata回写造成其它Q覆盖。request Row、covered bitmap、队列和job提交仍由原post owner唯一修改。所有Q完成之后才进入post；predecessor事件继续检查，不提前等同URMA/CQ完成。Context/RuntimeShape ABI尺寸不变，serving replay门禁不变。

独立窗口kernel不执行score的ProcessVec1，因而不应调用其AllocEventID/FreeEventID预置score bank额度；它与窗口本地配对事件6/7发生冲突。Model检出了duplicate SetFlag，本轮已移除独立窗口的这组初始化/回收。修复适用于原window和新query两条路线，未改MR23固定提交及MIX score kernel的事件银行。仿真串行控制复用原AdvanceWindow计算主体和修正后的独立窗口初始化；这不等同0f846ec二进制逐指令不变。

| B | score P | score AIC/AIV | 旧有效TopK AIV（W4） | 新TopK AIV | 旧/新LI launches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 16/32 | 1（实际W2） | 4 | 6/8 |
| 2 | 16 | 32/64 | 2 | 8 | 10/14 |
| 4 | 8 | 32/64 | 4 | 16 | 10/14 |
| 8 | 4 | 32/64 | 8 | 32 | 10/14 |
| 16 | 2 | 32/64 | 16 | 64 | 10/14 |

新方案每窗口新增一个kernel和一次完整Q汇合，不能先宣布更快。旧lane有next-Q DMA预载，新方案改为各Q独立输入流水；原prefix的GM读写量不消失。仅metadata/progress/Row header的DMA，新方案约1792B/request/window，旧方案约1600B，增加约192B；未计额外初始化、标量发布、cache维护或输入流水差异。省下的四Q串行时间必须覆盖这些代价。

例如B2时，原lane0处理request0的Q0→Q1→Q2→Q3，lane1处理request1的四Q。新方案8个AIV分别拥有一个Q的12KiB TopK prefix（4KiB score+8KiB ID）；随后两条request lane分别读取各自四Q的ranked候选，按原规则去重/准入/提交。各Q已经读取同一窗口的完整有效history，不需要对history分片TopK再归并。完整prefix仍有每Q约12KiB读取和12KiB回写，并行并不消除这部分流量。

可用下式理解收益门槛（仅分析式，不是仿真测量）：

```text
每窗口净省时 ≈ 原request四Q串行TopK时间 − 新并行Q的最长TopK时间
             − 新增launch/gap − 新增初始化及搬运 − 预取重叠变化
完整链净省时还须加上Resolve/Get、Final和其它边界的实测变化。
```

因此B2/B4的任务不足更值得先测该候选；B8/B16的原request并行度较高，新增64路prefix访问可能受带宽限制。若额外launch抵消收益，可探索一个常驻AIV kernel内的Q完成发布与唯一request owner提交，但必须先证明所有Q同时驻留、GM可见性和不会等待未调度block。本轮使用同stream kernel边界保证Q汇合，未实现这个单kernel变体。

## 其它方向及优先级

1. **窗口和分核预算分别调优。** B2/P16的Hot每片仅1024 keys，B4/P8为2048；P_hot不必等于P_cold。先用现有parts-map测B2/P8对P16、B4/P4对P8，并比较W2/W4、pf0/1024；之后再考虑独立P_hot接口。pf0关闭预算，但不会自动消除原Service/增量TopK固定成本。此方向已有实验参数，未宣称完成自动最优路由。
2. **候选形成和精确选择减少重复扫描。** 可借鉴2026采样认证论文，对当前行构造粗边界，并在producer或一次Cold扫描中认证/压缩；必须保留不足候选回退、ties与BF16语义。当前residentθ是安全筛选的一部分，不能用未经认证的预测阈值直接替代。该方向尚未实现，不沿用旧五阶段sharded radix。
3. **让Fetch与真实SFA consumer重叠。** 小batch LI很短，继续加窗口不一定提供足够覆盖时间。FlashOffloading借用后续hit attention计算覆盖搬运并减少一次HBM读取，可能比仅在LI期间预取更有空间。需要适配SFA的hit/miss依赖、输出slots及Index提交，属于后续端到端任务，本轮不伪称已落地。

本库score的S2 tile为128 keys，窗口分片按256 keys页对齐。256是合法分片粒度，不是已测出的最高效history长度；实际最优值需要含launch、DMA和后处理的真机扫参。B2/B4先测P减少一半，可以判断Hot的8/16个tile是否仍足以摊平初始化；Cold长度更大，不应由Hot结论直接决定其P。miss率通常只能从近期步骤估计，允许它改变预取预算/窗口，却必须让最终精确选择和缺失Fetch完成负责正确性，不能因预测低miss而省掉最终搬运。

## 仓库脚本

在宿主机更新 `codex/lifetch-score-t2`，容器从共享目录执行：

```bash
cd /home/w00568585/wt-lifetch-score-t2
bash experiments/lifetch_score_t2/run_small_batch_a5.sh \
  --device 5 --kvaf-root /home/w00568585/KVAF-score-t2-1c131038 \
  --kvaf-server 127.0.0.1:28885 --batches 2,4 \
  --miss-percent 0,3,5,10,15,20 --prefetch 1024 --intermediate-fetches 4 \
  --rounds 3 --warmup 4 --iterations 20
```

默认三条融合路线为MR23、旧window_budget、新query_budget；每条仍配对原SDK串行参考。扩展可传 `--batches 1,2,4,8,16`；预取控制传 `--prefetch 0,1024`，窗口控制另跑 `--intermediate-fetches 2`。summary新增 `query_topk_us`、`over_window_budget`、`net_gain_vs_window_us` 和配对轮次；不把post阶段全部解释为URMA传输时间。

有界真实AscendC对照：

```bash
bash experiments/lifetch_score_t2/run_query_topk_model.sh \
  --mode sim --device 0 --kvaf-root /home/w00568585/KVAF-score-t2-1c131038
```

也可 `--mode npu --device 5` 检查该局部工作。仿真fixture为H8192/R2048/Q4/W4/epoch1，比较实际原AdvanceWindow和新AdvanceQueryWindow；它不执行score、residentθ、post、URMA、Final或Resolve。数据不同Q/owner、精确TopK值/ID、候选合法性/hash、cutoff/cursor、metadata及输入/guard检查。timed work在InitBuffers之后、pipe.Destroy之前，`work_span_cycles`为工作区间最晚结束减最早开始；不是完整kernel延迟、全链收益或A5微秒。Model实际SoC与目标不同必须单独记录。当前fixture仅覆盖covered为空、一个增量epoch；非空covered、多epoch、真实URMA及末段正确性仍需A5全链PASS。

本轮实际仿真与编译记录见 `query_validation.json`，只有PASS的同尺寸结果用于比较。未完成/系统终止的运行不计算收益。完整A5三路收益待真机验证。

### 本轮实际局部结果

| batch | 原request四Q循环：cycles | 每Q并行：cycles | 新/原 | 局部工作区间减少 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 41301 | 14628 | 0.354 | 64.58% |
| 4 | 42352 | 14762 | 0.349 | 65.14% |

四个正式局部运行均通过精确TopK、候选、cursor/metadata、输入/共享状态/guard及ACL清理检查，进程退出0。每batch两路的候选数/hash完全一致，B2为512、B4为1024。原始完整日志、摘要和summary见 [evidence/query_small_batch](evidence/query_small_batch/summary.csv)；失败与未完成运行的记录另列于query_validation.json，不混入本表。

最终CANN 9.1/Ascend950PR_9599完整NPU bridge编译链接及新TopK/post/forward符号检查通过；源码快照摘要与实现提交一致。23项实验检查、3项构建恢复检查通过。Model上报SoC为Ascend950PR_9589，与编译目标分开记录；不可据此替代A5实测。

这个结果支持把按Q并行作为B2/B4的下一步候选；未证明完整LI/Fetch减少65%，也未证明它优于MR14优化串行。新增kernel/post初始化、完整H128K的输入规模、非空covered及预取重叠变化仍须通过A5三路实测决定。

CANN 9.1纯AIV Model在数值检查及显式ACL清理之后的进程析构存在重复释放。`RUN_MODE=sim` 的局部host程序会在释放所有buffer、销毁stream、reset device并成功aclFinalize后记录标记，再用`_Exit(0)`跳过这次重复atexit析构。NPU程序正常退出，不启用该处理；有kernel事件错误、ACL清理失败或非零退出的结果仍禁止汇总。此处理只涉及计时之后的Model进程退出。

该fixture只更新首个Cold窗口：以已准备好的resident Top2048为前缀，加入2048个新score，扫描至cap4096，两个MR23 cursor分别为3072/4096。并未计算H8192的最终全history TopK。每route仅运行一次，不提供p50/p99统计；H131072/R16384的A5场景仍由上面的完整链脚本验证。
