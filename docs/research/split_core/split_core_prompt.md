> 历史研究归档：测试基线为81b1472＋实验overlay，不代表本MR新流式main的性能。本地.aiinfra路径仅标识原始证据；可移交摘要见 results/model-records.json，当前集成状态见 README.md。

> 此为首轮历史 prompt。用户已授权仿真并纠正物理资源含义；以 [split_core_v2_prompt.md](split_core_v2_prompt.md) 的完整修订版为执行依据。旧版“默认不运行大型仿真”已被后续授权替代。

# KVIP AIC AIV 分核策略研究 Prompt

你是熟悉 AscendC、异构流水、长上下文稀疏推理和 KV 缓存传输的 AIInfra 技术负责人。请直接执行以下研究，交付中文分析、可复算的场景矩阵和可执行实验计划。

## 代码与工作流

1. 工作目录为 `${KVIP_ROOT}`。从 KVIP_agent 对应主仓 `https://gitcode.com/CacheOS/KVIP.git` 同步远端 main，固定实际 commit、同步时间及源码哈希。原 `${HISTORICAL_SOURCE}` 的未提交代码、任务和资源不作修改。
2. 读取并复用“复核 ComputeMm1 优化分析”session 的 AGENTS、aiinfra-task、aiinfra-review、aiinfra-benchmark、角色与验收约定。区分同步到当前目录的工作流和旧实验记录，不把旧结果作为当前 main 的性能证据。未建立人工接受的协作提交时仅进行咨询式只读并行，由 coordinator 单写研究产物；不伪称正式 freeze/gate/close。
3. 默认不修改业务实现，不提交、推送或运行大型仿真。无可用真机只交付源码事实、解析模型、CPU验证和待测方案；不得编造性能、设备规格、创新性或实验通过。

## 参数与解释

- BS（packed request batch）= 1、2、4、8、16。
- miss 率 = 3%、5%、10%、15%，原文字样 `%10/%15` 归一化为百分比。
- 最大 history = 128K = 131072 token；有效 history 比例 = 30%、50%、100%。优先解释为每请求有效 history 长度，非 resident 比例，非固定 batch 总和。说明整数和 tile 对齐方法，同时指出其他解释会如何改变结论。
- 覆盖全部 5×4×3=60 个基本场景；按 32 AIC/64 AIV 与 28 AIC/56 AIV 两种资源预算分别分析，共120行。用户“A50 cube 32/28 配对的2倍64/56”作为待核实资源假设，不当作已证实 SKU。检查预留核、可用核与实际 launch、batch 约束。
- Q/request、Indexer heads G、维度 D、TopK、resident budget、payload 字节、prefetch rounds、page/block size、网络能力等从当前代码提取；对未指定参数明确默认、敏感性和待补信息。至少讨论 G32/G64、splitQ0/1、miss 在 query 级与去重后 request 级的区别。

## 必须回答的问题

1. 源码走读全算子链路的 AIC/AIV 配额、launch grid、batch/query/head/history/tile 归属；区分逻辑 block 与物理同时驻留。比较公开 main 与既有三 kernel 实验方案，明确哪些结论通用、哪些属于实验分支。
2. 走读 LI 内的 bN2 → query组 → history tile、L1/L0/MMAD/FIX、双 AIV 的分工，说明 M/N/K、CV/VC、UB bank、阈值、全history TopK、hits/Progress、Service 的因果关系。不要把两 AIV 误解为分半 history，或把两个128误认为同一维度。
3. 当前分核重点解决什么问题，优化目标、决策变量、约束和瓶颈是什么？以 selected KV 消费就绪的关键路径为主目标，区分 LI 局部耗时、整算子、TPOT、吞吐、P99、传输等待及不确定性。
4. 建立异构资源约束 DAG / 流水 makespan 模型：计算、HBM与页表搬运、TopK归约、metadata、异步网络与completion、cache apply/一致性。拆开 query selected miss、request 去重 miss、speculative prefetch、有效提前量。不要将 miss 直接乘 history 当作必需 payload，不把提交等同完成。
5. 对60场景及两种核预算给出当前调度的核利用上限、tile与波次、打分/读K/写score/必要miss字节、尾部风险和建议。标识解析估算，不能把缺少测量的比例变成时延或最优分核数。
6. 分析 batch/query/history/head/tile 多层切分及资源分配的收益与代价：小batch如何展开history并行、中大batch如何保K复用和MMAD效率，splitQ复制流量、精确全局TopK归并、resident阈值早出、Service抢占和prefetch窗口的耦合。说明不适合照搬 attention softmax split-K 合并公式。
7. 对32/64与28/56预算提出以何种问题模型为核心的合理策略，给分batch的候选搜索空间和切换条件，而非固定核数拍板；核实当前host支持或拒绝的batch范围。
8. 检索并实际阅读高相关原始论文和官方文档。给出标题、链接、核心思想、可迁移点、差异与局限。至少覆盖 IO-aware/fused attention、decode序列并行、动态调度/负载均衡、KV缓存和预取。创新点必须标记为研究假设，提出可证伪实验和已有工作差异，不宣称未经检索验证的首创。
9. 按收益依据、实现成本、正确性风险给出优化优先级、最小实验、观测指标、失败条件、消融和回滚。实验覆盖60场景，报告实际miss整数、分布、query重叠、冷热倾斜、真实拓扑、source/build hash、预热/重复、正确性oracle和方差。

## 交付与验收

- `docs/research/split_core_analysis.md`：结论先行的完整研究，源码路径与行号、公式、决策表、论文链接、创新假设及实验路线。
- `docs/research/split_core_matrix.csv` 与生成脚本：120行可复算解析场景，包含输入口径、实际核数/约束、主要成本量和建议。
- `.aiinfra/tasks/SPLIT-CORE-001/`：任务、同步与证据说明、独立只读审查、真实命令和结果；形式状态与咨询研究交付状态分开。
- 独立复核源码映射、矩阵算术、模型假设和引用，并验证 main 业务源码未被修改。报告无法验证的硬件性能项；交付完整研究不依赖等待用户补充可合理假设的信息。
