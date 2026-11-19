# 执行任务：小 batch LI/Fetch 的进一步优化与仿真

以当前 `0f846ec` 的 MR14 history/direct/predicate + MR23 T2 窗口联合实现为优化前基线，重点研究 B2、B4，扩展 B1、B8、B16 控制。先检索截至2026-11-19可核验的论文原文及Ascend 950/3510公开文档，记录日期、平台和公开规格，区分论文方案、可借鉴机制及本项目推断，不直接搬用GPU性能数字或假设AIV间共享UB。

固定Q4、head32、H131072、R16384、Top2048、seed43、miss0/3/5/10/15/20%、固定KVAF版本和同图独立状态计时。默认history P为16/16/8/4/2，B×P不超过32 AIC，64 AIV；P、窗口数、Service/TopK并行度和Fetch宽度分别规划，不自动宣称用满核即最优。

1. 梳理当前critical path，尤其每request的四Q在一个Service lane串行进行增量TopK与admission，而score已分核。提出至少三条方向：按Q并行后处理、窗口/P/预取预算按规模调优、面向真实consumer的搬运融合。说明每条方案的收益来源、GM/UB成本、同步约束及与当前接口的适配范围。
2. 优先实现可检验的按Q并行候选：每真实Q一个AIV更新现有增量Top2048及ranked候选，随后按request执行原跨Q去重/Service/URMA提交。保持canonical、bitmap、唯一residentθ、原TopK数学语义、predecessor fence、mapping、safe/conflict Fetch、CQ join、Output及Index commit。各Q不得覆盖同一request的完整metadata行；不得把局部完成误发布为全局连续ready，不添加history分片radix merge。
3. 保留原窗口候选与MR23。新候选必须显式选择，默认生产入口不变；记录新增kernel边界，并证明并行节省能否覆盖新增启动/GM成本。提供仓库中的A5全链对比脚本，B2/4为默认重点，允许B1/8/16与W2/W4、pf0/pf1024控制。
4. 使用真实AscendC函数构建一个有界、同尺寸、同输入的CANN Model对照；重点验证B2/B4每Q的TopK、ID恢复、阈值、候选rank、metadata、guards与共享状态隔离。仿真不具备URMA时只验证对应局部阶段，记录Model SoC与配置、源码/二进制摘要、阶段计数/周期或whole-run ticks，并明确各指标范围。不得将whole-run ticks称为A5 kernel延迟，不将局部结果伪装成完整LI/Fetch加速。
5. 若不启用海量指令日志，使用可复核的Model summary和静态任务/搬运计数；若需要指令占比，只对有界trace计数并保留原始证据。先检查磁盘，保留已有证据；不生成无依据的收益数字。
6. 运行覆盖并行ownership、原算法数学结果、CLI/核预算与同图所有kernel归属的必要检查。完整A5收益只在所有correctness PASS后汇总；r0丢弃，保留同图stage/trace、配对轮次及串行漂移。
7. 交付研究与实现分析、实际仿真结果及局限、可直接运行的仓库脚本、提交和宿主机更新命令。若某候选劣化，保留对照说明原因，不宣称论文机制必然适合小batch。当前已有B1 r0 PASS数据为用户提供的初步证据，不能当成B2/B4实测数据，也不能替代同协议MR14 direct优化串行控制。
