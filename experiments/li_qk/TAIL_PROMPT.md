# LI history 分核后处理：需求与可执行实验 prompt

在当前 MR14 与独立 LI 控制实验上，分析并实现 score 后续流程的持续分核方案。先核验实际源码，再以相同输入、核预算和正确性标准进行设备仿真，不把理论工作量或静态指令数当作实测收益。

## 目标与基线

固定目标几何 B1/B16、每 request Q4、H131072、G32、D128、resident16384、TopK2048，A5预算32 AIC/64 AIV；B16 score预算沿用30/60。核验最新 origin/main、MR14 legacy与MR14 staged三个通路，记录commit与实际kernel序列。独立数字LI与含Service/Fetch/Index的生产LIFetch分别描述，不能互相冒充对照。

1. 明确反量化、ReLU、G32加权归约是在QK tile生成时还是最终score GM落盘后执行；列出输入输出shape、dtype、总字节和每核tile。区分沿D/G的数值归约与沿history的TopK集合合并。
2. 列出resident TopK/唯一θ、2048个LRU候选的Hot predicate、Cold predicate、Final TopK、Final classification、四Q保护并集的shape、数据规模、真实寻址和就绪依赖。不得将LRU2048候选分类错误描述成每次完整R扫描。
3. 说明哪些步骤可由原history producer/AIV继续执行，哪些必须等完整owner/Q覆盖或候选合并；保留唯一θ、四QHot完成、唯一Service/slot/victim owner、映射稳定及后续Fetch/commit依赖。

## 候选实现与控制变量

建立独立、显式的post-score实验入口，输入为同一canonical sortable uint16 score；不执行QK/Fetch/Service/Index，不重复反量化。阶段边界保留，实验不能声称producer/consumer已重叠。

- C0：按owner扫描Cold，resident/Final沿现有全区间TopK执行。
- C1：Cold按history分片、复用核组两AIV的Q归属；resident与Final仍相同。
- C2：Cold保持C0；Final沿history执行局部TopK2048，再对每Q的P×2048候选做相同TopK，并映射回真实token。
- C3：C1＋C2，测组合效果。
- 另提供独立resident_parts控制；局部resident结果必须精确合并后才生成θ和Hot predicate，不使用局部θ。先保持resident_parts=1作为四组共享控制；实现允许时补测2，验证收益或回退。

局部TopK使用现有库上uint16 TopK primitive；history片段读取canonical时保留真实Q行stride，局部索引与最终token显式映射。用候选集合并集包含全局TopK的性质证明正确性；保留tie合法性、queryEnds、无效候选与guard。P×2048随P增长，不能假设分核越多越快。

Final分类复用生产分类primitive或明确标出缩小状态范围；固定只读mapping/LRU/hot-valid状态，保留每Q locations/error/protection bitmap和四Q并集。此模型仅验证分类数值，不能证明动态Service映射、URMA或Index提交正确。

## 仿真与测量

在专用CANN950 Model容器中fresh build，记录SoC、32/64拓扑、日志mask、fixture/source/binary SHA、任务计划、命令、退出码和正确性。优先B1/H128K/R16K；先以H32K/R8K等缩小几何验证，再补B16。缩小几何必须单独报告，不能将旧H128K超时标成成功。

执行C0/C1/C2/C3相同配置；单独抓Cold和TopK控制以解释关键路径，再执行真实post-score链确认组合收益。固定同binary/fixture，所有活跃核开启动态issue日志；用实际kernel entry PC与ELF核验归属，按kernel/AIV/指令类统计总量、最忙核issue窗口及整个Model运行ticks。指令占比不是时间占比，issue窗口含等待，Model tick不能换算A5微秒。

对真实输出进行逐元素score只读、residentθ、Hot/Cold bitmap、TopK唯一合法/严格高于Kth必选/tie、真实token映射、Final locations/bitmap/error与四Q并集核验。失败或回退如实保存，不以正收益为通过条件。

## 论文与方案依据

查询并引用可核验的主流高性能实现：FlashAttention-2的sequence并行与规约代价、Stream-K的工作量分配，以及TopK/radix选择的分片候选或直方图规约。逐项解释哪些原理适用于本算子的sortable key和K2048，哪些CUDA/SM假设不能直接迁移到Ascend的异构AIC/AIV、独立UB与CV/VC。

形成生产接入路线：tile内继续反量化/归约与predicate；resident完整覆盖后一次θ；Cold段级ready；Final独立局部选择/精确合并；classification等mapping稳定；唯一owner后处理与safe/conflict Fetch/CQ/旧映射消费后commit。对尚未实现部分明确标注，不宣称整链提速。

## 交付与验收

交付prompt、源码、实验与A5/msprof脚本、阶段shape/字节表、论文来源、仿真结果与局限、下载包。旧七点/八点/B16入口保持兼容。按既有授权同步MR14，先检查并保留远端并发改动，禁止force push。仅A5真机数据可宣称A5实际加速。
