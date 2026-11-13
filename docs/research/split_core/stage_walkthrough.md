# 从Final Top2048到输出：处理规模与T/M/W/V

Final处理完整history的**score**，Classification检查选中的K个token，Resolve把未满足位置去重并分槽，Index exchange维护Key/scale与映射，Output生成全部K个输出及新增Fetch描述符。它们不是都重复处理H条128维向量，也不是都只处理miss。

## 符号与来源

| 符号 | 本文含义 | 来源 |
|---|---|---|
| B / BS | batch内request数 | 用例配置 |
| Q | 每request的query数，标准为4 | 用例配置 |
| H | 每query有效history长度，标准131072 | 输入长度 |
| R | resident/hot容量，标准16384 | 缓存配置 |
| K | 每query选择数量，标准2048 | 算法配置 |
| G、D | 前半打分的head数32/64及维度128 | 模型几何 |
| P | LI score生产的history分片数 | 当前条件P2或fallback P1；与TopK分片T不同 |
| T | 每query的TopK history分片数 | 研究策略参数；标准脚本支持1/4/8，生产仍T1 |
| M_b | request b的四Q里selected非PAD且location未满足的位置数，含跨Q重复 | Resolve初始化扫描实际数据得到 |
| D_bq | query q首次认领、分槽的新增unique token数 | 有序dedup/分槽实际得到 |
| S_b | Service实际分配的prefetch事件数 | 运行时计数 |
| E_b | Index事件数S_b+ΣD_bq | 运行时计数 |
| W | 每Q跨request分配的Fetch worker数 | 调用者配置1/2/4，默认2，当前没有自动公式 |
| V | 本次允许使用的AIV核预算 | 物理资源与运行配置；不是所有阶段始终用满 |
| C | 本次允许使用的AIC核预算 | 与V分别约束；没有物理1:2任务配对要求 |

注意：GEMM文献或前半score代码中的`M/N/K`是矩阵行/列/归约维度。该`M`不是本文未满足位置M_b，该矩阵`K=128`也不是TopK的K=2048。相同字母必须结合上下文阅读。

## 一个request的完整例子

设B1/Q4/H131072/R16384/K2048、miss目标10%，整数m=round(2048×10%)=205。当前fixture令Q0与Q2共享冷集合A，Q1与Q3共享冷集合B，A/B不交叠，各205个。这里miss比例的分母是**最终选中的2048个token**，不是128K history。

为了看清各量，先假设无prefetch、hot均有效、冷token均未满足、没有额外tail/失效。

| 阶段 | 此例主要处理量 | 算法与产物 |
|---|---|---|
| Final Top2048 | 每Q131072个score，四Q共524288个score | 前面QK已将G/D归约为每token一个score；逐trunk合并running K，输出4×2048个选中位置 |
| Classification | 8192个选中位置，外加resident元数据检查 | 查hot-valid/reserved/mapping，得到location或未满足标记，生成保护bitmap；当前每Q还分块读取R元数据并多遍检查K |
| Resolve/Plan | 初始化检查8192位置；其中M=4×205=820 | 先合并全Q保护，再跨Q dedup、认领victim/slot；新增D=(205,205,0,0)，ΣD410 |
| Index exchange | E=0+410=410个新增事件 | 按依赖闭包快照后移动Key/scale、更新映射；事件涉及的行数不必恰好410，提交仍可能扫描H |
| Output/pack | 仍输出8192个token的ID/slot；另打包410个新增Fetch描述符 | Q2/Q3仍有完整2048输出，但共享已认领slot，不重复请求同一cold payload |
| Fetch | 410份unique payload，另受safe/conflict和完成依赖约束 | 656B时410×656=268960B逻辑payload；2624B时1075840B；不含协议开销 |

因此M820、unique D410和输出8192是不同数量。Index事件E还包含Service事件；预取已经认领的token不应重复repair，pending预取也可能已得到location，但payload仍未完成，必须等待真实完成再发布valid。

若有效预取已认领集合A里的40个token，则在同样其他条件下，这40个在Q0/Q2两处location已知，未满足位置降为M740，新增repair降为D370；Service事件S40，Index事件仍可为E410。这个例子只说明计数关系，不代表当前默认Service保证认领40个、预取已完成，或总带宽必然减少。存在无用prefetch/hot-victim恢复时，E和实际传输还可增加。

## 每个阶段具体在做什么

**Final Top2048**读取前面已生成的uint16 score，而不是再次对H条128维向量执行QK。T1表示一个query的有效H没有交给多个history-local TopK任务；UB放不下全部H也无妨，当前按8K/16K trunk迭代，将上轮K与下一块合并得到新的K。

T8时可将H均分为8段、每段16384个score，8个local任务各取2048，再从8×2048=16384个候选中取最终2048。每query多一个merge阶段。分片任务可以轮转到核上，T不是永久独占核数；精确实现还需保持score/tie/索引语义。

**Classification**回答“这2048个token分别位于哪里、哪些槽不能覆盖”。命中slot可直接得到location；未满足位置打标；所有query会用到的resident槽须保护。即使miss为0，仍要完成K个输出的分类；当前还存在R元数据读取成本。

**Resolve/Plan**回答“同一个miss是否已被别的query认领、应放入哪个槽”。例如Q0首次遇到A中token Y，为它分配slot s；Q2再次遇到Y只复用s。分victim必须先看到所有Q的保护并集，避免把Q3仍要使用的token换掉。

**Index exchange**维护打分侧的缓存布局和映射。若slot s中旧token X可被替换，则把所需Y的Key/scale安排到resident位置，并维护被挤出X与Y的双向关系。多个迁移可能形成链或环，必须先保存会被覆盖的行再写。每行是128B FP8 Key+4B scale，**不是656/2624B payload**。后者由Fetch搬运；二者必须在提交/完成时正确汇合。

**Output**回答“下游每个query的2048个结果，分别是什么逻辑token、读取哪个slot”。命中和共享结果都要输出；只有首次认领且需新搬运的unique token形成相应Fetch描述符。因此Output有固定K工作，也有随D变化的工作。正式mapping不能过早改掉，否则尚未完成的Output会读到半更新映射。

## 小B为何拆history，大B为何先沿b/q

研究组件中，local逻辑任务数为`B×Q×T`；T>1另有BQ个merge任务。B1/Q4/T1只有4个query任务；T8变成32个local任务，可利用更多AIV缩短单query的长扫描。代价是候选写读、merge、同步和调度。

B16/Q4本来就有64条逻辑query；T8会产生512个local任务和64个merge任务。已有batch/query并行时，应先检查能否平衡这些query，再决定是否值得增加T；任务变短和merge增加需实测，不能只按任务波次数判快慢。

这是一条**优化规划原则**，不是说当前生产已经将64条query分别放到64个核：C28/B16的general Q4路径目前使用32个score AIV，每个顺序处理两Q；P2 Final路径则按独立query映射。独立TopK研究组件与生产Final的调度边界也不同，尚未替换生产classification/publication。

## T、M、W如何确定

- **T：**配置/策略选择。现在脚本测1/4/8，生产T1。`ceil(V/(BQ))`只能提供填充并行度的候选参考，不能直接当最优公式；还要比较local长度、merge、UB与带宽，且遵守实现支持范围。
- **M：**扫描四Q的selected/location得到实际计数。无prefetch且每Q m个未满足时可为Qm；有失效、预取、PAD或重复时要按真实location统计。它不是可配置核数，也不是unique miss。
- **W：**当前人工配置，默认2。现尾部核预算要求`3B+4W≤V`，所以仅有上界`W≤floor((V−3B)/4)`，还必须取实现支持的1/2/4。这不是当前自动选择算法。

例B16/V56：W1需52核，W2需56核，W4需64核，因此W4不合法；W1/W2哪个快取决于实际ready字节、冲突和credit。B1/W2虽配置8个Fetch角色，但按b mod W只有每Q一个lane分到request，另一个为空；不会自动把一条request拆成两份均摊。

源码与版本边界见[全通路审计](end_to_end_schedule.md)、[自适应方案](adaptive_scheduling_plan.md)。本说明不新增性能结论。
