# MR14 分核候选：完整探索与验证边界

本轮基于 `aa1ea91` 的 direct canonical＋Cold predicate、phase prepare、inline ready 和独立 Ph/Pc 参数继续验证。默认生产入口仍是 legacy。**现阶段不能宣布候选已最优，也不能将源码启动/字节缩减比例作为 A5 提速比例。** 数学覆盖、实际 CANN 编译、Model 周期、A5 URMA 整链分别验收。

## 重新核对慢因

`native_chain` 是 SDK QLI → LIDU prepare → KVAF 的串行参考。比较库上 P2 必须看 `score_route=legacy` 的 `prototype_with_update`，不能用不受路由参数影响的 native_chain 判断后处理是否受影响。

原 staged 的 resident TopK/全局 θ 只做一次。Hot 完整 canonical 就绪后调用一次 resident TopK；W8 后续七次 progress 各分类一个互不重叠的 Cold 区间，然后发布 ready。用户阶段汇总中 progress 的 362.4µs 是实际瓶颈，但“每窗口重新扫描完整 resident 八次”不符合当前源码。还需区分 SIMT 分类、kernel 初始化和 ready 发布的成本。

当前 direct 将 Cold 分类分配给每个 history score task 的两个配套 AIV，并在 reduction 后写真实 owner/Q/history 的 canonical＋bitmap。它继承了 score 的 task 并行度；resident/Final TopK、单 owner Service 与尾部依赖仍各有自己的并行来源。GM 中存在分数不意味着后续必须等待完整 history：legacy 可以先由完整 resident 建立 θ，随后推进 predicate/Service/预取。

尚有四个关键成本：

- direct reduction 后仍调用 SIMT `PackColdThresholdUB`。B1/H128K/R16K/Q4 总计 1,792 次打包 VF（Cold 896 个 128-key tile × 两个 AIV），每次 256 线程中仅 64 个有效；改变 P 并不减少这些调用的总数。
- 每 task 每真实 Q 的 θ DMA 读取 256B，实际阈值只有 2B。W8/P28 请求 200,704B θ padded DMA，尚未优化这个重复加载。
- 每 epoch 完成才运行 Service；inline 消除独立 progress launch，未引入 score/Service 设备并发。phase 只是合并 prepare 启动，Cold prepare 仍位于 Hot Service 后。
- resident 与 Final TopK 每真实 Q 的 history 分片仍为 1；Final 与 classification 的 buffer 别名依赖、Fetch safe/conflict/CQ join、Output 旧映射消费后 Index commit 全部保留。

phase 保留 Hot Service 的执行位置，却不保证首次实际 Fetch 更早：它会在第一个 Cold score 前准备全部七个 Cold 窗口；若首次有效预取来自 Cold，第一个 Cold ready 可能因集中复制而推迟。invocation 又会把所有复制移至 Hot 前，两者需按首次有效 Fetch 和 ConsumerReady 实测。可后续探索每2/4个Cold窗口合并prepare，或让producer以独立真实输入映射直接读取共享Q/weight/scale，避免虚拟row重复复制；当前未实现这些变体。

## 参数探索

执行真实 host dispatch 和真实 Python task allocator，检查每 owner/Q 区间精确覆盖。主网格 9,438 项覆盖 B1/C28/V56、W2/W8、Ph/Pc 1..28、三种 prepare、两种 Service，以及原 separate/scatter/direct 控制。补充网格 3,710 项覆盖 B1/2/4/16/32、C2/4/8/16/28/32、V16/32/56/64、Fetch1/2/4、history/BSQ/auto。共 13,148 项有效源码调度通过。

计数没有执行设备 ISA；准备字节仅表示 Q/weight/qscale 请求载荷的每个读/写方向，不含完整 GM 流量、cache sector 或 CQ 带宽。`critical iterations` 是各 score wave 最长 task 的 128-key 循环数之和，Q4 内可比较；Q2 的每循环计算量不同，不能直接横向按循环数估速度。

BSQ将Q4分为两个Q2 task，并不使逻辑QK MAC翻倍：两组加总仍为2,147,483,648 MAC（B1/H128K/G32/D128），源码也为Q2减半mBaseSize。它会让两组分别读取history，并改变key复用、每task循环长度、有效核数及Cube/AIV利用率；真实指令/带宽负担可以不同。“BSQ打分数学负载2×”因此不宜直接作为已证实的慢因。

以下均为 direct/phase/inline、B1/H128K/R16K/G32/C28/V56/Fetch2：

| W | Ph/Pc | 关键 128-key 循环 | 准备字节/方向 | θ padded DMA 字节 |
|---|---|---:|---:|---:|
| 2 | 4/8 | 144 | 205,824 | 8,192 |
| 2 | 8/16 | 72 | 411,648 | 16,384 |
| 2 | 16/28 | 40 | 754,688 | 28,672 |
| 2 | 26/28 | 37 | 926,208 | 28,672 |
| 2 | 28/28 | 37 | 960,512 | 28,672 |
| 8 | 4/8 | 144 | 1,029,120 | 57,344 |
| 8 | 8/16 | 72 | 2,058,240 | 114,688 |
| 8 | 16/28 | 43 | 3,636,224 | 200,704 |
| 8 | 26/26 | 40 | 3,567,616 | 186,368 |
| 8 | 28/28 | 40 | 3,842,048 | 200,704 |

Ph4/Pc8 将准备载荷降至约 21%（W2）/27%（W8），却将关键 score 循环增至 P28 的 3.89×/3.60×，所以低分片不应直接设为“最优”。W2 的 Ph26/Pc28、W8 的 Ph26/Pc26 与 P28 保持相同关键循环数，准备载荷分别少 3.57%/7.14%。这是源码成本上的有利候选，仍不能推导实际延迟优势。

大 batch 的最优参数会变化：B16/W8 显式 P28 形成 16 Hot＋112 Cold score waves、161 次 dispatch；P8 形成 5＋35 waves、62 次 dispatch，而两者关键循环总数均为 640。B1 的 P 值不能直接推广给 B16。

[source-exploration-20261117.json](results/source-exploration-20261117.json) 绑定完整 CSV、图、源码 SHA 与条件分析。源码图表见实验包中的 `tradeoffs.png`，只展示循环与载荷权衡。

## 条件分析能说明什么

用户提供的 W8 阶段近似值合计 764.6µs；本地没有原始 trace，故没有把它当设备测量重新认证。360 组系数敏感性分析分别改变 score/prepare 启动占比、新增 direct predicate 成本和 inline ready 成本。Ph4/Pc8 相对这个 staged 汇总的假设比值范围为 W2 0.278..0.836、W8 0.450..0.925。范围很宽，只说明一个 profile 点不足以识别最优 P；这些值不是 A5 预测，也不能写成已取得的加速比例。

更实用的检查是保留其余阶段不变：即便理想化删掉全部 progress/prepare/scatter，W8 仍剩约 298.6µs；保留一次 Hot scatter 则约 305.0µs。两者仍高于用户报告的 P2 kernel 合计约 277µs。修复 staged 劣化与超过 P2 必须分别验证，W2、Final/Service/尾部和重叠是否改善都很重要。该检查不是所有未来设计的绝对下界，因为未来也可能改变其它阶段及其重叠。

## 实际编译与 Model 验证

当前源码实际使用 CANN 9.1/950 编译，修复了 `GM_ADDR` 多声明在 CANN 地址空间类型上的编译错误。Model 设备构建和原生 URMA bridge 构建通过；生成元数据确认 prepare/scatter/progress/Service/TopK 为 AIV-only，score/score_direct 为 MIX1:2。公开 ABI11、私有 Plan ABI3/64B 不变。

Model driver 已支持实际生产 score、legacy 完整入口、W2/W8、separate/scatter/direct、wave/phase/invocation、inline、独立 Ph/Pc 与 Fetch1/2/4。旧 11 参数入口兼容；增加可选七参数。保留 score/全局 θ/Final 集合/owner/dedup/Index 四字段/656B payload/guards/completion 检查，legacy 按实际融合契约不要求落盘完整 canonical。两个公共 Model clock kernel 夹住 forward，排除 fixture 上传和 oracle 回读；结束 footer 与完整 oracle 都通过后才计算周期比值。

实际 Model 观察见[结果记录](results/model-exploration-20261117.json)：baseline一次1200秒超时、一次exit139，本轮重试仍无完整oracle/正常footer，状态NOT_ACCEPTED，没有可发布周期比值。不能沿用历史 wrapper 的 PASS 证明此次生产 score/direct kernel，也不能将 smoke 的 H8192/R6400 当成 H128K/R16K 性能验证。

Model 环境为独立容器 CANN9.1/950、32 AIC/64 AIV 模型，应用预算单独指定；L2 cache 默认关闭，日志显示 `Run in serial mode`。这描述模型配置和宿主执行方式，不足以证明模型忽略了逻辑多核。Model 用 prefetch0 与同步 GM final Fetch adapter，关闭了中间预取，故不能验证上述首次Fetch/隐藏通信等待收益；也不覆盖 A5/CANN9.2.1 的 URMA/CQE/DMA、非零预取以及真实调度带宽。Model 比例只能按这个 scope 报告。

CPU 全 42 组 B1..32/miss3..50 sparse-FP8/TopK/owner oracle 通过，12 项实际 host/pack/prepare/ready/汇总检查、9 项 staged 回归以及 60 项 runtime/kernel/binding 检查通过。CPU/math PASS 与真实设备 PASS 分开记录。

## 下一步优化顺序

1. 先测 W2 与 W8 下高/低 P 候选，固定相同构建、预算、Fetch/prefetch、fixture；逐项控制 direct、phase、inline。性能参数依据整链结果选择，不修改默认 legacy。
2. 若 direct 的分类启动仍显著，适配已有 `vector1::BatchClassifyScore`/register classify 路径，将比较留在 reduction 的寄存器内，直接产出需要的 2-bit bitmap。已有 `ClassifyScore` 的布局和尾 mask 与 staged 不完全相同，必须验证真实 Q/owner、queryEnds、score==θ、uint16 sortable key 和 bitmap 格式，不能简单关闭 scoreOnly。
3. 使用正确对齐/填充的阈值读取，减少 256B/Q 的重复 DMA；必要时分别处理 score-only 初始化的公共 TopK buffer 成本。源级字节收益需实际指令/Model/A5验证。
4. 再实现段级 descriptor 和连续前缀 ready，Service 保持 owner 唯一队列/slot/victim。C28/V56/P28 已占满 56 AIV；同时运行两 AIV Service 至少要 P≤27。W8/P26 保持五个关键 tile 且留下四 AIV，是可试预算，并不保证运行时真能重叠。多流、资源预留、可见性及乱序缺口都需设备验证。
5. Final 数值选择与 classification 分开时先引入私有 buffer，消除 locations/TopK 输出别名。分别探索 TR=1/2/4、TF=1/2/4/8；局部 TopK 需要保留每分片 K=2048，再全局 merge，不能以局部 θ 做 predicate。resident 小分片中 K 占比高，额外选择和 merge 可能抵消收益，TR/TF 不应跟随 Ph/Pc 自动增大。

若目标是进一步接近P2的融合流式路径，还应比较producer局部TopK＋全局merge：staged目前完整canonical写入加Final读回各约1MiB（B1/Q4/H128K），仅direct bitmap并未消除Final全量扫描。局部TopK输出最多每Q每分片2048个候选；P28下每Q仍有57,344候选，local selection与merge成本不能忽略。需明确全局token索引、tie规则、完整θ、TopK私有buffer和各producer完成依赖。它改变当前score-only契约，不能通过直接关闭scoreOnly实现，也尚未实施。

步骤 2..5 尚未实施。本轮输出是经编译与分层验证的可调候选、参数探索和可运行控制脚本，不是生产性能切换结论。

## A5 对比

[用例 README](../../../tests/operators/lifetch/cold_parallel/README.md) 提供完整命令。优先 W2 Ph/Pc=4/8、8/16、16/28、26/28；W8=4/8、8/16、16/16、26/26；保留 Pauto 和同构建 legacy 控制。prefetch0/512 分组，一轮丢弃＋三轮路由交错；不能同时占用同 device/带宽域。预算 28/56 先核对真实有效资源，报告必须记录 queryGroup/rows/stride。

整链 candidate/legacy <1 才表示超越 P2；candidate/staged-before <1 仅说明修复旧 staged。选中参数后固定参数独立复跑，避免搜索偏差。普通计时与 msprof 分开，采集 kernel 时间、首次 Fetch、未覆盖 join、ConsumerReady，保留 correctness/receipt/log/binary SHA；后面三个精确时间点仍需真机 trace 或后续设备埋点。
