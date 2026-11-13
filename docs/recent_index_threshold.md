# URMA 融合算子的 resident＋近期 Index 早阈值

融合算子在最终 TopK 完成前，先用 resident Index 估计第 K 大分数，决定哪些 KV 值得预取。刚退出 ABC 的 token 可能尚未出现在 resident 前缀中，遗漏这些高分候选会使初始阈值偏低，增加最终未使用的搬运。

本实现对 resident 与近期 Index 使用本轮 query 重新打分，统一求精确 Kth。近期范围由调用方提供为 `[前轮 ABC 起点，本轮有效末尾)`，每个 query 再按自己的因果边界裁剪；两段重叠位置只算一次。仅补 Index 计分，不晋升或重排 KV。

## 实现与接口

- 两段连续地址接入同一 LI 流水，复用 Q、weights、query scale。
- 初始阶段只求 Kth，不生成后续不用的完整 TopK 名单；适用长度下中间选择留在寄存器。大候选集合等不满足快速路径条件时，保留精确回退。
- 复用片上 resident 分数生成 hot 槽保护位图，保持原来的同步、覆盖保护和 Fetch 窗口协议。
- 全历史扫描、最终有序 TopK、KV 拉取与 Index 更新保持原语义。

`LIFetchBinding` 增加可选 `threshold_bounds`，为设备端连续 `int32[batch, 2]` 张量。调用方保持张量地址和生命周期稳定，在每轮执行前更新内容；区间必须对应当前 packed request 行，不能包含其他请求的前驱状态。`tail_bounds` 仍独立描述 KV 的当前 ABC 范围。URMA Context ABI 从 11/320B 升至 12/328B，Python binding 拒绝混用旧库。

vLLM adapter 在 policy 提供 `threshold_bounds` 时绑定其对应 group/batch view；旧 policy 没有该属性时继续 resident-only。独立合入本 PR 不会替调用方创建前驱状态。调用方须在成功准备一轮后推进前驱 ABC 起点，并在请求释放/owner 复用时清理；vLLM-DSA 的配套范围维护需单独同步。UBmem 不启用本次 URMA 优化。

## 五组真实输入回放

基线是原始 KVIP main `13d2898892652382ea36a0652030874efc31097d` 的 URMA **融合**实现，不是串行、较慢的中间版本或在新代码中关闭某个开关。数据来自最新一轮 `host16-urma-overlap-victims.5TTwxS` 的 `urma-register-kth` 档；未使用同轮已被否决的 overlap 档结果，也未跨轮挑选最优值。

环境：16节点物理 NPU4、Ascend950DT_9582，BS16/MTP3（每请求 Q4），128K 场景切片，hot cache 16384，单层 KV 656B，URMA/THP。八个 DP 的真实 Q/Index、请求与缓存状态合成单卡 BS16；KV 内容为确定性合成字节。每档预热2次、正式计时10次，无额外 profiling。

| 负载 | cold miss | main p50/mean μs | 本方案 p50/mean μs | p50降低 | mean降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| G2 | 2.1157% | 407.345 / 408.371 | 400.064 / 399.959 | 1.79% | 2.06% |
| G7 | 19.8192% | 690.879 / 704.615 | 638.961 / 639.548 | 7.51% | 9.23% |
| G10 | 3.3267% | 436.072 / 439.298 | 422.260 / 423.285 | 3.17% | 3.64% |
| G17 | 5.7658% | 473.989 / 476.170 | 447.463 / 448.440 | 5.60% | 5.82% |
| G4 | 9.8091% | 429.917 / 438.077 | 419.996 / 420.992 | 2.31% | 3.90% |

收益为 `1 − 本方案耗时 / main耗时`。百分比沿用原始日志；由三位小数时延反算可能相差0.01个百分点。

| 负载 | main误预取条数/调用 | 本方案误预取条数/调用 | 误预取字节减少 |
| --- | ---: | ---: | ---: |
| G2 | 598.9 | 31.2 | 94.79% |
| G7 | 2703.2 | 1746.1 | 35.41% |
| G10 | 1182.0 | 254.8 | 78.44% |
| G17 | 2343.8 | 1040.5 | 55.61% |
| G4 | 1283.7 | 456.4 | 64.45% |

误预取指已传输、最终选择未使用的 KV。每条656B；降幅以main的误预取字节为分母，不是总流量降幅或时延降幅。表内计数是10次平均数，字节降幅按显示计数换算。五组均通过有效 LI 分数逐位一致、最终有序 TopK 不变及 KV/Index 输出校验。

测量范围与限制：

- 包含新增近期计分、阈值计算及完整设备链；排除快照恢复、CPU调度和输入H2D，不能解读为端到端TPOT收益。
- 实测顺序为 main→overlap→register；main独立进程/DDR分配，两个feature共用arena。未保证主线与feature物理页相同，也未随机交替测试，因此不宣称统计显著性。
- G7/G4 main最大值842.084/507.534μs抬高均值收益，结论优先看p50，全部样本保留。G17上一轮main p50为495.213μs，本轮473.989μs；本次采用同轮5.60%，不引用前轮10.57%作为稳定收益。
- 新增槽编号读取重叠在本轮五组均劣化，未合入。未获稳定收益的跳过近期重算、seed-UB等实验路径，以及profiling分支也不进入本PR。

## 复现来源

历史回放工具及完整对照仍保留在实验提交 [`8f403014750e01bc5d8968aab459e42890e33b5e`](https://gitcode.com/CacheOS/KVIP/commit/8f403014750e01bc5d8968aab459e42890e33b5e)。其 `python -m kvip.diagnostics.compare --study overlap-victims` 输出中的 `urma-main-resident` 与 `urma-register-kth` 对应以上两档。实验工具、数据及新增启动脚本不随本PR合入。

原始结果位于验证目录的 `recent-index-threshold/host16-urma-overlap-victims.5TTwxS/g{2,7,10,17,4}-compare/summary.json`；每个文件保留全部计时样本、输入摘要和流量计数。上表是收敛前等价有效配置的实卡结果；收敛后的PR另做本地回归与CANN编译验证，不把本地检查冒充新一轮实卡性能测试。

收敛后验证：Python/adapter/runtime及几何测试64项、6个subtests通过；完整host/support回归119项通过；CANN 9.1对Ascend950DT_9582的URMA及UBmem共享源码兼容编译通过，Ruff和diff检查通过。
