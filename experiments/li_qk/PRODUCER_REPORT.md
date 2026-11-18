# B1/P16：score producer 与 Top2048 融合控制

固定 B1、Q4、H131072、G32、D128、K2048、history_parts16。一个 score task 处理完整 Q4 和8192 keys；score 使用16 AIC/32 AIV，三个候选的尾部均为4 AIV。输入沿用相同 finite FP8/BF16 fixture，全部真实Q使用完整有效H。仅执行数值LI，无resident阈值、Cold分类、Service、Fetch或Index。

## 四条完整计算链

| 名称 | producer | score之后 | 完整LI launch |
|---|---|---|---:|
| split_scatter | 原score写虚拟行scratch | scatter＋原per-Q radix Top2048 | 3 |
| direct | score直接写真实Q/history的canonical | 原per-Q radix Top2048 | 2 |
| high_fused | 直接写canonical，同时累积每片高8位直方图 | 一个per-Q kernel：合并高位、扫描低位、精确τ、GT/EQ输出 | 2 |
| local_topk | 每AIV在UB保留两行8192分数，score结束后原生局部Top2048，写候选值/真实ID | 每Q一次radix选择32768候选，再映射真实ID | 2 |

独立输出sidecar保留虚拟输入row与真实owner/Q/history的映射，不修改Q/K/scale寻址。宏`LI_PRODUCER_CONTROLS`仅由新producer目标启用，旧`li_score`和生产默认路由不变。基线入口链接原`li_score`库，不使用包含producer hook的同名副本。

`high_fused`的当前循环单位是128 key。原生histogram以256元素统计，UB补128个0后从所有累计桶减去128，跨tile以uint32累计。每片每Q写256×4B的累计直方图，B1/P16总64KiB。唯一全局高位桶确定后，尾部每Q先一次搬入全部16片高位统计（16KiB，复用input UB），再按16K块扫描低位，随后读一遍score输出GT/EQ。完整τ属于全Q，不使用局部阈值替代。

`local_topk`新增32KiB UB保留本AIV的两行分数；局部选择复用score-only路径中空闲的原TopK scratch。每片保留2048/8192，即25%候选，每Q全局候选数32768。局部分数与真实token ID分别写GM（全Q合计768KiB），全局只读取候选分数并对选中的2048个候选读取真实ID。无需有序归并；局部候选无序也可做精确全局TopK。同分数允许任意合法EQ子集，保证所有全局GT必选。

为保持输出与oracle检查一致，三个候选都保留完整canonical score写出；因此`local_topk`的计时**包含完整score输出与候选额外写出**。本轮没有省略完整分数输出。后续是否可以省略需要依消费接口另做控制，不能从本轮结果直接减掉估计时间。

## 计时口径

A5原始完整LI直接以ACL event测量，不使用阶段p50相加。`score`指标包含该方案全部producer工作（high累积或局部TopK）；`tail`指标包含score之后全部工作，基线的tail包含scatter。三轮交错同轮配对，warmup4/iterations20，丢弃r0。比值`li_over_baseline`小于1才表示完整LI改善。msprof另行运行，包含完整LI、producer-only及tail-only；不能把这些调用的kernel总量当作一次完整LI。

Model采用CANN9.1.0 / Ascend950PR_9599、32AIC/64AIV拓扑，固定用例每条链warmup0/iteration1。LOG只开core0，不限制计算核数。L2模拟关闭。whole ticks不是A5微秒，core0日志不代表全核指令数。本轮目标是完整数值正确性及初步Model对照。

## A5仓库入口

宿主机更新共享checkout：

```bash
cd /home/w00568585/wt-mr14
git fetch origin codex/split-core-unified-comparison
git merge --ff-only FETCH_HEAD
```

容器执行（不调用Docker，不联网）：

```bash
cd /home/w00568585/wt-mr14
bash experiments/li_qk/run_producer_a5.sh --device 5 --profile
```

输出目录为打印的`li_producer_*/`，四组结果在`results/summary.csv`，trace在`msprof/r0_<case>_trace/msprof/`。脚本记录源码commit/status、构建与输入hash、正确性report和全部轮次日志；build/运行失败会非零退出并显示错误尾部。

独立Model命令（CANN环境中）：

```bash
python3 experiments/li_qk/producer_compare.py build --run-mode sim \
  --cann-root /usr/local/Ascend/cann-9.1.0 \
  --kvaf-include third_party/KVAF/access_operator/demo/aiv_batch_get/include \
  --output /tmp/li_producer_sim_new
python3 experiments/li_qk/producer_compare.py model \
  --build /tmp/li_producer_sim_new --output /tmp/li_producer_model_new \
  --model-lib /usr/local/Ascend/cann-9.1.0/tools/simulator/Ascend950PR_9599/lib
```

## 提交与验证

- `1467ce7`：方案一，独立输出映射和direct producer。
- `28ddd0a`：方案二，高位histogram融合及单kernel尾部。
- `94fae0a`：方案三，局部候选与单kernel全局选择，四组A5入口。
- `f46991e`：方案二修正，高位统计合并的UB复用同步，以及压缩计数读取同步。

推荐以最新MR14源码构建一次，再在同一二进制中比较四条链；以上提交划分用于代码审阅。sim/npu目标编译链接通过，host CPU检查覆盖四条完整链、未写scratch、canonical与全局TopK oracle，以及高位统计和局部候选的精确性/guard。标准用例的完整Model结果见本报告后续记录及`evidence/producer_model_20261118`。

## 高位合并同步修正

第一次完整`high_fused`运行未通过TopK检查，未接受其ticks。隔离尾部并导出中间值后，低位累计直方图与CPU oracle一致；合并高位统计存在偏差，导致nextK/τ偏移、GT/EQ配额错误。原循环在同一个1KiB scratch上重复`MTE2→Add`，只有`MTE2_V`门槛，缺少防止下一次MTE2覆盖旧vector输入的反向依赖；`PIPE_V`屏障不代替跨流水的`V_MTE2`依赖。

修正版一次搬入全部16片高位CDF到16KiB input UB，再做纯vector合并；低位扫描复用input前明确等待`V_MTE2`，压缩完成后再读AR计数。无新增kernel或额外UB，非法GT计数会停止输出，避免覆盖其他Q/metadata。修正后的隔离尾部已通过四Q的完整数值、mandatory GT、合法EQ和guard检查；该诊断不计入完整LI对比。

## 初步Model结果与证据

| 完整数值LI链 | Model ticks | 相对原分片链 | ticks变化 | correctness |
|---|---:|---:|---:|---|
| score＋scatter＋原TopK | 108827 | 1.0000× | 基线 | PASS |
| direct＋原TopK | 96466 | 0.8864× | 下降11.4% | PASS |
| high_fused＋单kernel精确radix尾部 | 146422 | 1.3455× | 上升34.5% | PASS |
| local_topk＋候选选择 | 107596 | 0.9887× | 下降1.1% | PASS |

这是单次Model程序`Total tick`对照，不是A5微秒或统计性能结论。原对照已使用P16 score分核，比较范围为三个producer优化的增量收益；不以SDK原生QLI、库上P2或LIFetch整链作这张表的分母。

优先在A5验证`direct`。`high_fused`虽然省掉scatter和高位的GM重扫，但当前每128-key tile补齐/统计/累积与屏障，以及尾部低位、输出两遍history扫描，仍是额外成本；若继续，应扩大producer统计聚合粒度并用trace确认。`local_topk`将每Q全局输入从128K降到32K，但每片保留2048/8192（25%），新增局部选择和768KiB候选输出抵消大部分收益，当前结果不足以替换原TopK。三个候选都是2次launch，性能差异说明需比较producer工作与全局尾部总成本。

证据位于`evidence/producer_model_20261118/`：四组report、可恢复原始stdout的gzip、未修改的原manifest、build receipts、输入hash、计算源码对齐记录、core0日志条目统计，以及`summary.csv/json`。原始大trace保留在本地容器`/tmp/li-producer-all-model`、`/tmp/li-producer-local-model`、`/tmp/li-producer-high-final-model`，未纳入Git。


这里只接受完整链的成功行。原始manifest保留运行时的commit、dirty状态和失败行；不将部分成功的原运行改成PASS。`split_scatter`与`direct`出自第一次完整运行，`local_topk`单独使用同一个原二进制运行，`high_fused`使用修正后重新构建的二进制运行。四组输入hash、拓扑和计时范围相同；A5脚本会在最终同一个二进制内交错对比全部四组。

原sim receipt记录构建时的本地提交`0728f85`和dirty状态，后续方案二提交整理为`28ddd0a`。receipt原文保留；`source_alignment.json`核对score/producer、radix原语、production hook等实际计算源码的hash，不能把构建时dirty代码伪装为该旧提交的干净构建。修正版的sim/NPU receipt另行保留。

Model日志仅记录core0。附带的`core0_instruction_counts.json`是三个硬件单元各自`instr_log`条目计数，不是全32AIC/64AIV的指令总数；它也不能代替关键路径ticks。比如直接输出增加了部分寻址指令，但消除了scatter依赖与搬运，其完整ticks仍下降。局部候选减少了所采样的日志条目，完整ticks却改善很小。最终物理设备收益应看直接测量的`li_p50`和独立msprof trace。
