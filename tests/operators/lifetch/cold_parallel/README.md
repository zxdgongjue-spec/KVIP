# A5：MR14 Cold predicate 并行整链对比

32 AIC / 64 AIV 的新一轮实验及可直接执行的窗口短名单见[实验报告](../../../../docs/research/split_core/a5_budget_experiment_20261117.md)。`run_a5_budget.py` 包装本目录的 `run_controls.py`，W2/prefetch512 为40个case；默认W2/W8、prefetch0/512共160个case，均包含同构建P2与原staged控制。预算不足会过滤超预算短名单。源码调度12,288项通过，不代表A5计时或设备ISA验收。

此目录随 MR14 源分支一起发布，直接使用当前 checkout 的 `test_lifetch.py`。无需另外应用补丁。

默认保持 `score_route=legacy`。显式 staged 候选提供 `score_post=separate|scatter|direct`、`score_prepare=wave|invocation|phase`、`score_service=separate|inline`，以及独立 `score_hot_partitions` / `score_cold_partitions`。Cold 分类继承 score task 的两个配套 AIV；direct 在原 reduction 后直接输出 canonical＋bitmap。phase 分别合并 Hot、Cold prepare，Cold prepare 仍排在 Hot Service 后；inline 由 Service AIV 发布自己的 owner ready，消除独立 progress launch。每次调用仍刷新实时输入。

分片数0沿用原自动策略；显式1..32必须满足实际 `min(AIC,AIV/2)` 预算和128-key tile数量，只支持Q4/history或选择Q4的auto。`P_hot`、`P_cold`可不同；resident/Final TopK的history切分仍各为1，Fetch宽度单独选择。保留完整resident θ、四 Q Hot gate、owner唯一admission/slot/victim、epoch、Final/Fetch/CQ/Index的原完成依赖。inline不是段级Service并发：每epoch全部score完成后才消费，不能据此宣称消除了epoch等待。

私有 Plan ABI v3 为64B，Python与bridge必须一起重建，旧 staged bridge会被拒绝。公开Context/RuntimeShape ABI 11不变。CPU通过和源码操作仿真不是A5编译/数值/性能验收；A5结果仍待执行。

## 运行

更新单独的 MR14 checkout（有本地修改时先保留）并启用原有CANN/TorchNPU/KVAF环境：

```bash
git fetch origin codex/split-core-unified-comparison
git switch codex/split-core-unified-comparison
git pull --ff-only origin codex/split-core-unified-comparison
```

可先做CPU校验（Python＋NumPy、C++17）：

```bash
python3 tests/operators/lifetch/cold_parallel/verify_cpu.py
```

先打印实验命令，确认KVAF路径、device与设备是否空闲：

```bash
python3 tests/operators/lifetch/cold_parallel/run_controls.py \
  --repo "$PWD" --kvaf-root /path/to/KVAF --device 5 \
  --windows 8 --prefetch 0,512 \
  --output /home/w00568585/mr14_cold_parallel_bench
```

确认后加 `--execute`：

```bash
python3 tests/operators/lifetch/cold_parallel/run_controls.py \
  --repo "$PWD" --kvaf-root /path/to/KVAF --device 5 \
  --windows 8 --prefetch 0,512 \
  --output /home/w00568585/mr14_cold_parallel_bench --execute

python3 tests/operators/lifetch/cold_parallel/summarize_controls.py \
  /home/w00568585/mr14_cold_parallel_bench
```

默认固定B1/H131072/R16384/MTP Q4/heads32/miss10/seed43，q_workers2，warmup4、iterations20。一轮预热(r0丢弃计时)+三轮测量，各轮六配置交错、所有case串行。prefetch0/512分别汇总。需要比较窗口边界时用 `--windows 2,8`；检查尾mask可用 `--valid-history 131069`。`--aic-budget 28 --aiv-budget 56` 只在确认实际资源足够后使用；默认由原入口查询真实budget。运行前确认同设备/带宽域无其它benchmark；脚本不提供跨主机互斥。

六配置为：

| route | post | prepare | 用途 |
|---|---|---|---|
| legacy | separate | wave | 同构建库上P2生产控制 |
| history | separate | wave | MR14 staged原行为 |
| history | scatter | wave | 分类继承paired AIV，减少Cold GM回读 |
| history | direct | wave | producer融合输出，消除Cold scratch→scatter |
| history | separate | invocation | 单独控制prepare launch |
| history | direct | invocation | 两项候选联合 |

每case的源码先与candidate.json匹配；原入口按源hash构建，不要使用旧bridge缓存。运行失败立即停止并保留日志。汇总核对完整manifest计划、全部receipt/exit/log SHA和所有report PASS后再丢弃r0计时，拒绝缺测或失败case。输出 `full_chain.csv`、`summary.json`；child report保留实际资源、wave queryGroup/rows/stride与binary hash、原始samples。

## 独立分片与关键通路调优

先用W2对齐B1库上P2的有效窗口数。下面包含同构建P2、原staged、direct、phase、inline逐项控制，以及Ph4/Pc4、8、16、28的phase/invocation候选；共52个case（含一轮丢弃）。核预算28/56只在实际资源足够时填写，否则删去预算参数，并把分片列表限制在真实容量内。

```bash
python3 tests/operators/lifetch/cold_parallel/run_controls.py \
  --matrix tuning --repo "$PWD" --kvaf-root /path/to/KVAF --device 5 \
  --windows 2 --prefetch 512 \
  --hot-partitions 4 --cold-partitions 4,8,16,28 \
  --prepares phase,invocation --fetch-workers 2 \
  --aic-budget 28 --aiv-budget 56 \
  --output /home/w00568585/mr14_score_tuning_W2 --execute

python3 tests/operators/lifetch/cold_parallel/summarize_controls.py \
  /home/w00568585/mr14_score_tuning_W2
```

去掉`--execute`先预览。然后另选新output，将`--windows 2,8 --prefetch 0,512`补齐策略对照；可扩展`--hot-partitions 4,8`、`--fetch-workers 1,2,4`，但每个组合都保留匹配的P2和staged控制，case数相应增加。不要同时启动多组跑同一个device。

`candidate_over_legacy_mean<1`才证明相对库上P2整链变快；`candidate_over_staged_before_mean<1`只证明改善原MR14 staged。`best_candidates.json`按W/prefetch/Fetch/预算选择本轮网格中均值最快的候选，不能当作所有场景的最优点。将选中的参数固定后再独立复跑，确认收益复现。汇总核对实际AIC/AIV和binary hash；请求W8的legacy仍是有效W2，所以W8结果需单独解释。`native_chain`继续只是SDK串行参考。

首次生成后，可额外核对CANN配置（将路径换成日志中的实际build）：

```bash
python3 experiments/split_core/verify_staged_metadata.py \
  --auto-gen /path/to/build/auto_gen/v9_kernel
```

新`service_ready`/prepare_all/scatter_predicate必须是AIV-only，score和score_direct必须是MIX1:2。普通计时外采集msprof，确认阶段时间、首次Fetch、未覆盖join和ConsumerReady；该脚本不伪造尚未接入的设备时间戳。

## 如何判定

比较同轮、同W/prefetch/resources的 `prototype_with_update`：candidate/legacy <1 才表示整链变快。`native_chain` 是SDK QLI→LIDU prepare→KVAF串行参考，不是P2。保留原始056fe62的legacy复测，检查新增编译代码是否让基线漂移。B1 legacy请求W8实际Shape为W2；staged保留W8，必须记录这个策略差异，不能称两侧窗口数完全相同。

全Cold bitmap在计时外验证canonical key、唯一resident θ和queryEnds；原TopK/payload/mapping/LRU/Index/CQ检查继续。UBmem仍限legacy；staged diagnose/all没有统一timeline decoder，继续拒绝。要观察kernel时间可从bench外部采集msprof，不将其扰动计时混入普通样本。

实际before/after host代码仿真：B1/C28/V56/history/W8，原staged separate/wave为45次dispatch；上一版direct/invocation为31次；新direct/phase/inline/Ph4/Pc8为24次（相对原staged减少46.7%，相对上一版减少22.6%）。准备Q/weight/qscale载荷每个方向从3,842,048B降至1,029,120B（减少73.2%）。Cold score载荷仍保持上一版direct的3.5→0.875MiB收益，本轮没有再减；Cold比较次数与数学工作量不变。减少P降低复制/θ重复加载，却增长每task扫描长度，需要实测其算力与带宽权衡。

W2同候选为12次dispatch，Hot每组4096keys，Cold每组14336keys；W8的Cold每组2048keys。invocation同候选分别为11/23次，但所有Cold prepare移到Hot之前，须比较首次Fetch是否被推迟。源码操作/请求载荷不能称A5 ISA或实际速度，尚不能把原755.9µs直接乘比例得到新延迟。段级Service并发/Final私有buffer拆分尚未实现。

重现源码仿真（需要git中保留04ddfaa基准提交）：

```bash
python3 tests/operators/lifetch/cold_parallel/simulate_schedule.py \
  --output /tmp/mr14_score_schedule_simulation.json
```

结果见`docs/research/split_core/results/critical-path-simulation.json`；实现与限制说明见`docs/research/split_core/critical_path_candidate.md`。

## 全量探索后的短名单

源码主/补充网格共13,148项有效调度已检查。Ph4/Pc8复制少，但关键score循环较P28增长约3.6–3.9倍，因此不要只测低分片。建议先固定phase、Fetch2，W2与W8分别执行短名单；每组均包含五个逐项控制，prefetch0/512、一轮丢弃＋三轮测量，共72个case。预算28/56须先核对真实容量。

```bash
python3 tests/operators/lifetch/cold_parallel/run_controls.py \
  --matrix tuning --repo "$PWD" --kvaf-root /path/to/KVAF --device 5 \
  --windows 2 --prefetch 0,512 --partition-pairs 4:8,8:16,16:28,26:28 \
  --prepares phase --fetch-workers 2 --aic-budget 28 --aiv-budget 56 \
  --output /home/w00568585/mr14_explore_W2 --execute

python3 tests/operators/lifetch/cold_parallel/summarize_controls.py \
  /home/w00568585/mr14_explore_W2

python3 tests/operators/lifetch/cold_parallel/run_controls.py \
  --matrix tuning --repo "$PWD" --kvaf-root /path/to/KVAF --device 5 \
  --windows 8 --prefetch 0,512 --partition-pairs 4:8,8:16,16:16,26:26 \
  --prepares phase --fetch-workers 2 --aic-budget 28 --aiv-budget 56 \
  --output /home/w00568585/mr14_explore_W8 --execute

python3 tests/operators/lifetch/cold_parallel/summarize_controls.py \
  /home/w00568585/mr14_explore_W8
```

`--partition-pairs`覆盖hot/cold的笛卡尔网格；去掉`--execute`预览。首轮找候选，之后固定最佳参数独立复跑；再选择一组phase/invocation与Fetch1/2/4比较，避免一次大网格搜索替代独立复验。原056fe62的legacy需另行复测基线漂移；当前控制是同一新构建中的legacy/original-staged行为。

源码探索可重现：

```bash
python3 tests/operators/lifetch/cold_parallel/explore_schedule.py \
  --output tests/results/mr14-source-exploration-new
python3 tests/operators/lifetch/cold_parallel/plot_exploration.py \
  tests/results/mr14-source-exploration-new/primary_grid.csv \
  --output tests/results/mr14-source-exploration-new/tradeoffs.png
```

绘图需要matplotlib；源码调度需要NumPy、C++17。详细结论和实际Model边界见[探索报告](../../../../docs/research/split_core/optimization_exploration_20261117.md)。Model的生产score完整driver说明见[experiment README](../../../../experiments/split_core/README.md)。源码计数/条件分析/Model周期/A5微秒数分别记录，不能互相替代。

## 补测 scatter 与 direct 的打包粒度

默认短名单仍是 direct。补测可加 `--candidate-posts direct,scatter`，同时保留P2、原staged及逐项控制；`run_a5_budget.py`也接受该选项。默认32/64完整计划仍为160case，加两种post后为256case。低分片大task下，scatter的4096-key pack调用较少，但保留更多GM中转，不能仅按字节大小选direct。先用两个分片点补测：

```bash
python3 tests/operators/lifetch/cold_parallel/run_controls.py \
  --matrix tuning --repo "$PWD" --kvaf-root /path/to/KVAF --device 5 \
  --windows 2 --prefetch 0,512 --partition-pairs 4:8,8:16 \
  --candidate-posts direct,scatter --prepares phase --fetch-workers 2 \
  --aic-budget 32 --aiv-budget 64 \
  --output /home/w00568585/mr14_pack_grain_W2 --execute

python3 tests/operators/lifetch/cold_parallel/summarize_controls.py \
  /home/w00568585/mr14_pack_grain_W2
```

该补测为80case，计数含一轮丢弃＋三轮测量。预算不足时降低分片及预算；与主实验使用不同output，保持串行。然后固定较优组合独立复跑，必要时补W8。Model入口新增`scatter_inline_auto_W2/W8`，按同fixture、同budget与direct/phase/inline比较；没有设备测量前不宣布哪一条更快。
