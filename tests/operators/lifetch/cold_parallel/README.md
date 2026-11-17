# A5：MR14 Cold predicate 并行整链对比

此目录随 MR14 源分支一起发布，直接使用当前 checkout 的 `test_lifetch.py`。无需另外应用补丁。

默认保持 `score_route=legacy`。显式 staged 候选提供 `score_post=separate|scatter|direct` 与 `score_prepare=wave|invocation`。Cold 的 scatter 分类继承 score task 的两个配套 AIV；direct 在原 reduction 后直接输出 canonical＋bitmap；invocation 合并 prepare launch，同时保留每次调用的动态输入刷新。resident 全覆盖 θ、四 Q Hot gate、owner 唯一 admission/slot/victim、epoch、Final/Fetch/CQ/Index 的原完成依赖保持。

私有 Plan ABI v2 为64B，Python与bridge必须一起重建，旧 staged bridge会被拒绝。公开Context/RuntimeShape ABI 11不变。CPU通过和源码操作仿真不是A5编译/数值/性能验收；A5结果仍待执行。

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

## 如何判定

比较同轮、同W/prefetch/resources的 `prototype_with_update`：candidate/legacy <1 才表示整链变快。`native_chain` 是SDK QLI→LIDU prepare→KVAF串行参考，不是P2。保留原始056fe62的legacy复测，检查新增编译代码是否让基线漂移。B1 legacy请求W8实际Shape为W2；staged保留W8，必须记录这个策略差异，不能称两侧窗口数完全相同。

全Cold bitmap在计时外验证canonical key、唯一resident θ和queryEnds；原TopK/payload/mapping/LRU/Index/CQ检查继续。UBmem仍限legacy；staged diagnose/all没有统一timeline decoder，继续拒绝。要观察kernel时间可从bench外部采集msprof，不将其扰动计时混入普通样本。

源码仿真B1/C28/V56/history/W8表明dispatch45→31、Cold score载荷3.5→0.875MiB，但direct新增1792次predicate VF与PIPE_V barrier、200704B θ DMA，每VF仅64/256线程有工作。这是源码操作/请求载荷，不能称A5 ISA或实际速度。prepare_all只合并launch，复制量不减，且全部prepare提前到Hot之前；首次Fetch是否提前、未覆盖join与ConsumerReady下降仍需设备埋点。段级Service并发/Final私有buffer拆分尚未实现。
