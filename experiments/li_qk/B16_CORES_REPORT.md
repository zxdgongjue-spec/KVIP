# B16：30/32 核 score 调度对比

固定 B16，每个 request（owner）为 Q4/H131072/G32/D128/TopK2048。比较 30 AIC＋60 AIV 与 32 AIC＋64 AIV 的 score 预算，每个 AIC 核组配两个独立 UB 的 AIV。所有组用相同 fixture、同一次构建的 binary、相同 Final TopK。

这次测量隔离的数字 LI：`score → 必要的 scatter → 每个真实 Q 的完整 history Final TopK`。score 包含原 producer 的反量化、head reduction 和 sortable key 生成。没有 resident θ、Hot/Cold 分类、Service、Fetch 或 Index 更新；不能用这些数字直接宣称生产 P2/LIFetch 整链提速。

## 四种调度策略，各测 30/32 核

| 策略 | 行为 | 要验证的收益 |
|---|---|---|
| `waves / G2048` | 每 owner 平分两段 H64K，合计 32 个 task；按核预算逐波启动 | 原来的 30＋2 尾波与 32 核一波直接对比 |
| `balanced / G2048` | 展平全部 owner/half 工作，按 2048-key 量子给核分配等量预算；核在一次 score kernel 内循环处理片段 | 消除第二次 score 启动和尾波空闲 |
| `balanced / G128` | 同上，分配量子缩小到原生 128-key tile | 缩小最大核工作量与最小核工作量的差距 |
| `balanced_owner / G128` | 使用相同均衡预算，片段仅在真实 owner 边界切断，允许跨过原 H64K 人工边界 | 减少重复片段初始化和片段之间的 fence |

G 是**工作量分配量子**，不是 head 数，也不是每次重新初始化的循环长度。head 数始终 32，内部 score 计算 tile 始终 H128；同一 owner 的连续 tile 合并成片段。保留真实 owner 的 query、weight、scale、page 和输出位置，每核循环切换片段时沿用已验证的 CV/VC fence。

`balanced_owner` 的 T2 标签表示与“每 owner 两半”的基准比较；它取消人工 half 边界后，不保证每个 owner 恰好由两个物理核组处理。要严格保持两核/owner，使用 `waves` 控制。

默认 H128K 的实际计划：

| score 核数 | 策略 | score 启动 | 有效片段 | scatter 启动 | 每核累计 key 数范围 |
|---:|---|---:|---:|---:|---:|
| 30 | waves | 2 | 32 | 2 | 65536–131072 |
| 30 | balanced G2048 | 1 | 60 | 2 | 69632–71680 |
| 30 | balanced G128 | 1 | 60 | 2 | 69888–70016 |
| 30 | balanced_owner G128 | 1 | 44 | 2 | 69888–70016 |
| 32 | waves | 1 | 32 | 1 | 65536 |
| 32 | balanced G2048 | 1 | 32 | 1 | 65536 |
| 32 | balanced G128 | 1 | 32 | 1 | 65536 |
| 32 | balanced_owner G128 | 1 | 32 | 1 | 65536 |

key 数是沿 history 的累计工作量，每个 key 均计算完整 Q4/G32/D128。全批工作量为 `16×131072=2097152` 个 history key。30 核不能整数均分，按 H128 tile 均衡后只相差一个 tile。

30 核下原 32 个任务需要两波，第二波仅两个有效核组，score 临界路径需要先后完成两轮 H64K。表中 waves 的累计工作量按跨 launch 同号逻辑 block 相加，并不保证复用同一个物理核。均衡方案让 30 个核组分担全部工作，一次 score 启动。只保留 owner 边界后，有效片段由 60 降至 44，但 slot-major scratch 仍有两个槽，仍需两次 scatter。这里的 scatter 是分数重排到真实 `(owner,Q,history)` 位置，不是局部 TopK 的规约。Final 仍等完整 score 就绪，沿原完整 history TopK 路径执行，没有改变 Final 算法。

32 核时上述均衡方案与普通两半切分的任务映射相同，但循环 wrapper 带有额外控制/fence；它们未必比 waves 快。此前 G128/G512 优势来自 H8192 的 Model 实验，见 [GRAIN_REPORT.md](GRAIN_REPORT.md)，不能据此认定 H128K/A5 最优。可用 `--fine-grain 512` 替换两个细粒度控制，仍是八组；C30 的每核预算变为 69632–70144 key。

Final 统一使用 64 AIV（16×Q4），与 score 的 60/64 AIV 预算分开。30 核 score 期间余下的四个 AIV 没有安排并发后处理，本实验验证调度收益，不验证常驻 producer/consumer 流水。

## 宿主机更新与容器执行

宿主机和容器已有 `/home/w00568585 → /home/w00568585` 映射。宿主机更新**实际工作目录**：

```bash
cd /home/w00568585/wt-mr14
git fetch origin codex/split-core-unified-comparison
git checkout --detach FETCH_HEAD
git log -1 --oneline
ls experiments/li_qk/run_b16_a5.sh
```

`fetch` 本身不更新工作目录中的脚本。如果 checkout 因本地修改冲突而停止，先保留修改再处理冲突；不使用 `reset --hard`。容器无需 Git 联网、Docker 命令或共享 release。

在已有 A5 容器内：

```bash
cd /home/w00568585/wt-mr14
# 仅查看八组完整分配，不编译、不运行 NPU
bash experiments/li_qk/run_b16_a5.sh --plan-only
# 正式计时，随后单独采集八组 msprof
bash experiments/li_qk/run_b16_a5.sh --profile
```

默认 device5、实际 SoC `Ascend950PR_9599`，从当前环境寻找匹配 CANN 和 KVAF device headers。环境需要显式指定时：

```bash
bash experiments/li_qk/run_b16_a5.sh --device 5 --profile \
  --cann-root /usr/local/Ascend/cann-9.2.1 \
  --kvaf-include /home/w00568585/KVAF/access_operator/demo/aiv_batch_get/include
```

可另跑 `bash experiments/li_qk/run_b16_a5.sh --fine-grain 512 --profile`。512 与默认 128 分别生成新的结果目录；八组内严格共用 binary/fixture。

## 数据与判断标准

每组 warmup4/iterations20；整套 r0 丢弃，再三轮交错测量，每轮随机排列八种配置。独立 msprof 使用 warmup4/iterations5、phase=li，不重复额外 score-only 计时；不把 profiler 时间混入 benchmark。

结果目录 `/home/w00568585/li_batch16_cores_*`：

- `plan.json`：核预算、每段真实 owner/history 和每核任务量。
- `results/summary.csv`：八组 score/LI p50 均值及每轮配对比值；`measurements.csv` 保存单轮值。
- `results/manifest.json`、每组 report/receipt：binary/fixture、命令、oracle、guard 与日志证据。
- `msprof/C30_T2_*/trace/`、`msprof/C32_T2_*/trace/`：独立采集导出的 trace；未启用 `--profile` 时不生成。
- `checks.log`、`build-console.log`、`bench.log`、`exit-status.txt` 和 `launcher-status.txt`：执行阶段和状态；结果同时归档为 `.tar.gz`。

比值小于 1 表示该阶段更快：

| 列后缀 | 对照对象 | 回答的问题 |
|---|---|---|
| `over_32_waves` | 同轮 C32/T2/waves | 对正常 32 核两半切分有无收益 |
| `over_own_budget_waves` | 同轮相同核预算的 waves | 同为 30 或 32 核，调度改进多少 |
| `over_matching_32` | 同轮相同策略/量子的 C32 | 预留两核后的性能代价；C32 自比为 1 |

score 更快而 LI 更慢，说明重排/Final/启动等尾部成本覆盖了 score 收益；以 `li_over_32_waves` 和同预算的 LI 比值选候选，再看 trace 中 score、scatter、Final 的实际时间。不能把两个独立计时的 `LI p50-score p50` 当作精确尾部耗时。

只有八组全部通过数值 oracle、guard、源码/输出哈希和完整 r0/测量轮证据才生成汇总。成功标记为 `SUCCESS: eight B16/core strategies completed`；只看到 Results directory 或 CPU checks OK 不代表真机实验完成。

## 本地验证边界

2026-11-18：C++/Python 计划一致、每个真实 owner/Q/history 坐标唯一覆盖、八组实际 host 的 ACL stub 打包/score/scatter/TopK/guard 检查、汇总配对与错误预算拒绝、正式入口与 profile 计划检查通过。编排检查使用真实本地 Git 和模拟 NPU/backend，覆盖不联网、不改 checkout、参数传递、独立 profile、build 退出23后的停止及失败日志。

CANN9.1.0/Ascend950PR_9599 fresh NPU 编译与链接通过；当前计算 C++/header SHA 与构建记录一致。尚未在 A5 上运行新增八组，也没有新的 `balanced_owner` Model 性能数据。60→44 是静态片段数量变化，不是实测指令数或加速百分比。详细验证记录见 [VALIDATION.md](VALIDATION.md)。
