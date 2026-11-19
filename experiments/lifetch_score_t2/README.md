# MR14 分核 direct producer ＋ MR23 T2 融合尾部

默认实验已切换为窗口分核候选。MR14 参考提交为 `c684d79f3a4f8432bf0915edeba392c7d54ba289`，MR23 基线固定为 `a56154c37bebb0928da4a4095e9670898e6b21c7`。这里适配 MR14 的 history 分核、真实 canonical 地址直写和 producer predicate 机制；不是把旧 staged adapter 的 Context/lease/ABI 整文件移入 MR23。

## 默认五个 Batch、两条融合路线

默认 `--cases mr23,window_budget --batches 1,2,4,8,16`。`mr23` 导出固定提交的原融合链；`window_budget` 为分核 Hot → 原 θ → Service0 → 各 Cold 窗口分核 score+predicate/原 Service → 原 Final。

| Batch | history P | 候选 score AIC/AIV | MR23 score AIC/AIV | 实际窗口 W（请求4） | 候选 LI kernel 数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 16/32 | 2/4 | 2 | 6 |
| 2 | 16 | 32/64 | 4/8 | 4 | 10 |
| 4 | 8 | 32/64 | 8/16 | 4 | 10 |
| 8 | 4 | 32/64 | 16/32 | 4 | 10 |
| 16 | 2 | 32/64 | 32/64 | 4 | 10 |

每个 batch 的每个 request 用 P 个 score 核组，整256-key页均分。默认 P 是实验起点，B1 保留已有 P16；这不是五种规模均已实测最优的声明。可以用 `--parts-map 1:16,2:16,4:8,8:4,16:2` 调整，并用 `--batches 2,4` 单测子集。脚本拒绝重复、未覆盖、无效配置及 `B×P>32`，不会悄悄改为多波。原 MR23 为4次 LI kernel；后续 Resolve/Get 每路另1次，全部包含在测量内。

每路后续均执行 MR23 的 Resolve、safe/conflict Fetch、CQ join、Output 和 Index commit。每个任务同时测量 `SerialBefore = SDK LI + LIDU + 完成的 BatchGet`。它是优化前串行参考，**本脚本没有测 MR14 的优化后串行链 SerialAfter**，不能用 `/SerialBefore` 证明融合比优化串行快。SFA 是计时外的正确性检查，不报告模型吞吐。

score 与 Cold predicate 使用同一 AIV 的 UB 结果，直接写原 `[packed_row, Q4, H]` sortable BF16 uint16 和原 bitmap。P 以整 256-key 页均分；每 AIC 的两个 AIV 分别处理 Q0/1 与 Q2/3。Cold 使用唯一 resident θ，严格 `score > θ`。没有 score scratch、scatter、另一个全 Cold score 读取或分片 radix 合并。

所有 P 分片完成后，独立 Service kernel 才发布两个 T2 ready cursor并调用原增量 TopK/ranked admission。producer 不允许以乱序结束位置提前发布全局 ready。resident 和 Final 仍每真实 Q 一个 AIV；跨 Q 去重、victim/映射依赖留在 MR23 后处理。

B1/P16 新增 score/predicate 并行度；B16/P2 和 MR23 都用 32 AIC/64 AIV，主要区别是解除 score 核内 Service 停顿并消除额外 Cold score 再读。窗口启动、重新载入 Q/scale 和独立 Service 的代价可能抵消收益，不能保证 B16 更快。新方案并未让 TopK/Service 全部继承 P16 核数。

## A5 执行

宿主机更新共享工作树（容器不需要网络，也不需要 docker 命令）：

```bash
git -C /home/w00568585/wt-lifetch-score-t2 fetch origin codex/lifetch-score-t2
git -C /home/w00568585/wt-lifetch-score-t2 switch --detach FETCH_HEAD
```

容器中直接运行仓库脚本，使用已经准备好的固定 KVAF：

```bash
cd /home/w00568585/wt-lifetch-score-t2
bash experiments/lifetch_score_t2/run_a5.sh \
  --device 5 --kvaf-root /home/w00568585/KVAF-score-t2-1c131038 \
  --kvaf-server 127.0.0.1:28885 \
  --batches 1,2,4,8,16 --cases mr23,window_budget \
  --miss-percent 10 --prefetch 1024 --intermediate-fetches 4 \
  --rounds 1 --warmup 4 --iterations 20
```

默认每轮10个任务，同 batch 的基线与候选相邻排布，轮次反转路线顺序并旋转起点，r0 丢弃。上述 `--rounds 1` 共20个任务；`--rounds 3` 共40个任务。每个任务只请求一种 batch，保持基线/候选相同的 DDR pool 容量。每个任务内部执行指定的全部 miss 用例。正式多 miss 结果使用 `--miss-percent 0,3,5,10,15,20 --rounds 3`；预取控制使用 `--prefetch 0,1024`，任务数相应加倍。输入、seed、KVAF、后处理和全图计时协议保持相同。所有测量轮 correctness 必须 PASS，失败保留日志而不生成收益结论。

额外加入之前的“完整 score → Cold 再读分类”控制：

```bash
bash experiments/lifetch_score_t2/run_a5.sh \
  --device 5 --kvaf-root /home/w00568585/KVAF-score-t2-1c131038 \
  --kvaf-server 127.0.0.1:28885 --miss-percent 10 \
  --prefetch 1024 --intermediate-fetches 4 --rounds 1 --warmup 4 --iterations 20 \
  --batches 1,2,4,8,16 --cases mr23,window_budget,full_budget
```

`full_budget` 使用同一 P 表，隔离“全 H 算完再 θ、T2 再读 Cold score”的成本。旧固定名称 `window_p16/window_p2/score_t2_p2/score_t2_p16` 继续可用，但只覆盖其原来支持的 B1/B16，不替代新的五 batch budget 路线。

单候选调试入口：

```bash
python tests/operators/lifetch/test_lifetch.py bench \
  --transport urma --device 5 --kvaf-root /home/w00568585/KVAF-score-t2-1c131038 \
  --kvaf-server 127.0.0.1:28885 --batches 1 --history 131072 --resident 16384 \
  --index-heads 32 --miss-percent 10 --prefetch-tokens 1024 --intermediate-fetches 4 \
  --li-score-parts 16 --li-score-mode window --warmup 4 --iterations 20 \
  --output /home/w00568585/lifetch_window_b1
```

不传 score 参数仍为当前 source 的原入口；精确 MR23 基线应通过包装脚本导出固定提交。窗口候选暂限定 synthetic Q4/head32、B≤16、统一长度和原 T2 shape。serving replay/ABC 的近期区、causal/padding 与热缓存布局尚未适配，CLI 明确拒绝使用；不能当成 vLLM 模型端到端已验证。

## 依赖、构建与结果

KVAF 固定 `1c1310387ebfb6776a07316874e924265b209ada`。已有其它版本时，添加 `--prepare-kvaf`，从 `--kvaf-root` 的本地 Git 历史准备一个独立 pinned checkout，并在其中编译/应用本仓 device-header 适配；保留源目录和服务器。若本地历史没有目标提交，先在可联网宿主机 fetch，容器不会自动联网或跳过版本保护。UMDK 如需定位，可设置 `KVAF_UMDK_ROOT` 为含 `include/urma_api.h` 的目录。

失败 CANN cache 使用新 `attempt-*`，避免 `merge_obj_text.sh` 原地把 REL `.o` 链为 EXEC 后重试产生 `unknown file type`。两路统一使用该启动构建恢复；基线 kernel、launch 与计时源码仍为固定 MR23，替换及 SHA 保存在 `baseline_build_override.json`。

结果目录包含 manifest、jobs、每路日志、源码 diff、samples、summary、input hashes。`score_mode/score_geometry` 记录实际窗口/核数/ready 范围；requested W4 在 B1 实际收敛为 W2。正式时间来自同一测量图 event/calls，包含所有新增 Service kernel。原始 profiler chrome trace 位于各路 `fused/diagnose/round-1/trace.json`，可用 Perfetto 查看。

`summary.csv` 新增以下列：

- `score_peak_aic/aiv`、`actual_windows`、`service_aivs`、`li_kernel_launches`：记录实际资源/调度，不把设备总核数当作活跃核数。
- `serial_li_us/serial_lidu_prepare_us/serial_fetch_us`：优化前串行链同图阶段；`fused_event_us/li_us/resolve_fetch_us/gap_us` 为融合整链与阶段。
- `score_hot_us/score_cold_us/score_total_us/theta_us/service_us/final_us`：窗口候选从同一测量图的具名 kernel 提取，每次调用检查 W 个 score、W 个 Service、1个 θ 和1个 Final，合计与 LI 必须一致。MR23/full 的内部 score/Service 没有独立 kernel 边界，对应列留空，不按比例猜分。
- `other_and_edges_us`：整链 event 减去 LI、Resolve/Get 和调用内 gap 后的余量，包含调用间隙、event 边缘及计时误差，不解释成 CPU idle。
- `net_gain_vs_mr23_us` 及 `li/resolve_fetch/gap/other_and_edges_gain_vs_mr23_us`：基线减候选，正数表示省时；四项损益加总必须等于完整链净收益。`per_round_over_mr23` 和 `serial_drift_over_mr23` 用于识别波动，不能只挑最快一轮。

score 加速是否转化为完整链收益，见 [ANALYSIS.md](ANALYSIS.md) 的五 batch 分析。旧独立 LI 的 B1/P16 `direct` 约52.14µs采用不同通路/计时方式，不能直接填进此次 SerialAfter 或用作此次加速比。

## 正确性检查

```bash
python3 -m unittest discover -s experiments/lifetch_score_t2 -p 'test_*.py' -v
bash experiments/lifetch_score_t2/run_window_contract.sh \
  --device 5 --kvaf-root /home/w00568585/KVAF-score-t2-1c131038
```

后者验证 B1/P16、B16/P2 的真实分核 score+predicate（H32768/R16384、各 Q/owner 不同 score/θ、严格 ties、输入/guard、禁止未汇合 ready 发布），不需要 KVAF server。`--mode sim --device 0` 可运行 CANN Model。多 owner 的快速契约检查可增加 `--batches 16 --history 4096 --resident 2048`；这只是地址/数学检查，不能代替 H128K 真机测量。它不验证 Service、Final 或 URMA。之前消费者 DMA 的 V→MTE2 同步修复及 sortable score 审计仍保留；旧 Model 记录见 `CORRECTNESS.md`，不能替代新链正确性。

本轮五 batch 脚本与实际 C++ 分片/cursor/核预算及同图拆解通过20项实验检查，构建恢复通过3项检查；本轮没有改动 kernel。前一提交58aa570的CANN 9.1编译链接、B1/P16（H32768/R16384）与B16/P2（H4096/R2048）的score/predicate Model契约已通过，详见 `window_validation.json`。这些 Model 用例不含Service、Final或URMA，也不是同尺寸性能对比；五 batch A5完整链正确性与收益待实测。
