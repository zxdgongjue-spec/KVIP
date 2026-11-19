# MR23 score 分核接原 T2 后处理

实验基线固定为 main 的 MR23 合并提交 `a56154c37bebb0928da4a4095e9670898e6b21c7`。本目录属于 `codex/lifetch-score-t2` 的实验候选；不是 MR14 staged 路线，也没有改默认生产入口。

## 调用与算法差异

| 路线 | LI kernel 链 | LI launches | 后续 |
| --- | --- | ---: | --- |
| MR23 原 T2 | Hot score → resident θ → Cold score/classify/窗口 Service → Final | 4 | 原 Resolve/Fetch/Output/Index commit |
| score_t2 | 全 H 分核 score 直写 canonical → 原 resident θ → T2 Cold classify/窗口 Service → 原 Final | 4 | 同一原 Resolve/Fetch/Output/Index commit |

T2 是每 request 两个 history owner：2 AIC/4 AIV。score 的每 AIV 处理两个 Q；resident θ/Final 的四个 AIV 分别处理一个真实 Q。两个 history owner 分别处理半个 resident、每个 Cold 窗口的一半。不是每 Q 各抢一个 victim。

新 producer 使用每 request `P` 个 AIC、`2P` 个 AIV；以整 256-key 页均分，内部仍为 128-key tile。`B*P <= effective_aic`，P 不能超过非空页数。真实 score 地址为 `(packed_row*4+Q)*H+history_position`，packed row 再通过原输入 block table/rowOwners 对应 cache owner。物理 AIC ID 不作为输出行。

scoreOnly 保留原 FP8 QK、反量化、权重/head reduction、BF16 sortable key 的数学路径；禁止在局部分片中执行 resident/Final TopK 或 Service。完整 score kernel 结束后复用原 resident θ kernel。Cold 消费每个 tile 一次，加载唯一 θ，严格使用 `score > θ`，调用与原 score kernel 共用的 bitmap/progress 发布函数。后续保留原 ranked admission、增量候选、Service predecessor fence、共享四 Q hot/victim 保护及唯一 owner。

全 score 已就绪也不提前发布所有 Cold ready：原 T2 按半窗口发布 bitmap + progress，Service 只消费其合法连续覆盖。Final 仍在 Cold/Service kernel 结束后调用，避免 `ctx->locations`、增量 TopK 和 classification scratch 重用冲突。URMA CQ join、safe/conflict Fetch、Output 消费旧映射、Index commit 均沿用原 resolver。

## 固定 A5 对比

先在可访问 Git 的宿主机更新本候选分支；宿主机和容器共同可见 `/home/w00568585`。容器不需要 Docker 命令，也不需要 Git 网络。进入候选 KVIP checkout，使用已构建且固定在 `1c1310387ebfb6776a07316874e924265b209ada` 的独立 URMA KVAF；不自动切换/覆盖 vLLM 的其他 KVAF。

```bash
cd /home/w00568585/wt-lifetch-score-t2
bash experiments/lifetch_score_t2/run_a5.sh \
  --device 5 \
  --kvaf-root /home/w00568585/kvaf-mr14 \
  --kvaf-server 127.0.0.1:28885 \
  --miss-percent 0,3,5,10,15,20 \
  --prefetch 1024 \
  --intermediate-fetches 4 \
  --rounds 3 --warmup 4 --iterations 20
```

路径请替换为实际已更新的 KVIP 与已构建的固定 KVAF。需要原 SDK `torch_npu` QLI、已注册的 A5 LIDU/SFA vendor，以及 CANN/URMA 环境；沿用之前已正常执行 MR23 bench 的环境。脚本调用新版 `tests/operators/lifetch/test_lifetch.py bench`，不是旧的无子命令 `lifetch.py`。

`run_a5.py` 本地 `git archive` 导出精确 MR23 基线，避免把未提交编辑或 build 软链接复制进去。两个 source 通过源文件摘要缓存分别编译 LIFetch。所有测试在同一个固定 KVAF 上执行，不替换 KVAF接口；native 串行参考会在每路测试中实际执行。

默认交错运行：原 MR23（B1/B16）、score P2（B1/B16）、score P16（仅 B1），r0 丢弃，再测 r1/r2/r3。P2 的 B1 控制组用于判断全 score 屏障/Cold 再读的代价；B16/P2 已使用 32 AIC，没有凭空增加 score 并行度。若设备有效核数小于 32，B16/P2 会拒绝执行，禁止超预算隐式分波。

要单独观察预取重叠贡献，改成 `--prefetch 0,1024`。要复用之前 W8/pf512 几何，使用 `--intermediate-fetches 8 --prefetch 0,512`。这些是不同参数实验，不能与 MR23 提交附表混成同场景复现。

仅跑一个标准控制组便于先确认编译/正确性：

```bash
bash experiments/lifetch_score_t2/run_a5.sh \
  --device 5 --kvaf-root /home/w00568585/kvaf-mr14 \
  --kvaf-server 127.0.0.1:28885 --miss-percent 10 \
  --prefetch 1024 --intermediate-fetches 4 \
  --rounds 1 --warmup 4 --iterations 20
```

独立入口（已有完整环境）：

```bash
python tests/operators/lifetch/test_lifetch.py bench \
  --transport urma --device 5 --kvaf-root /home/w00568585/kvaf-mr14 \
  --kvaf-server 127.0.0.1:28885 --batches 1 --history 131072 \
  --resident 16384 --index-heads 32 --miss-percent 10 \
  --prefetch-tokens 1024 --intermediate-fetches 4 \
  --li-score-parts 16 --warmup 4 --iterations 20 \
  --output /home/w00568585/t2_candidate_b1
```

不加 `--li-score-parts` 是当前 source 的默认入口；严格基线对比请用包装脚本的精确提交 source。`diagnose` 也支持该选项；`--level 2` 单独进行详细打点重放，正式时间仍来自未开启详细打点的整图。

## 产物与时间口径

脚本直接生成 `manifest.json`、`jobs.json`、每路 `.log`、源码 diff、`samples.json` 和 `summary.csv`。原始 profiler chrome trace 位于各路 `fused/diagnose/round-1/trace.json`，可用 Perfetto 查看；不是独立 `msprof` 指令 trace。正式结果使用同一次重放的 graph event/calls，另列 profiler p50/LI/Resolve 与调用内 gap，r0 不纳入均值。对比所有真实 operand `.bin/.npy` 校验和不一致即报错，结果不沿用之前独立 p50 的计时。

`Resolve/Get` 是融合 kernel 区间，包含 Resolve、Fetch join、Output、Index update 的工作与等待；不能直接当成纯 resolve 算法时间。LI 区间包含 Cold Service 内的预取推进，不能当成 score-only。SFA 用于完成后的正确性校验，**不在这个算子链计时内**；该脚本不能报告模型 token 吞吐。

## 当前验证状态

CPU 测试与脚本 dry-run 已执行；CANN 9.1 的 npu/sim 编译链接通过。B1/P16 score producer 的 CANN Model oracle/guard 校验通过，整次 Model 为 48726 ticks。完整 Cold/Service/Final/URMA 候选流程及 A5 硬件收益仍待验证；该 score 用例没有验证整个尾部。详见 `ANALYSIS.md` 和 `model_score_result.json`。候选暂不用于生产性能结论。
