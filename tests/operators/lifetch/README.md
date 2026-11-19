# LIFetch 统一测试

唯一入口：`test_lifetch.py`。URMA、UBmem 共用 `check`、`bench`、`diagnose`，
通过 `--transport` 选择。调用仓库正式算子，不保留历史优化版本选择开关。
完整接口及日志字段见[测试设计](../../../docs/performance/lifetch-tests.md)。

## 代码结构

```text
lifetch/
  test_lifetch.py
  README.md
  support/
    workloads.py       # 配置、合成负载、真实输入及转换、轮间前态
    execution.py       # 初始化、共享资源、正式算子执行、进程管理
    correctness.py     # 数值参考、状态约束、固定检查用例
    performance.py     # 图计时、串行/融合配对、Fetch-only
    diagnostics.py     # 算法统计、阶段/逐核打点、带宽、失败现场
    reporting.py       # JSON、CSV、终端表格
```

## 准备

在 KVIP 根目录运行，使用已构建的 KVIP/KVAF 和已启动的对应 CPU service：

```bash
source /usr/local/Ascend/cann/set_env.sh
source scripts/env.sh
```

默认 `--device 0 --kvaf-root third_party/KVAF`；其他安装或设备显式指定。
URMA 和 UBmem 命令相同，但 KVAF 目录及服务必须与所选 transport 对应。

## check：正确性

```bash
python tests/operators/lifetch/test_lifetch.py check \
  --transport urma --device 4 \
  --kvaf-root /vllm-workspace/vllm-ascend/third_party/KVAF \
  --kvaf-server 127.0.0.1:28885 --timeout-seconds 600
```

默认覆盖初始化、BS1/2/4/8/16、全命中/全 miss、预算为零、跨 Q 重复/空补拉、
冷启动、冲突、LRU 候选外兜底、padding、连续多轮和非法参数。
`--suite basic --cases basic_bs1` 可筛选；`--suite contract` 不需要 NPU。
冲突/兜底/padding 的实际覆盖证据写入 `coverage.json`，不以普通用例 PASS 代替触发证据。

## bench：配对性能

```bash
python tests/operators/lifetch/test_lifetch.py bench \
  --transport urma --device 4 \
  --kvaf-root /vllm-workspace/vllm-ascend/third_party/KVAF \
  --kvaf-server 127.0.0.1:28885 \
  --batches 1,2,4,8,16 --miss-percent 3,5,10,15 \
  --warmup 4 --iterations 50 --timeout-seconds 3600
```

同一输入对比原生串行 LI/LIDU/Get 与正式融合完整图。终端只展示最终表格，
按命令的 miss、BS 顺序排列；保存 mean/p50/p99、原始样本、正确性和实际读取量。
默认合成器为内置标量参数的 `compact_builtin_v1`，无需 trace 或外部参数文件。
目标 miss 调整 hot residency；实际逐 Q miss 写入报告。check 仍使用可精确控制分支的固定 fixture。

一个 worker 共用最大 BS 的 DDR、runtime 和 HBM 参考 KV；DDR Put 一次，后续 case 不重复 Put。
构建使用运行时 shape 缓存，不按 BS/miss 编译。每组独立恢复 Index/hot/LRU，恢复及校验在计时外。
bench 不运行诊断打点或 check 的边界专项；资源准备、复用次数及墙钟成本保存在结果目录。

### A5 三路线完整链对比

在已构建 KVIP/KVAF、已启动 URMA CPU service 的容器中执行：

```bash
cd /home/w00568585/wt-mr14
source /usr/local/Ascend/cann/set_env.sh
source scripts/env.sh
bash tests/operators/lifetch/run_full_chain_a5.sh \
  --device 5 --kvaf-root third_party/KVAF \
  --kvaf-server 127.0.0.1:28885
```

KVAF 目录和服务地址需替换为实际配置。脚本在当前 checkout 执行，不更新 Git、不调用 Docker。
默认固定 Q4/H128K/resident16K/G32/seed43、AIC32/AIV64、W2/FetchWorkers2，
扫描 B1/B16、miss=0/3/10/50/90/100、prefetch=0/512，交错运行三条路线：

- `legacy`：P2 融合控制组。
- `history_before`：history/separate/wave/separate。
- `history_direct`：history/direct/phase/inline。

每条路线通过 `test_lifetch.py bench` 配对测 SDK LI → LIDU → BatchGet 的 Serial 完整链。
额外丢弃 r0，保留三轮测量，每轮 warmup4/iterations50。默认共 24 次 harness 调用、
288 个 BS/miss 用例（含 r0），不是 24 个 kernel。可先跑小矩阵：

```bash
bash tests/operators/lifetch/run_full_chain_a5.sh \
  --device 5 --kvaf-root third_party/KVAF --kvaf-server 127.0.0.1:28885 \
  --batches 1 --miss-percent 10 --prefetch 0 --rounds 1 --iterations 20
```

`--plan` 只打印命令，不执行或创建文件。默认执行结果保存至
`/home/w00568585/lifetch_full_chain_<timestamp>_<suffix>/`。
`summary.csv`/`summary.log` 给出测量轮 p50 的均值及逐轮配对比值；
`full_chain.csv` 保留 r0、mean/p50/p99、实际 score geometry、读取量和 report 路径。
`fused_over_serial`、`fused_over_legacy`、`fused_over_history_before` 小于 1 表示更快。
任一用例执行不完整或正确性非 PASS 时停止并保留已完成结果；不会生成成功汇总。

W2 避免 legacy B1 请求 W8 时实际降为 W2 的几何差异。`--q-workers` 控制 Fetch 并发，
不是 Q 切分。预算是请求上限，实际设备预算和调度以 report 为准。
串行与融合的预取及缓存维护工作不同；本脚本比较可用整链，不隔离纯融合收益。
`history_direct` 是 Cold canonical/predicate 候选，不是 LI 数值组件的 producer route=1。

### 原串行 / 优化后串行 / 库上 P2 融合三场景

需要隔离 LI 优化对串行链的影响时，使用这个入口；上述 `history_direct` 不属于优化后串行链。
脚本在 KVIP checkout 中执行，下面的 `wt-mr14` 是 KVIP 仓库目录。若 KVIP 安装在
vLLM-DSA 的 `third_party/KVIP`，也应先进入该 KVIP 目录。三路默认 miss 为
0/3/5/10/15/20%，可通过 `--miss-percent` 指定其他列表。

```bash
cd /home/w00568585/wt-mr14
source /usr/local/Ascend/cann/set_env.sh
source scripts/env.sh
bash tests/operators/lifetch/run_serial_vs_fused_a5.sh \
  --device 5 --kvaf-root third_party/KVAF --kvaf-server 127.0.0.1:28885 \
  --batches 1,16 --miss-percent 0,3,5,10,15,20 --prefetch 0,512 \
  --rounds 3 --warmup 4 --iterations 50
```

目录和服务地址需使用实际配置。同一用例内测三组：

| 输出 | 实际执行链 |
|---|---|
| `SerialBefore` | SDK QLI → 原 NativeMtp LIDU → 原 KVAF BatchGet |
| `SerialAfter` | 数值 direct LI（pack → 分核 score 直写 canonical → 原 TopK）→ **同一** LIDU → **同一** BatchGet |
| `LegacyFused` | 当前 checkout 的 legacy/P2 LIFetch 完整链 |

两条串行链使用相同 Q/K/weights/scales、初始缓存、队列和 KV 来源，恢复在计时外。
优化后 LI 的 query/weight/scale/block-table 打包计入完整 LI 和串行整链时间，
不会把组件脚本预打包后的 52.14µs 直接作为整链值。三组仅在传输完成后结束计时。
TopK 以现有独立分数参考校验；串行实际 unique miss 数必须一致，否则比较失败并保留日志。
边界相近分数可能产生不同 TopK 集合，实际读取量校验不能替代模型精度评估。
report 同时保存逐 Q 的集合/顺序一致性，便于识别边界 tie 或描述符顺序差异。

数值 LI 默认采用 Q4、均匀 history 分片，P 是不超过 `min(16, paired_score_capacity/B)`
的最大 2 次幂，最小 P1。B1/C32 或 C28 默认 P16；B16/C32 默认 P2，B16/C28 默认 P1。
可传 `--serial-history-parts 16` 固定 B1/P16，或 B16 用 `--serial-history-parts 2`。
超出一轮容量时有多轮 score；实际 P、waves、核数和二进制 hash 写入 report。
SDK 自行决定内部 tiling；预算参数约束候选和融合路线，默认采用整机上限。
首次自动构建数值 LI 共享桥接库，后续使用源码/工具链/SoC 缓存；不会改动 SDK 或生产默认路由。

默认三测量轮加丢弃 r0；两档 prefetch 共 8 次 harness 调用，每次包含三场景。
prefetch 仅作用于 legacy 融合链，两条串行链均为 demand fetch。
结果保存为 `/home/w00568585/li_serial_vs_fused_<timestamp>_<suffix>/`：

- `summary.csv`/`summary.log`：三条完整链以及 `LI_before`/`LI_after` 独立组件计时。
- `full_chain.csv`：各轮 mean/p50/p99、actual miss、实际分核几何、report 路径。
- `serial_after_over_before`：LI 优化对串行整链的影响。
- `serial_after_over_fused_legacy`：优化串行链与库上 P2 融合链的比较。

比值小于 1 表示加速。LI 单独计时与整链分别采样，不能用差值冒充精确 LIDU/Fetch 阶段时长。
现有公共入口也支持同一次测试：

```bash
python tests/operators/lifetch/test_lifetch.py bench \
  --transport urma --device 5 --kvaf-root third_party/KVAF --kvaf-server 127.0.0.1:28885 \
  --batches 1 --history 131072 --resident 16384 --queries 4 --index-heads 32 \
  --miss-percent 0,3,10,50,90,100 --score-route legacy --intermediate-fetches 2 \
  --compare-optimized-serial --serial-history-parts 16
```

## diagnose：算法与时间拆解

```bash
python tests/operators/lifetch/test_lifetch.py diagnose \
  --transport urma --device 4 \
  --kvaf-root /vllm-workspace/vllm-ascend/third_party/KVAF \
  --kvaf-server 127.0.0.1:28885 \
  --batches 4 --miss-percent 10 --level 3 --iterations 3 --timeout-seconds 600
```

各等级均按“算法/KV 统计”和“时间拆解”组织：

- level 1：总需求、miss、预取/补拉/unused、读取字节、完整图及阶段时间、区间有效带宽。
- level 2：逐请求/逐 Q/逐窗口数量、活动核、Fetch 分组计数与时间。
- level 3：逐核时间线、内部步骤和等待、LI 分类细节、可用的 Fetch-only 配对。

`--stages li,resolve,update,fetch` 筛选展示，不跳过执行依赖。性能图与打点图分开采样。
URMA 合成负载支持 Fetch-only；UBmem 和真实回放仍使用完整图诊断。
`all` 执行默认 check 后，用同一次配对测试生成 bench 和 level 1 诊断，不另跑 report。

## 真实回放与旧 trace 转换

已有 schema-4 `dp*/series.json` 可直接连续回放，必须包含真实 Q、Index keys/scales、weights、query scale：

```bash
python tests/operators/lifetch/test_lifetch.py bench --input real \
  --trace-root /path/to/trace --group 10 --round-range 1:16 \
  --transport urma --device 4 --kvaf-server 127.0.0.1:28885
```

串行与融合从相同初态各自演进；完整序列重复时才恢复初态。
Q/Index 来自采集，KV 使用可校验的身份数据；此项验证算子回放，不是模型服务 TPOT。
形状从 trace 读取，不再传 synthetic 的 BS/history/miss 参数。

历史 `pid*-sample*.pt` 或之前转换出的 analysis bundle 使用同一入口转换，仅需 CPU、NumPy、PyTorch：

```bash
python tests/operators/lifetch/test_lifetch.py convert-trace \
  /vllm-workspace/vllm-ascend/experiments/kvaf/results/lifetch-inputs-dp2-bs4 \
  --output /tmp/lifetch-real-replay
```

这替代了原 `convert_trace.py` 加 `convert_replay.py` 两步命令。
输出 `capture-00000` 等独立快照和 `conversion.json`；终端显示各快照的 BS、group 和校验结果。
旧采集不能证明跨轮稳定请求身份，因此每个快照独立运行，`--round-range 1:1`，group 取转换输出：

```bash
python tests/operators/lifetch/test_lifetch.py bench --input real \
  --trace-root /tmp/lifetch-real-replay/capture-00000 --group 10 --round-range 1:1 \
  --transport urma --device 4 --kvaf-server 127.0.0.1:28885
```

每次运行结果由唯一目录保存，失败也保留已完成 case。
