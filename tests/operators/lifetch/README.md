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

## 12 号机 Fixed Q4 九档配对

在 `kvaf_vllm_ascend` 中复用现有 CANN、KVAF `813f1750` 和 LIDU。生产树固定为 `af52a02b`，并使用 `10442f1f` 的 packed worker。宿主机执行 `npu-smi info` 确认物理卡 2 空闲。容器启动脚本用 ACL 核验物理卡 2 对应的进程设备号。

```bash
# 在容器内生成合成输入，直接跑九档。
bash tests/operators/lifetch/run_fixed_q4_a5_12.sh packed-synthetic

# G2_TRACE 是容器内已准备好的真实 G2 回放目录，如 case00002-r2-g2。
bash tests/operators/lifetch/run_fixed_q4_a5_12.sh packed "$G2_TRACE"
```

覆盖 BS1、2、4、8、16、20、24、28、32，每档 cohort 0。先在同卡为九档生成新的双 GET worker 串行 TopK oracle，再逐档按串行→融合、融合→串行测量。每个计时图包含十次独立状态，整图预热四次。保留 packed 预取预算 1024、W4、rank 1024。每个调用完整校验 TopK、KV 和 Index。CPU service 使用 THP，记录实际 NUMA、UB adapter，并核对 oracle 和两臂的放置。从 `RESULT_ROOT` 下的 `packed_summary.json`、`summary.csv` 读取九档结果。需要保留原始 trace 时追加 `--keep-trace`。若容器已固定 NUMA 或 UB adapter，可传 `--cpu-numa N` 和 `--cpu-ub-device NAME`。

合成入口固定 Q4、history 131072、resident 16384、32 index heads、目标 miss 10% 和 seed 43。合成请求复用同一组算分输入，每个请求保留独立的 KV 页身份。结果标记为 `synthetic`，用于评估分核趋势。真实 G2 性能仍以回放入口的测量为准。

`packed-synthetic --dry-run` 和 `--dry-run` 分别枚举合成及 G2 的九个 oracle 和 36 个计时图。

## 12 号机 Fixed Q4 加末窗 Q1

固定分核沿 K 页并行算分。末窗 Q1 在原 Service owner 取到 Score 就绪快照后，用 owner 和三个已完成打分的 AIV 各算一个 Q 的 Top2048。原 owner 汇合后跨 Q 去重，提交预取。BS1 保留 P8/W2，BS2、4、8、16 分别保留 P8、P8、P4、P2 及 W4。预取预算和 rank 起点沿用 MR38。

在 24 号机 MR35 的同 W2 对照中，Q4→4×Q1 在 BS1、2、4、8 上的 KV ready 降幅分别为 17.2%、21.3%、23.5%、23.3%。BS16 的 W4 末窗对照在 miss 5%、10%、15% 分别为慢 3.38%、慢 5.08%、快 0.53%。原宽 batch 数据用分组调用。此处的 BS20、24、28、32 测原生整批，输出串行和融合基线，Q1 标为不支持。

```bash
bash tests/operators/lifetch/run_fixed_q4_a5_12.sh packed-q1-pair --dry-run
bash tests/operators/lifetch/run_fixed_q4_a5_12.sh packed-q1-pair
```

沿用九档合成 Q4、H131072/R16384、目标 miss 10%、seed 43，先跑九个同卡串行 oracle 和原串行／融合 AB/BA 的 36 个图，再用同一二进制为支持 Q1 的五档各跑四个 Q1 开关图。每图十调用、预热四次。KV、Index、Top2048 及候选逐 Q 完成标记均在计时范围外核验。`RESULT_ROOT` 下的 `summary.csv` 和 `packed_summary.json` 给出整图与阶段耗时、预取、最终 Get 和工作量一致性。

## 12 号机原生大 batch 末窗 Q1

用相同合成输入独立跑 BS20、24、28、32。每档生成一份同卡强串行 TopK oracle。同一融合二进制运行以下三臂。

| 路线 | 末窗 W3 的执行 |
|---|---|
| 原生 `native` | Score 同一启动内，各请求由两个 AIV 各做两个 Q，随后单 owner 去重发单 |
| 后移 `deferred_pair` | Score kernel 完成后，由独立 AIV kernel 按每请求两路 Q2 执行 |
| 后移 `deferred_q1` | 与后移 2+2 使用相同步数和请求映射，每请求四路 Q1 |

两条后移路线由 16 个四 AIV 组按行处理，使用同一末窗就绪分数、每请求一个发单 owner。原融合的提前最终补取候选筛选只保留在原生臂，后移两臂由原 Terminal 完整处理。W0～W2、Cold Score 分片、预取预算和最终正确性规则沿用原几何。比较两条后移路线可测逐 Q 并行增量，比较后移 Q1 与原生可测整链是否抵消阶段拆分的开销。

```bash
bash tests/operators/lifetch/run_fixed_q4_a5_12.sh packed-large-q1-pair --dry-run
bash tests/operators/lifetch/run_fixed_q4_a5_12.sh packed-large-q1-pair
```

固定 Q4、H131072/R16384、目标 miss 10%、seed 43。三臂两轮反序，四档共 4 个 oracle 和 24 个计时图，每图十个独立调用。逐请求搬运计数和阶段耗时写入 `RESULT_ROOT/summary.csv` 及 `packed_summary.json`。

容器入口默认使用物理卡 2。若宿主机确认另一张卡空闲且该容器可见，在执行前把 `KVIP_A5_PHYSICAL_NPU` 设为该物理卡号。入口按该卡解析 ACL 设备号、持有同卡锁并写入结果目录，所有 oracle 与三臂仍在同一张卡上。

### 原生大 batch 的 Q1、Q2、Q4 和自动混排

```bash
bash tests/operators/lifetch/run_fixed_q4_a5_12.sh packed-large-q-policy --dry-run
bash tests/operators/lifetch/run_fixed_q4_a5_12.sh packed-large-q-policy
# 只对照原生、一波 Q2 和自动混排
bash tests/operators/lifetch/run_fixed_q4_a5_12.sh packed-large-q-policy --q-policy q2 auto
```

遍历 BS20、24、28、32，保持 Q4、H131072/R16384、W4、目标 miss 10%、seed 43、原预取预算和原生整批。每个 Q 独立做 Top2048。固定 Q1、Q2、Q4 表示每个 AIV 分别做 1、2、4 个 Q，每行分别占 4、2、1 个 AIV。Q2 是一波 32 支双 AIV 队伍，原三路入口保留 16 支四 AIV 队伍的后移 2+2 作为可复现对照。每行仍由一个 owner 汇合、去重和发单。

自动混排根据 W3 有效待扫范围估计本行各 Q 的工作量。令单 Q 成本为 `c[q] = 1 + ceil(新增可见 token / 128)`。每行的 Q1、Q2、Q4 选取耗时估计分别为 `max(c)`、`max(c[0]+c[1], c[2]+c[3])`、`sum(c)`。求最小单波尾延迟上界 `L`，每行选择能满足该上界且占核最少的策略，合计用核数不超过 64。若预计尾延迟和全 Q2 相同，沿用全 Q2。该估算未计入真实候选分布和硬件等待，净收益按同卡计时判断。

默认五臂包括原生、Q1、Q2、Q4、auto，按正反顺序执行。四档各一份串行 oracle，40 个计时图，每图十个独立调用，预热四次。逐 Q 回执验证覆盖完整；汇总区分实际 Q 分工、末窗耗时、整链及逐请求预取／最终 Get。标准合成输入各请求同长度，AUTO 可能全部选择 Q2，此时它主要测到规划开销。混排收益需使用有有效长度差异的输入实测。

## 12 号机 Q1/W2 加倍复测

用两个独立的干净检出目录。MR35 W2 实测版本为 `a9f4fe3b`，主仓基线为 `df8ad259`。

```bash
bash tests/operators/lifetch/run_fixed_q4_a5_12.sh q1-w2 "$Q1_CHECKOUT" "$MAIN_CHECKOUT"
```

保留 24 号机的合成输入、原生 P2 与 W2 Q4/Q2/Q1 四臂、512/128 预取预算、rank 1024、预热两次与五次计时。两轮共 64 臂，保持原矩阵的分组规则。BS20 为 5×BS4，BS24 为 3×BS8，BS32 为 2×BS16。结果放在 `results/q1-w2-card2` 下各次打印的 `Run output`。Fixed Q4 九档使用原生大 batch，按各自的计时口径解读两套结果。
