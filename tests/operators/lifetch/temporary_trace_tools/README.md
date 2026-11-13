# LIFetch 历史 trace 转换与合成器校准（临时工具）

## 1. 数据来源与输出

| 输入 | 来源 | 可提取信息 |
| --- | --- | --- |
| `pid*-sample*.pt`，metadata.schema=1 | 历史 `experiments/kvaf/capture_lifetch_inputs.py`；例如 lifetch-inputs-dp2-bs4 | 原始Q/Index、hot/LRU/布局、最终TopK、跨Q重叠、初始cold与位置分布 |
| schema-4 `manifest.json` + NPZ | `kvaf_decode_score_trace`、content=scores | 以上信息及逐窗口记录、分数/阈值、预取未使用量；保存real输入时支持原格式回放 |

旧采集器来源已核对 vLLM 历史提交 `2bdb0eef0`。转换只读原始文件，输出必须选择新目录。使用CPU Python、numpy；旧PT额外需要支持`weights_only=True`的PyTorch。不需要ACL/NPU。

输出：

```text
converted/
  index.json                       # 输入清单和完成状态
  capture-00000/
    manifest.json                  # 来源、校验和、原始dtype/shape和编码
    arrays.npz                     # 无pickle，保留原始数组；BF16/FP8按字节保存
    schema4/                       # 仅现代输入：保留可被现有loader加载的格式
      manifest.json
      scores.npz
```

`lifetch_analysis_bundle`是临时离线分析格式，不直接传给bench。旧PT缺少完整分数、逐窗口记录及稳定request ID，不能伪装成schema-4完整回放。旧PT的before/after字段全部保留，后续补转换不必重新采集已有字段。现代schema-4本来已经是现有接口格式，无需改变布局；原始`dp*/series.json`继续用于连续回放，工具不猜测或重建series。

## 2. 执行命令

在 KVIP 仓库根目录运行；这是临时目录，用完可以整体删除：

```bash
python tests/operators/lifetch/temporary_trace_tools/convert_trace.py \
  /vllm-workspace/vllm-ascend/experiments/kvaf/results/lifetch-inputs-dp2-bs4 \
  --output /tmp/lifetch-old-converted

python tests/operators/lifetch/temporary_trace_tools/collect_fit_targets.py \
  /tmp/lifetch-old-converted --output /tmp/lifetch-old-fit
```

现代trace（以下输入路径替换成实际目录，每次仅处理一个采集session）：

```bash
python tests/operators/lifetch/temporary_trace_tools/convert_trace.py /path/to/score-trace \
  --kvip-root /home/x00965276/KVIP --output /tmp/lifetch-score-converted

python tests/operators/lifetch/temporary_trace_tools/collect_fit_targets.py /tmp/lifetch-score-converted \
  --kvip-root /home/x00965276/KVIP --output /tmp/lifetch-score-fit
```

现代trace也可直接使用已有统计器，避免额外复制大文件：

```bash
python tests/operators/lifetch/support/calibration.py \
  --trace-root /path/to/score-trace --group 2 --round-range 1:16 \
  --output /tmp/lifetch-score-calibration
```

先看`fit_targets.json`总体分位数，再看`request_rows.json`的联合分布。缺失数据为null或samples=0，不当作零。旧trace不按owner推断跨轮request身份；现代trace只比较同DP/group/request/owner且scheduler round相邻的记录。一次输入只包含同一服务采集session。

## 3. 合成器需要采集哪些信息

| 类别 | 每个请求/每轮的字段 | 用于生成器 |
| --- | --- | --- |
| 工作负载分组 | 模型及版本、权重/量化、layer/group、DP、BS、history、resident、heads、有效Q、预算/窗口数 | 区分不同分布，不把不同模型/层混成一个均值 |
| 原始输入抽样 | Q、Index key、weight、query/key scale及dtype | 验证生成的Q/Index确实能通过LI产生目标分布 |
| 分数与边界 | 每Q最终分数、TopK cutoff、hot cutoff、cutoff附近margin、ties | 校准阈值误判、候选增长和低精度边界 |
| 位置 | 逻辑token位置、物理Index位置、hot有效状态、ABC范围、LRU候选 | 校准recency、局部性、hot命中和兜底压力 |
| 跨Q | 每Q TopK集合、交集矩阵、新增unique、cold集合交集 | 避免把Q3/Q4机械复制成空补拉任务 |
| 窗口 | 窗口已处理范围、阈值、候选数、预算、提交的逻辑token/目标slot | 校准候选到达顺序、预算消耗和预取曲线 |
| 有效预取 | 本轮最终选中/未选中的预取token、最终补拉列表、slot冲突 | 预取precision、覆盖率、读放大和冲突率 |
| 跨轮 | 稳定匿名request ID、owner、scheduler round、前后布局、hot状态 | 集合保留率、新增需求、再次访问距离、冷热转换 |

这里“未使用预取”=本轮预取了但未被最终TopK选中，不等同于错误的hot命中。需要按token连接记录，不能仅用两个计数相减。历史摘要只有均值/耗时表的日志不能替代这些输入。

现有工具已经提取上述数据中可直接计算的部分。缺失的候选/预算拒绝原因、真实KV内容、完整跨轮复用距离不由转换器补造；后续采集按表补齐。性能与带宽仍在正常bench中测，采集耗时不能当作算子性能。

## 4. 采集与拟合顺序

1. **先分析现有16轮BS4数据**：得到Q重叠、cold比例、位置直方图；用作初始样本，不能据此拟合所有模型负载。
2. **补轻量连续记录**：每请求连续32～64轮，stride=1，带稳定匿名ID；按BS 1/2/4/8/16、历史长度和层组分层，分别保留warmup/稳定decode。不是每个组合都强制采完整Q/Index。
3. **少量完整输入**：每种主要分层保存若干Q/Index样本及分数，用于数值重建验证。大张量按磁盘预算限制，逐轮manifest记录是否完整。
4. **以真实请求/session划分拟合集和验证集**：不把同一请求相邻轮拆到两边。保留模型、层组和几何标签。
5. **先拟合联合分布**：位置/recency + Q共享成分 + 分数margin，再拟合跨轮相关性。不要仅匹配miss%，也不要直接写目标TopK绕过LI。
6. **验证生成器**：实际运行LI，比较每Q/去重miss、Q交集、cutoff/ties、每窗口posted/unused/final、读放大及跨轮保留率。展示p10/p50/p90和窗口曲线，验收偏差阈值依据真实样本方差确定。

工具没有修改当前生成器或任何kernel；实际历史数据应留在服务器，不提交仓库。


## 第一轮：Q/Index 数值与相关性统计

```bash
python tests/operators/lifetch/temporary_trace_tools/profile_inputs.py   /tmp/lifetch-old-converted-v2 --output /tmp/lifetch-input-profile-v1
```

CPU执行，无训练、无NPU依赖。需要numpy和支持float8_e4m3fn的PyTorch。固定seed、每份输入抽样8192个Index向量及至多8192个标量，结果可复现；Q统计过滤padding并使用unpack_indices。Index先按物理池抽样，单列零向量比例及非零向量统计，不将“非零”误认为有效hot。输出profile.json（聚合统计）、layouts.json（布局元信息）、summary.log（回传日志及完整参数块），没有原始Q/Index值。

本轮先据真实数值、scale和相关性选择生成器结构；分数重建、完整Q/Index联合模型及连续轮身份核验尚未在此脚本实现。summary末尾保留完整参数精度以供复制拟合，前面可读表格使用两位小数。

### Aggregate generator and comparison (first iteration)

Run on CPU using converted legacy captures; no NPU, training, or tensor upload:

```bash
python tests/operators/lifetch/temporary_trace_tools/fit_generator.py \
  /tmp/lifetch-old-converted-v2 --output /tmp/lifetch-generator-v1
```

The script samples matching key/scale positions from active owners and valid
logical lengths. Alternating captures fit aggregate quantiles and Q correlations;
the other captures compare real and generated operands using the same CPU score
formula and sampled selection fraction. Adjacent captures may share requests:
this is a capture split, not independent-request validation.

Send `summary.log`. `generator.json` contains only aggregate parameters and can
be versioned; `generate(parameters, history, seed)` needs no real tensors.
To evaluate revised parameters, add `--parameters /path/to/generator.json` and
choose a new output directory. Defaults: 8192 sampled keys/request, selection
fraction 2048/131072. Configure `--sample-keys` and `--top-fraction` if needed.

Version 2 preserves aggregate Q coordinate means, cross-Q residual covariance,
and per-head joint weight/scale/norm statistics. Index generation uses conditional
coordinate means/covariances by scale bucket, with zero vectors in a separate
class. Parameters contain aggregates, not individual captured Q/Index rows.
The same command refits version 2; old version 1 parameter files are rejected.

`SCORE AUDIT` compares sampled captured winners against sampled non-winners after
BF16 score rounding and causal masking. Zero inversions are necessary before
interpreting score-fit results; this sampled test is not full TopK verification.
`score_audit.json` retains per-request counts. No conclusion about fitting quality
is inferred from local regression tests: compare the real-capture output.

The model does not yet represent cache residency or temporal evolution. Conditional
Gaussian coordinates approximate higher-order structure, and scale correlation
within each bucket is not retained. Sampled selection statistics are not full-history
TopK or actual prefetch counts. The existing benchmark generator is unchanged.
CPU scores approximate the mathematical operator, not instruction-level rounding.

Version 3 replaces independent per-head draws with joint cross-head Q residual
covariance and joint weight/scale/norm covariance. A fixed 12-candidate search
adjusts Q differences and conditional Index noise using four fitting snapshots.
Its objective includes input standard deviations, Q cosine, score quantiles,
selection overlap and union ratio. It never edits scores or TopK after generation.
`tuning.json` saves every candidate and the fitting targets. Evaluation uses only
the other captures and different random seeds; no evaluation metric selects a
candidate. This is a bounded CPU parameter search, not neural-network training.
Old parameter files must be refitted. Use a new output directory such as
`/tmp/lifetch-generator-v3`. The model still needs real-capture evaluation: a
lower fitting loss alone does not establish realistic benchmark behavior.

To isolate remaining version 3 errors without refitting, reuse its parameters:

```bash
python tests/operators/lifetch/temporary_trace_tools/fit_generator.py \
  /tmp/lifetch-old-converted-v2 \
  --parameters /tmp/lifetch-generator-v3/generator.json \
  --output /tmp/lifetch-generator-v3-replacements
```

Every held-out request compares real Q + synthetic Index, synthetic Q + real
Index, and synthetic both. Weights and query scales always travel with Q; key
scales travel with Index. All modes use the same sampled real rows and generated
operands. `summary.log` shows means and paired absolute errors;
`replacement_comparison.json` also saves per-request metrics. Only aggregates
are written, not real operands. Replacement effects are not additive causal
contributions. Reuse the same seed/sample settings as the preceding run for a
comparable synthetic-both reference. This adds two CPU scoring passes per
held-out request, but `--parameters` skips the twelve-candidate fitting search.

### Current iteration: version 4

Replacement results localized most overlap and central-score errors to generated
Q operands. Version 4 therefore holds the conditional Index model fixed and
samples Q coordinates, weights and query scales with the same random factors.
It computes a thin SVD of the standardized fitting snapshots (at most 16 factors),
without allocating a full coordinate covariance matrix. Exported means and
covariance factors are aggregate statistics; no captured rows or sample
coefficients are exported. This is not a privacy guarantee for those statistics.
The log reports retained variance so truncation is visible.

The version 3 parameter search and Q-difference shrinkage are removed. Refit with:

```bash
python tests/operators/lifetch/temporary_trace_tools/fit_generator.py \
  /tmp/lifetch-old-converted-v2 --output /tmp/lifetch-generator-v4
```

All three replacement comparisons still run on held-out captures. Version 3
parameter files must not be passed to this version. No additional Index-tail
fix is claimed; the replacement table continues to measure that remaining error.

### Compact built-in generator (no parameter transfer)

The usable small generator is `compact_generator.py`. It needs **no real trace,
no generator.json and no fitting step**. Constants were calibrated locally from
the previously supplied aggregate logs, not copied from the version 4 model.
It models a shared Index direction, negative-weight head alignment, correlated
Q4, a small weak-scale tail, and paired zero-key/default-scale rows.

```bash
python tests/operators/lifetch/temporary_trace_tools/compact_generator.py \
  --output /tmp/lifetch-compact --save-operands
```

`summary.log` compares 32 generated snapshots (8192 keys each) against the supplied
real-log averages. `--save-operands` additionally saves one full-history synthetic
operand set to `operands.npz`; `--history` defaults to 131072. Arrays are decoded
float32; Q/keys are rounded to E4M3FN and weights to BF16. This NPZ is not the
full replay schema: cache state and transport inputs are not included.
Python callers can use `generate(history=131072, heads=32, seed=...)` directly.
The default 32-head configuration is the calibrated one. The bench fixture has
not been switched by this change.

Local CPU evaluation (seed 20261104, 32 snapshots, 8192 keys) produced:
Jaccard 0.91 vs reference 0.87; union ratio 1.10 vs 1.14; score p50 -2167.87
vs -2037.68; p90 -1548.11 vs -1570.17; p99 -394.45 vs -477.53.
This is comparison to aggregate reference logs, not paired validation against
real tensors. Per-request, per-head, temporal, and cache-residency distributions
are not reproduced by these few constants. In particular, weight/scale sharing
across Q is a simplifying assumption, and overlap is slightly too high.

### Bench and diagnose integration

`test_lifetch.py bench` and `diagnose` now use the built-in compact generator for
both URMA and UBmem synthetic inputs. `check` keeps the controlled fixture.
The generator lives in `support/compact_generator.py`; the old temporary entry
for offline evaluation forwards to it. No parameter file is required.

`--miss-percent` controls how much of the selected Q4 union resides in hot cache.
It changes residency, not Q/Index values. Per-Q rates can differ from the target;
the fixture report stores `actual_miss_percent_per_query` and the detailed logs
show the measured counts. For example H131072/R16384/seed20261104/target10 gives
9.67%, 9.86%, 9.77%, 9.96% in the CPU selection. Device cutoff ties can change the
exact counts. The comparison table's Miss% remains the requested target.

Native and fused paths use the same operand bytes. The independent numerical
TopK check accepts cutoff ties and one BF16 relative spacing around the CPU
cutoff for reduction differences; it still rejects duplicate/out-of-range IDs,
low-score selections, and omitted above-cutoff winners. KV bytes, ownership,
Index updates and SFA checks remain enabled. Controlled boundary coverage stays
in check mode, rather than relying on the compact workload to force collisions.
The fixture manifest identifies `compact_builtin_v1`; previous benchmark numbers
from the controlled fixture must not be interpreted as an operator regression
comparison against this new workload.
