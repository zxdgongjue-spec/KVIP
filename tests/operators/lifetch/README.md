# LIFetch 统一测试

唯一 NPU 测试入口是 `test_lifetch.py`。URMA、UBmem 使用同一组命令，
通过 `--transport` 选择传输。测试调用正式算子，不包含历史优化版本的开关。
设计与完整字段定义见 [测试设计](../../../docs/performance/lifetch-tests.md)。

内置 check 已在 URMA、UBmem 真机通过；bench 矩阵待真机验收。合成负载仍是可控的
Q/Index fixture；真实分布校准需要下文的 trace 统计反馈，不能视为已完成拟合。

## 准备与运行

在 KVIP 根目录执行，使用已构建的 KVIP/KVAF，并确认对应 CPU service 已启动。
环境和依赖目录必须属于所选安装：

```bash
source /usr/local/Ascend/cann/set_env.sh
source scripts/env.sh

# 默认 URMA；服务地址按当前环境填写。
python tests/operators/lifetch/test_lifetch.py all --kvaf-server HOST:PORT

# UBmem 完整默认流程。
python tests/operators/lifetch/test_lifetch.py all \
  --transport ubmem --kvaf-server HOST:PORT
```

默认 device=0、KVAF=`third_party/KVAF`。其他目录或卡使用 `--kvaf-root`、
`--device` 明确指定。每次运行创建独立结果目录，`--output` 指定结果根目录。

## 三种独立模式

```bash
# 固定用例包括 BS2；配置和触发分支写入结果。
python tests/operators/lifetch/test_lifetch.py check --kvaf-server HOST:PORT

# 原生串行 LI/LIDU/Get 与当前融合路径的完整图性能。
python tests/operators/lifetch/test_lifetch.py bench \
  --batches 1,2,4,8,16 --miss-percent 10 \
  --warmup 4 --iterations 20 --kvaf-server HOST:PORT

# 完整图参考性能 + 独立打点采样；不拿打点耗时冒充无打点性能。
python tests/operators/lifetch/test_lifetch.py diagnose \
  --batches 16 --level 2 --kvaf-server HOST:PORT

# 内部步骤以及可用的 Fetch-only 对照。
python tests/operators/lifetch/test_lifetch.py diagnose \
  --batches 16 --level 3 --stages fetch --kvaf-server HOST:PORT
```

`all` 复用默认 bench 结果再采集 level 1；无需另跑 report 命令。
`check --suite contract` 的接口约束检查不需要 NPU。
阶段筛选只影响诊断内容，不跳过算子依赖。后端没有提供的专项显示
UNSUPPORTED，缺失事件显示 INCOMPLETE，不能按零耗时或 PASS 解读。

## 真实连续回放

目前读取已有 schema-4 score trace 的 `dp*/series.json` 与 NPZ。
必须包含真实 Q、Index keys/scales、weights 和 query scale；只有分数或 TopK
的采集可以用于离线分析，不能执行真实 LI 回放。读取时验证文件校验值、形状、
请求身份及连续轮号，不接受用合成数据补齐缺失输入。

```bash
python tests/operators/lifetch/test_lifetch.py bench --input real \
  --trace-root /path/to/capture --group 10 --round-range 1:16 \
  --transport urma --kvaf-server HOST:PORT
```

`round-range` 是源采集轮号闭区间。形状来自数据，不再传合成 BS/history/seed。
两条路径从同一初态各自演进，每次重复整个序列才恢复初态。KV 内容使用可校验的
合成身份数据，真实 Q/Index 参与 LI；该回放不是原始模型输出精度或服务 TPOT 测试。
地址重建保留页共享关系，不保留原始虚拟地址间隙。

## 离线生成器拟合分析

这一步只读取 trace，不启动 NPU，不需要 ACL，也不运行额外的单算子入口。
需要 Python 3.11+ 和 NumPy。沿用现有 trace 格式，含分数但未保存真实 Q 的数据也可分析：

```bash
python tests/operators/lifetch/support/calibration.py \
  --trace-root /path/to/capture --group 10 --round-range 1:16 \
  --output /path/to/calibration-output
```

输出包括：

- `summary.log`：请求轮数、初始 cold 比例、预取/unused/补拉数量、Q 间重合。
- `fit_targets.json`：分布与分位数、位置直方图、跨 Q 和相邻轮的 Jaccard 目标。
- `request_rounds.json`：每轮每请求的窗口数量、阈值、分数和重合统计。
- `sources.json`：输入文件校验值和选择范围。
- `analysis_status.json`：必须为 COMPLETE 才可使用本次输出；失败重跑可能保留旧产物。

请先回传 `summary.log` 与 `fit_targets.json`，无需上传原始 Q/Index。
这批统计描述采集时的缓存轨迹，不是重跑另一种策略的结果；阈值处等分时，
选中 token 的重合也受合法 TopK 选择影响。输出明确标记 `fitted_generator=false`。
后续用这些目标联合拟合 Q 重合、分数间隔和位置局部性，再通过真实 LI 检验生成负载，
而非直接生成 TopK 绕过 LI。

## 本地测试

```bash
PYTHONPATH=python python -m pytest -q tests/operators/lifetch/support
```

本地测试覆盖输入契约、状态推进、计时/报告和离线解析；不替代 NPU 的
正确性、图执行与性能验证。数值失败时保留已完成性能，设备执行失败时停止
依赖未知状态的后续工作。

## 本次预览版的支持范围与缺口

| 内容 | 当前支持 | 补全工作与影响 |
| --- | --- | --- |
| all/check/bench/diagnose | 共用入口、默认矩阵、结果自动汇总；URMA/UBmem 运行分发已接入 | 尚未真机验收，不能据本地测试保证两种传输均可运行或性能无变化 |
| check | 初始化 Put/Get、BS1/2/4/8/16、命中/全miss/预算0、跨Q重复/空补拉、冷启动、多轮命中→淘汰→重取、基本非法参数 | boundary_regular 强制检查冲突描述符、候选外hot兜底及padding；coverage.json汇总实际证据。仍需URMA/UBmem真机通过后验收 |
| bench | 同输入原生串行/融合完整图，mean/p50/p99、原始样本，保留可恢复数值失败的计时 | 合成生成器尚未拟合真实分布；其收益不能直接外推生产负载 |
| diagnose 阶段 | 融合完整图与现有逐核/步骤打点；新增合成串行图的 LI/LIDU/Get 名称归因 | 串行归因未真机确认，未知内核名明确报 INCOMPLETE；LIDU 内地址准备不单独虚构耗时；真实回放的串行阶段采集尚未接入 |
| diagnose 算法 | 需求、初始cold、预取、逐窗unused、逐Q补拉、实际字节和跨轮统计 | 缺少消费身份及公开窗口计数，不能精确归因 useful/redundant/refetched 或候选/预算抑制；完整补齐需额外诊断数据，当前不改算子 |
| diagnose 带宽 | 现有预取提交/CQ观察、最终Get调用区间的有效带宽，保留时间范围标签 | 最终Get不是严格首WQE到末CQ；只能比较相同口径，不能据此断言链路峰值 |
| Fetch-only | 合成 URMA 复用正式 worker，与KVAF对比单Q及全部Q | UBmem 和真实回放专项尚未接入；不阻止完整图计时，但缺少这两类独立归因 |
| real | schema-4 series 连续回放、独立状态、真实Q/Index、页共享重建、完整Index及hot KV校验 | 不支持其他旧snapshot格式；缺真实Q的数据需重新采集或先转换；LRU仅采集的候选前缀准确，其余按固定顺序初始化 |
| 拟合分析 | 离线统计与短日志，不上传原始Q/Index | 用户运行后回传统计，再迭代生成器并检查真实LI得到的分布；当前没有已拟合模型 |
| 脚本收敛 | 删除旧矩阵/transport对比和独立init入口，保留统一入口及离线分析 | support仍复用较多旧执行辅助模块，未彻底拆净内部历史参数；不暴露为正式命令 |

建议先在两种传输上分别运行 `check --suite basic --cases basic_bs1`，
确认安装和运行链路后，再跑完整check、bench和diagnose。首次真机运行发生的框架/API适配
错误属于本预览版待修项，不能把失败报告当作算子回归结论。


## check 完整验收

默认 check 包含 init、basic、boundary、stateful、contract；真实数据用例需显式提供
`--suite real --snapshot /path/to/captured-round/manifest.json`，不随默认合成检查伪造执行。
两种传输分别运行默认 check，返回码为零且所有用例 PASS 后，内置检查验收通过。
真实数据能力还需实际 snapshot 的 real suite 通过。

`boundary_regular` 的冷启动会触发预取目的槽冲突和 hot 池兜底：断言每请求有冲突
描述符，且实际最终目的槽至少一个不在输入的2048个LRU候选中。这里证明的是
冲突依赖处理分支，不承诺硬件执行时一定发生非零CQ等待。Padding清空全部Q，
断言零读取、输出为-1、持久Index/有效位/KV未改变。UBmem另外保留混合Q数量及
owner换序检查。覆盖证据自动打印并汇总到对应case的coverage.json。

本地只校验测试逻辑、输入/状态管理及报告；NPU运行结果由完整命令验证。

## 合成 bench 矩阵验收

暂沿用现有 Q/Index 生成器，不依赖 trace 或拟合结果。固定 Q4，串行与融合使用相同
负载；多 miss 与多 BS 组成笛卡尔积。每个 transport 单独运行，依赖目录和服务需对应。

```bash
python tests/operators/lifetch/test_lifetch.py bench \
  --transport urma --device 4 \
  --kvaf-root /vllm-workspace/vllm-ascend/third_party/KVAF \
  --kvaf-server 127.0.0.1:28885 \
  --batches 1,2,4,8,16 --miss-percent 3,5,10,15 \
  --warmup 4 --iterations 50 --timeout-seconds 3600
```

UBmem 使用同一命令，改为 `--transport ubmem` 并指定其 KVAF 目录和 CPU service。
每种传输20组配对结果，每组包含串行和融合完整图性能。首轮按需构建，后续复用
所选 KVAF 目录下的运行时 shape 构建缓存；不按 BS/miss 编译。
整个合成 bench 使用一个 worker，共享最大 BS 的 DDR 和 runtime；超时预算对整组矩阵生效。
DDR 分块 Put 一次，后续 case 不重复 Put；每个 case 仍独立创建并恢复 Index/hot/LRU/图。
URMA runtime 固定40个worker、每worker两条Jetty；UBmem固定2个worker。
这会改变旧测试按BS创建URMA runtime的资源配置，历史小BS串行性能不是严格同配置对照。
启动及结束打印共享池统计，`cases/bench/pool.json`保存Put字节数、复用次数和worker数。

运行结束自动输出大表及 `results.csv`、`summary.json`；每组保留原始计时数据。
汇总包含 transport、BS、目标miss%、正确性、mean/p50/p99及按mean计算的改善比例。
这里是算子链完整图，不是服务TPOT。实际miss和读取数量仍保留在各case报告中。
设备执行失败会停止后续任务；已完成结果保留，未完成项不能当作性能样本。
